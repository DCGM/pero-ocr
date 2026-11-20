# Engine consistency checks: frozen model, input and output

Every engine has the same kind of check here: a **frozen model**, a **frozen input** and the **frozen output** that an earlier version of pero-ocr produced, all committed. The check runs the current code with the frozen model on the frozen input and requires the frozen output. So it fails when:

- a model **exported by an older pero-ocr, torch or brnolm** no longer loads or behaves differently, or
- the **engine code or a library** changes what the engine returns, even slightly.

The checks are small and run in about a second, so they show such changes long before a run on real data does.

| file | what |
|---|---|
| `test_ctc_engine.py` | `PytorchEngineLineOCR`: frozen CTC model, frozen line crops → frozen text, logits and logit coordinates |
| `test_transformer_engine.py` | `TransformerEngineLineOCR`: frozen transformer, frozen line crops → frozen text, logits and logit coordinates, with greedy decoding and beam search |
| `test_layout_engine.py` | `LayoutEngine`: frozen ParseNet, frozen pages → frozen regions, baselines and heights |
| `test_lm.py` | `construct_lm()`, `LMWrapper`, the `FAST-LOG-RAW` decoder: frozen language model, frozen CTC frames → frozen next-character probabilities and decoded text, with LM weights 1.0 and 0.4 |
| `test_region_sorter.py` | `TransformerRegionSorter` (`REGION_SORTER_TRANSFORMER`): frozen reading-order model, frozen region boxes → frozen reading order |
| `test_smart_sorter.py` | `SmartRegionSorter` (`REGION_SORTER_SMART`, no model): frozen region layouts → frozen reading order and the geometry it returns |
| `test_line_cropper.py` | `EngineLineCropper` (no model): frozen page and lines → frozen crops, mappings and `blend_in()` result |
| `frozen_engines.py` | what the checks and the script share: how each engine is run on its frozen input (`RUNNERS`), how the data is read and written, and how outputs are compared |
| `data/<engine>/` | the committed frozen data |
| `data/regenerate.py` | the script that writes `data/`; all code used only for regenerating is in it, including the frozen inputs (the models themselves are in `test/tiny_models.py`, because the engine tests build fresh copies of them in every session) |

The general engine tests are one level up in `test/engines/`.

The YOLO layout engine (`LAYOUT_YOLO`) has no check here: its model is loaded and run by ultralytics, whose model format and results pero-ocr does not control, and the tests replace ultralytics with a fake (`test/fake_yolo.py`). What pero-ocr does with the detections is covered by `test/engines/test_layout_yolo.py`.

## Why

The engine tests one level up run on tiny stand-in models that the test session builds **fresh** from the current code, and they check behaviour with expectations written in the test. That leaves two gaps:

- A model built from the current code can never notice that the current code stopped accepting models built by an older one. A change to how the engine reads the json, a renamed method the engine calls, or a torch upgrade that cannot load older torchscript files would all break models already deployed in production, yet every test using fresh models would still pass.
- An expectation written in a test checks what its author thought of, often with a tolerance. A change in the exact numbers an engine returns goes unnoticed, although on real pages it changes the recognized text. For example, opencv 5 changed `cv2.remap` so that about 6 % of the line-crop pixels differ by up to 3 grey levels, which changed 0.1 to 0.5 % of the characters on real pages; the general cropper tests still pass with opencv 5.

The frozen data closes both gaps: the models act like models exported in the past, and the outputs record exactly what the engines returned in the past.

## What is in `data/`

Each `data/<engine>/` holds:

| file | what |
|---|---|
| the model | small, fake networks saved as torchscript by `test/tiny_models.py` (see below); the smart sorter and the cropper have none |
| `input.npz` | the frozen input arrays: line crops, pages, CTC frames |
| `input.json` | the parameters of the run: beam sizes, layout-engine settings, LM weights, the cropper's lines and settings |
| `output.npz` | what the engine returned for that input, when the data was frozen |
| `generated_with.json` | the commit and the versions of Python, torch, numpy, opencv, scipy and shapely that produced the data (data frozen before shapely was recorded lacks it) |

The tiny models have no trained weights; their output is a deterministic function of the input:

| engine | model | behaviour |
|---|---|---|
| `ctc/` | `model.pt`, `ocr_engine.json` | every horizontal run of bright pixels in the line crop becomes one character, `characters[1]`; with an embedding id `e` it becomes `characters[1 + e % (n - 1)]` |
| `transformer/` | `model.pt`, `ocr_engine.json` (`exported: true`) | a line with *n* bright runs decodes to *n* characters `characters[1], characters[2], ...`, then the end-of-sequence token |
| `parsenet/` | `parsenet.pt` | a baseline at the bottom row of every bright block, constant line heights, no line endpoints and no region separators |
| `region_sorter/` | `model-cuda.pt`, `model-cpu.pt` | a reading-order model with the calling contract of the real one (region boxes, query types, page image → scores for "region *i* comes right after region *j*"); it orders regions by their left edge, then by their top edge, and ignores the image |
| `lm/` | `lm.pt` | a brnolm torchscript LM (zip of `model.zip`, `decoder.zip`, `vocabulary.pickle`); its state is the last character seen, kept over line ends; after `a` it strongly prefers `b`, otherwise uniform |
| `smart_sorter/` | none | the input is six region layouts with lines, in a scrambled order: a title over two columns whose gaps do not line up, a small overlap that `FakeIntersectionParameter` ignores, the same title and columns tilted by 4 degrees (the sorter rotates the page to sort it and back), a newspaper-like page, a pinwheel that no straight cut divides (the fallback ordering) and a single region; default config, as in production |
| `cropper/` | none | the input is a synthetic page with text-like strokes and lines cut from it with the settings pero-ocr uses: the production `LineCropper` (`LINE_HEIGHT 48`, `INTERP 2`, `LINE_SCALE 1.15`) on straight lines at several angles, a gentle arc, a short line and a line partly outside the page; the ALTO word alignment (height 32, `INTERP 2`, with its forward mapping); the baseline refiner (height 32, `INTERP 0`); and `blend_in()` |

The OCR and layout models are saved twice, as `X` and `X.cpu`, because those engines append `.cpu` to the checkpoint path when they run on CPU. The reading-order sorter instead replaces `cuda` with `cpu` in the file name, hence `model-cuda.pt` and `model-cpu.pt`. The LM loader does neither.

The inputs are reasonable cases that the current code handles. Cases it cannot handle yet are known bugs or limits with strict `xfail` tests in the general tests, for example line shapes the cropper cannot straighten (`test/engines/test_line_cropper.py`). When the code learns to handle them, add them here as new cases.

The inputs are committed, not generated in the check, so that a check depends only on the engine it checks. The cropper's page, for example, is drawn with opencv, the very library whose changes its check has to catch.

## How outputs are compared

Text, integers and crops (uint8) must match **exactly**. Floats (logits, baselines, log-probabilities) must match up to a relative and absolute tolerance of 1e-5 (`FLOAT_TOLERANCE` in `frozen_engines.py`): that ignores the last bits of float arithmetic, which can differ between torch builds, but not any change of behaviour. With numpy 2 instead of 1.24 all checks pass; with opencv 5 the cropper check fails.

The frozen outputs record **current** behaviour, not ideal behaviour. For example, the LM output with weight 0.4 includes the known bug that `BagOfHypotheses.best_hyp()` ignores the LM weight. A fix of such a bug changes the output on purpose, and the frozen output is then regenerated with `--outputs-only` (see below).

## Why tiny models

pero-ocr's engines are mostly plumbing around a network: reading the engine json, finding and loading the checkpoint, padding and batching line crops, decoding, turning outputs into logits and confidences, and building the `PageLayout`. That plumbing is what breaks when code changes. Real models are a poor fit for testing it:

- They weigh hundreds of MB, and many of them are not public.
- They need a GPU to run in reasonable time.
- Their output is hard to predict, so a test can only check "something came out".

pero-ocr loads every network as a torchscript file and only calls it through a fixed interface: `forward(...)`, `encode` / `adapt` / `decode_step` for the transformer, or `model(x, h)` / `init_hidden()` / `decoder(o)` for the LM. So a tiny module that implements the same interface goes through exactly the same engine code as a real model. It fits in a few KB, runs on CPU in milliseconds, and gives an output a test can predict exactly.

"Ink" means **bright** pixels, so the test images (`test/synthetic.py`) are bright text on black, the inverse of real documents. The reason is padding: the engines fill everything that is not image with zeros, which is black. The OCR engines pad shorter lines of a batch to the width of the longest one, the layout engine pads the page to a size divisible by its downsampling, and the line cropper fills the part of a line outside the page with black. Real models learned to ignore that padding; the stand-ins cannot learn. If dark pixels were ink, the padding would read as text: extra characters at the end of every line shorter than its batch, a baseline along the padded page border, and outputs that depend on how lines happen to be batched. With bright ink, padding is simply "no ink", and the expected outputs stay exact.

Real models are still needed to check recognition quality. That is the job of the separate GPU regression run on real data, not of these checks.

## Regenerating

Regenerate **only on purpose**, name only the engines the change concerns, and commit the data together with the change that needs it:

    python test/engines/consistency/data/regenerate.py --outputs-only cropper   # the output changed on purpose
    python test/engines/consistency/data/regenerate.py lm                       # a new engine or stand-in, or a deliberate break of the model format
    python test/engines/consistency/data/regenerate.py                          # everything

- **The output changes on purpose** (a fixed bug, a new logits format, a better cropper): use `--outputs-only`. It keeps the frozen model and input and recomputes only `output.npz`, so the model stays old and keeps checking that old models still load.
- **The model format breaks on purpose** (models have to be re-exported): regenerate the engine completely. After that, its check compares the current code with itself until the model is old again, so anything the old model exercised is no longer protected.
- **A new engine or stand-in**: add a runner to `frozen_engines.py` and an input to `regenerate.py`, then regenerate only that one.

The script records the commit and library versions in each directory's `generated_with.json`. It works from any directory and always uses the `pero_ocr` of the checkout it lives in. pytest itself never writes to `data/`.

## When a check fails

1. Read the failure: it lists every output that changed, with the largest difference.
2. If the change is **not intended**, fix the code, or pin the library version that caused it (e.g. opencv): deployed models depend on the old behaviour.
3. If it **is intended**:
   - a model no longer loads: this breaks every deployed model of that kind. Announce it, make sure the production models are re-exported, then regenerate the engine;
   - the output changed: check what it does on real pages first. The models were trained on the old crops and tuned with the old decoding, so compare the recognition accuracy on the consistency page sample before and after. Then regenerate with `--outputs-only`.

Never regenerate the data just to make a failing check pass.
