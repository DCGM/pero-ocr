# Tests

    pip install -e ".[test]"
    pytest                       # everything that runs on CPU without external data (the default)
    pytest test/page_parser      # one tier
    pytest test/io/consistency test/engines/consistency   # only the consistency checks
    pytest -m slow               # only the slow tests (parse_folder.py end-to-end runs)
    pytest -m gpu                # GPU tests (skipped automatically without CUDA)

## Layout

| directory | what | needs |
|---|---|---|
| `cli/` | `user_scripts/parse_folder.py` end to end: files, flags, process behaviour | CPU, ~20 s |
| `page_parser/` | `PageParser` built from config: which stages run and in which order, numbered sections, what `PageParser` itself does with the result, and full production-like pipelines | CPU |
| `engines/` | every engine as a black box, with all its config options: the stages a config section builds (layout, line cropper, OCR, decoder, sorters, postprocessing) and the engines inside them, driven by tiny torchscript models; YOLO with a fake ultralytics | CPU |
| `engines/consistency/` | every engine, with a frozen model, still turns a frozen input into exactly the frozen output (the line cropper has no model) | CPU |
| `io/` | PAGE XML / ALTO / logits: export, import (also from other tools), round trips, known bugs | CPU |
| `io/consistency/` | the exporters still write the committed golden IO files, and the committed PAGE XML still reads back | CPU |
| `unit/` | single functions and classes in isolation, mirroring the `pero_ocr` package | CPU |

The tiers follow the call stack: `parse_folder.py` wraps `PageParser`, which wraps the engines, which produce the `PageLayout` that `io` exports.

Each tier tests its own logic and treats the tier below as a black box:

- An **engine** is everything one config section builds: a `[LAYOUT_PARSER]` stage selected by its `METHOD`, `[LINE_CROPPER]`, `[OCR]` or `[DECODER]`, together with the engine classes inside it. Its tests in `engines/` build it from a config section and cover every option it reads: a section and a `PageLayout` in, a `PageLayout` out.
- **`page_parser/`** tests what `PageParser` adds on top: which sections are built and run (`RUN_*`, section names and order, numbered croppers and OCRs), the `METHOD` dispatch, passing `config_path` on, what it does after the stages (confidences, `FILTER_CONFIDENT_LINES_THRESHOLD`), and a few full pipelines in the shape of the production engines. It does not test the options of a stage.
- **`cli/`** tests `parse_folder.py` itself: arguments, files, skipping, errors, exit codes, and full runs.

| config section | engine | tests |
|---|---|---|
| `[LAYOUT_PARSER] METHOD = LAYOUT_CNN` | `LayoutExtractor`, `LayoutEngine` | `test_layout_cnn.py`, `test_layout_engine.py` |
| `METHOD = LAYOUT_YOLO` | `LayoutExtractorYolo`, `LayoutEngineYolo` | `test_layout_yolo.py` |
| `METHOD = REGION_SORTER_SMART`, `REGION_SORTER_NAIVE` | `SmartRegionSorter`, `NaiveRegionSorter` | `test_region_sorters.py` |
| `METHOD = REGION_SORTER_TRANSFORMER` | `TransformerRegionSorter` | `test_region_sorter_transformer.py` |
| `METHOD = REGION_WHOLE_PAGE`, `LINE_POSTPROCESSING`, `LAYOUT_POSTPROCESSING` | `WholePageRegion`, `LinePostprocessor`, `LayoutPostprocessor` | `test_layout_postprocessing.py` |
| `[LINE_CROPPER]` | `LineCropper`, `EngineLineCropper` | `test_line_cropper.py` |
| `[OCR]` | `PageOCR`, `PytorchEngineLineOCR`, `TransformerEngineLineOCR` | `test_page_ocr.py`, `test_ctc_engine.py`, `test_transformer_engine.py` |
| `[DECODER]` | `PageDecoder`, the decoders, the language model | `test_page_decoder.py`, `test_lm.py` |

Not covered: `REGION_SIMPLE_THRESHOLD`, `LINES_SIMPLE_THRESHOLD` and `LINE_FILTER`, which may be dropped (issues #101 and #102).

The markers `gpu`, `models` (real models, `PERO_TEST_MODELS`) and `regression` (the external page sample, `PERO_TEST_DATA`) are deselected by default.

## Consistency checks

The general tests check what the code does now. The consistency checks compare the current code with committed snapshots of what earlier code produced, so they catch changes that would break files or models already out in the world:

| | `io/consistency/` | `engines/consistency/` |
|---|---|---|
| snapshot in `data/` | golden IO files: PAGE XML, ALTO and logits | per engine: frozen tiny torchscript model, frozen input, frozen output |
| guards | the output format (and reading PAGE XML back) | loading of previously exported models, and the exact output of every engine (text, logits, baselines, crops) |
| made from | hand-built pages in `io/reference_layout.py` | stand-ins in `tiny_models.py`; inputs defined in `data/regenerate.py` |
| regenerate | `python test/io/consistency/data/regenerate.py` | `python test/engines/consistency/data/regenerate.py [--outputs-only] [engine ...]` |
| regenerate when | the output format changes on purpose | an engine's output changes on purpose (`--outputs-only`), or the model format is broken on purpose (rarely) |

Each `data/` directory has its own `regenerate.py`, the only thing that writes to it. It recreates the data with the current code and records the commit and library versions in `generated_with.json` (one per engine, since engines can be frozen at different times). pytest only compares. Never regenerate just to make a failing check pass. Each directory's README explains its snapshot and what to do when a check fails.

## Tiny models

pero-ocr loads every network as a torchscript file, so the engine tests use small stand-ins (`tiny_models.py`) that follow the same calling contract and read synthetic bright-on-black images (`synthetic.py`) predictably. For example, a line crop with *n* bright blocks decodes to *n* characters, and the tiny language model prefers `b` after `a`. The stand-ins are built from the current code for each test session.

The YOLO layout engine (`LAYOUT_YOLO`) is the exception: ultralytics, not pero-ocr, loads its model, so a stand-in model file would only test ultralytics. `fake_yolo.py` replaces the `ultralytics.YOLO` class instead (fixture `fake_yolo`), returns fixed detections, and the tests need neither ultralytics nor a model. One test runs the real ultralytics with an untrained YOLOv8n to check that the fake matches it; it is skipped when ultralytics is not installed.

## Known bugs and limits

Tests for known bugs, and for known limits such as line shapes the cropper cannot straighten yet, are marked `xfail(strict=True)`, and the reason names the bug, the limit or the issue. When a bug is fixed or a limit lifted, its test starts passing, which fails the run under `strict`. Remove the marker in the same PR as the fix.
