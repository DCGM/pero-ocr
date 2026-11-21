# Tests

The rules for where a test goes and when an engine needs a consistency check are summarized in `AGENTS.md` at the repository root.

    pip install -r requirements/lock.txt && pip install --no-deps -e .   # the basis environment
    pytest                       # everything that runs on CPU without external data (the default)
    pytest test/stages           # one tier
    pytest test/io/consistency test/engines/consistency   # only the consistency checks
    pytest -m slow               # only the slow tests (parse_folder.py end-to-end runs)
    pytest -m gpu                # GPU tests (skipped automatically without CUDA)

The basis environment (`requirements/lock.txt`) is one set of versions that installs on Python 3.8 to 3.10 and gives the same results as every live worker stack (issue #94). `pip install -e ".[test]"` works too, with whatever versions `pyproject.toml` allows; the consistency checks then tell whether those versions change any output. Frozen data and golden files are regenerated only in the basis environment.

## Layout

| directory | what | needs |
|---|---|---|
| `cli/` | `user_scripts/parse_folder.py` end to end: files, flags, process behaviour | CPU, ~20 s |
| `page_parser/` | `PageParser` built from config: which sections run and in which order, numbered sections, what `PageParser` itself does with the result, and full production-like pipelines | CPU |
| `stages/` | every kind of config section as a stage: which engine a section builds, and what the stage does around the engine that is the same for all its engines | CPU |
| `engines/` | every engine as a black box, with all its options, driven by tiny torchscript models; YOLO with a fake ultralytics | CPU |
| `engines/consistency/` | every engine, with a frozen model, still turns a frozen input into exactly the frozen output (the smart sorter and the line cropper have no model) | CPU |
| `io/` | PAGE XML / ALTO / logits: export, import (also from other tools), round trips, known bugs | CPU |
| `io/consistency/` | the exporters still write the committed golden IO files, and the committed PAGE XML still reads back | CPU |
| `unit/` | internals in isolation: algorithms and data structures with no config section of their own (decoder internals, hypotheses, confusion networks, confidences, alignments, the layout data model), mirroring the `pero_ocr` package | CPU |

The tiers follow the call stack: `parse_folder.py` wraps `PageParser`, which runs the stages, which wrap the engines, which produce the `PageLayout` that `io` exports.

Each tier tests its own logic and treats the tier below as a black box:

- An **engine** does the actual work for one kind of section: a layout method, the line cropper, an OCR network, a decoder, the language model. Its tests in `engines/` build it directly (from its own config section where it reads one, as the layout methods do) and cover every option it has.
- A **stage** is what one kind of config section builds: `[LAYOUT_PARSER]` (`layout_parser_factory`), `[LINE_CROPPER]` (`line_cropper_factory`, `LineCropper`), `[OCR]` (`ocr_factory`, `PageOCR`) and `[DECODER]` (`page_decoder_factory`, `PageDecoder`). Its tests in `stages/` always go through the factory and cover which engine a section builds (`METHOD`, `TYPE`, unknown values), the contract every engine of the stage keeps, and the logic around the engine: which lines it handles (`CATEGORIES`, confidence thresholds), failing lines, devices, paths relative to the config, writing results to the lines. Some stages have one engine or no logic of their own yet (the layout stage only picks the method, the cropper stage has a single engine); they get their file anyway, so that a stage that grows already has its tests in place.
- **`page_parser/`** tests what `PageParser` adds on top: which sections are built and run (`RUN_*`, section names and order, numbered croppers and OCRs), passing `config_path` on, what it does after the stages (confidences, `FILTER_CONFIDENT_LINES_THRESHOLD`), and a few full pipelines in the shape of the production engines.
- **`cli/`** tests `parse_folder.py` itself: arguments, files, skipping, errors, exit codes, and full runs.
- **`unit/`** tests what has no config section and is not an engine: the internals that the engines and stages are built from. A test that drives an engine through its public interface belongs to `engines/`, even if it checks exact numbers.

| config section | stage (`stages/`) | engine | engine tests (`engines/`) | consistency check (`engines/consistency/`) |
|---|---|---|---|---|
| `[LAYOUT_PARSER] METHOD = LAYOUT_CNN` | `test_layout_stage.py` | `LayoutExtractor`, `LayoutEngine` | `test_layout_cnn.py`, `test_layout_engine.py` | `test_layout_cnn.py` |
| `METHOD = LAYOUT_YOLO` | | `LayoutExtractorYolo`, `LayoutEngineYolo` | `test_layout_yolo.py` | none: ultralytics loads the model |
| `METHOD = REGION_SORTER_SMART`, `REGION_SORTER_NAIVE` | | `SmartRegionSorter`, `NaiveRegionSorter` | `test_region_sorters.py` | `test_smart_sorter.py` (smart only; the naive sorter after its bug is fixed) |
| `METHOD = REGION_SORTER_TRANSFORMER` | | `TransformerRegionSorter` | `test_region_sorter_transformer.py` | `test_region_sorter_transformer.py` |
| `METHOD = REGION_WHOLE_PAGE` | | `WholePageRegion` | `test_region_whole_page.py` | none: trivial |
| `METHOD = LINE_POSTPROCESSING` | | `LinePostprocessor`, `PostprocessingEngine` | `test_line_postprocessing.py` | `test_line_postprocessing.py` |
| `METHOD = LAYOUT_POSTPROCESSING` | | `LayoutPostprocessor` (`layout_helpers.retrace_region`) | `test_layout_postprocessing.py` | `test_layout_postprocessing.py` |
| `[LINE_CROPPER]` | `test_cropper_stage.py` | `EngineLineCropper` | `test_line_cropper.py` | `test_line_cropper.py` |
| `[OCR]` | `test_ocr_stage.py` | `PytorchEngineLineOCR`, `TransformerEngineLineOCR` | `test_ctc_engine.py`, `test_transformer_engine.py` | `test_ctc_engine.py`, `test_transformer_engine.py` |
| `[DECODER]` | `test_decoder_stage.py` | `GreedyDecoder`, `CTCPrefixLogRawNumpyDecoder`, the language model (`construct_lm`, `LMWrapper`) | `test_decoders.py`, `test_lm.py` | `test_lm.py` (FAST-LOG-RAW with the LM) |

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

pero-ocr loads every network as a torchscript file, so the tests use tiny models (`tiny_models.py`) that follow the same calling contract and read synthetic images (`synthetic.py`) predictably. Ink is bright on black; colours encode what a real network would recognize: a block of ink is one character for the OCR models (its colour picks the character and how sure the model is), white blocks are text lines for ParseNet (yellow marks line ends, red bars separate regions), grey marks in the page image set the order for the reading-order model, and the language model prefers `b` after `a`. `text_line()`, `draw_text()`, `layout_page()` and `mark_reading_order()` draw such inputs. The tiny models are built from the current code for each test session; frozen copies are committed for the consistency checks.

The YOLO layout engine (`LAYOUT_YOLO`) is the exception: ultralytics, not pero-ocr, loads its model, so a stand-in model file would only test ultralytics. `fake_yolo.py` replaces the `ultralytics.YOLO` class instead (fixture `fake_yolo`), returns fixed detections, and the tests need neither ultralytics nor a model. One test runs the real ultralytics with an untrained YOLOv8n to check that the fake matches it; it is skipped when ultralytics is not installed.

## Adding an engine

1. Engine tests in `engines/test_<engine>.py`: the engine built directly, every option it has, as a black box.
2. If its config section kind is new, a stage file in `stages/`; otherwise add it to the stage's `METHOD` / `TYPE` dispatch tests.
3. A consistency check in `engines/consistency/test_<engine>.py`, with a tiny model if it has a network, run as production would configure it. Engines without a meaningful one (a model loaded by a third-party library, or trivial code with no library underneath) are listed with the reason in `engines/consistency/README.md`.
4. The row in the table of config sections above, and a production-shaped pipeline in `page_parser/` if production will use it.

## Known bugs and limits

Tests for known bugs, and for known limits such as line shapes the cropper cannot straighten yet, are marked `xfail(strict=True)`, and the reason names the bug, the limit or the issue. When a bug is fixed or a limit lifted, its test starts passing, which fails the run under `strict`. Remove the marker in the same PR as the fix.
