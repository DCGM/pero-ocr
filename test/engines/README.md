# Engine tests

An engine does the actual work for one kind of config section: a layout method, the line cropper, an OCR network, a decoder, the language model. The tests here build it directly (from its own config section where it reads one) and cover every option it has; `consistency/` checks it against a frozen model, input and output (see `consistency/README.md`). Every engine gets both, unless it cannot have a meaningful consistency check; where a test is missing, its cell says why instead of naming a file.

| stage | engine | engine tests | consistency check |
|---|---|---|---|
| `[LAYOUT_PARSER]` | `METHOD = LAYOUT_CNN` (`LayoutExtractor` with `LayoutEngine`) | `test_layout_cnn.py`, `test_layout_engine.py` | `consistency/test_layout_cnn.py` |
| `[LAYOUT_PARSER]` | `METHOD = LAYOUT_YOLO` (`LayoutExtractorYolo` with `LayoutEngineYolo`) | `test_layout_yolo.py` | none: its model is loaded and run by ultralytics, whose model format and results pero-ocr does not control, and the tests replace ultralytics with a fake (`test/fake_yolo.py`) |
| `[LAYOUT_PARSER]` | `METHOD = REGION_SORTER_TRANSFORMER` (`TransformerRegionSorter`) | `test_region_sorter_transformer.py` | `consistency/test_region_sorter_transformer.py` |
| `[LAYOUT_PARSER]` | `METHOD = REGION_SORTER_SMART` (`SmartRegionSorter`, no model) | `test_region_sorters.py` | `consistency/test_smart_sorter.py` |
| `[LAYOUT_PARSER]` | `METHOD = REGION_SORTER_NAIVE` (`NaiveRegionSorter`) | `test_region_sorters.py` | none: its column clustering is a known bug (strict `xfail`s in `test_region_sorters.py`); freezing it would freeze the bug. Add a check with the fix |
| `[LAYOUT_PARSER]` | `METHOD = LINE_POSTPROCESSING` (`LinePostprocessor`, no model) | `test_line_postprocessing.py` | `consistency/test_line_postprocessing.py` |
| `[LAYOUT_PARSER]` | `METHOD = LAYOUT_POSTPROCESSING` (`LayoutPostprocessor`, no model) | `test_layout_postprocessing.py` | `consistency/test_layout_postprocessing.py` |
| `[LAYOUT_PARSER]` | `METHOD = REGION_WHOLE_PAGE` (`WholePageRegion`) | `test_region_whole_page.py` | none: trivial, one rectangle from the page size, no library underneath |
| `[LAYOUT_PARSER]` | `METHOD = REGION_SIMPLE_THRESHOLD` (`SimpleThresholdRegion`) | none: not covered by any test yet; it may be dropped (#101, #102) | none: as for the engine tests |
| `[LAYOUT_PARSER]` | `METHOD = LINES_SIMPLE_THRESHOLD` (`TextlineExtractorSimple`) | none: not covered by any test yet; it may be dropped (#101, #102) | none: as for the engine tests |
| `[LAYOUT_PARSER]` | `METHOD = LINE_FILTER` (`LineFilter`) | none: not covered by any test yet; it may be dropped (#101, #102) | none: as for the engine tests |
| `[LINE_CROPPER]` | `EngineLineCropper` (no model) | `test_line_cropper.py` | `consistency/test_line_cropper.py` |
| `[OCR]` | CTC, any other `METHOD` (`PytorchEngineLineOCR`) | `test_ctc_engine.py` | `consistency/test_ctc_engine.py` |
| `[OCR]` | `METHOD = pytorch_ocr-transformer` (`TransformerEngineLineOCR`) | `test_transformer_engine.py` | `consistency/test_transformer_engine.py` |
| `[DECODER]` | `TYPE = FAST-LOG-RAW` (`CTCPrefixLogRawNumpyDecoder`) with the language model (`construct_lm()`, `LMWrapper`) | `test_decoders.py`, `test_lm.py` | `consistency/test_lm.py` |
| `[DECODER]` | `TYPE = GREEDY` (`GreedyDecoder`) | `test_decoders.py` | none: trivial, the best class per frame |

Which engine a section builds, and the logic around the engine, are tested one tier up in `test/stages/`; the internals the engines are built from in `test/unit/`.
