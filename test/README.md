# Tests

    pip install -e ".[test]"
    pytest                       # everything that runs on CPU without external data (the default)
    pytest test/io               # one tier
    pytest -m slow               # only the slow tests (parse_folder.py end-to-end runs)
    pytest -m gpu                # GPU tests (skipped automatically without CUDA)

## Layout

| directory | what | needs |
|---|---|---|
| `unit/` | single functions and classes | CPU |
| `io/` | PAGE XML / ALTO / logits format contract: golden files and round trips | CPU |
| `engines/` | OCR and layout engines driven by tiny torchscript models | CPU |
| `cli/` | `user_scripts/parse_folder.py` end to end | CPU, ~20 s |
| `data/golden/` | reference exporter outputs | |
| `data/models/` | frozen tiny models: the backward-compatibility check for exported models | |

The markers `gpu`, `models` (real models, `PERO_TEST_MODELS`) and `regression` (the external page sample, `PERO_TEST_DATA`) are deselected by default.

## Tiny models

pero-ocr loads every network as a torchscript file, so the engine tests use small stand-ins (`tiny_models.py`) that follow the same calling contract and read synthetic bright-on-black images (`synthetic.py`) predictably. For example, a line crop with *n* bright blocks decodes to *n* characters. The stand-ins are built from the current code for each test session. `data/models/` holds copies frozen from an older version, to check that previously exported models keep working.

## Golden files

`io/` exports a hand-built `PageLayout` (`io/reference_layout.py`) and compares the result with `data/golden/`. Timestamps are masked. If a change to the output format is intended, regenerate the golden files and commit them together with the change, so that the format change shows up in the diff:

    pytest test/io --update-golden

## Known bugs

Tests for known bugs are marked `xfail(strict=True)`, and the reason names the bug or the issue. When a bug is fixed, its test starts passing, which fails the run under `strict`. Remove the marker in the same PR as the fix.
