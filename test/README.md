# Tests

    pip install -e ".[test]"
    pytest                       # everything that runs on CPU without external data (the default)
    pytest test/io               # one tier
    pytest test/io/consistency test/engines/consistency   # only the consistency checks
    pytest -m slow               # only the slow tests (parse_folder.py end-to-end runs)
    pytest -m gpu                # GPU tests (skipped automatically without CUDA)

## Layout

| directory | what | needs |
|---|---|---|
| `unit/` | single functions and classes | CPU |
| `io/` | PAGE XML / ALTO / logits: round trips, single features, known bugs | CPU |
| `io/consistency/` | the exporters still write the committed golden IO files, and the committed PAGE XML still reads back | CPU |
| `engines/` | OCR and layout engines driven by tiny torchscript models | CPU |
| `engines/consistency/` | models frozen from an older version still load and give the same results | CPU |
| `cli/` | `user_scripts/parse_folder.py` end to end | CPU, ~20 s |

The markers `gpu`, `models` (real models, `PERO_TEST_MODELS`) and `regression` (the external page sample, `PERO_TEST_DATA`) are deselected by default.

## Consistency checks

The general tests check what the code does now. The consistency checks compare the current code with committed snapshots of what earlier code produced, so they catch changes that would break files or models already out in the world:

| | `io/consistency/` | `engines/consistency/` |
|---|---|---|
| snapshot in `data/` | golden IO files: PAGE XML, ALTO and logits | frozen tiny torchscript models |
| guards | the output format (and reading PAGE XML back) | loading of previously exported models |
| made from | hand-built pages in `io/reference_layout.py` | stand-ins in `tiny_models.py` |
| regenerate | `pytest test/io/consistency --update-golden-io` | `pytest test/engines/consistency --update-frozen-models` |
| regenerate when | the output format changes on purpose | the model format is broken on purpose (rarely) |

Regenerating rewrites `data/` and records the commit and library versions in `data/generated_with.json`. Never regenerate just to make a failing check pass. Each directory's README explains its snapshot and what to do when a check fails.

## Tiny models

pero-ocr loads every network as a torchscript file, so the engine tests use small stand-ins (`tiny_models.py`) that follow the same calling contract and read synthetic bright-on-black images (`synthetic.py`) predictably. For example, a line crop with *n* bright blocks decodes to *n* characters. The stand-ins are built from the current code for each test session.

## Known bugs

Tests for known bugs are marked `xfail(strict=True)`, and the reason names the bug or the issue. When a bug is fixed, its test starts passing, which fails the run under `strict`. Remove the marker in the same PR as the fix.
