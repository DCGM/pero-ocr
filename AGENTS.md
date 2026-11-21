# AGENTS.md

Rules for agents (and people) changing pero-ocr. The details are in `test/README.md`, `test/engines/consistency/README.md` and `test/io/consistency/README.md`; this file is the short version that must not be broken.

## Running the tests

    pip install -e ".[test]"
    pytest                        # everything that runs on CPU without models or external data (~60 s)
    pytest -m "not slow"          # without the parse_folder.py subprocess runs (~15 s)
    pytest test/<tier>            # one tier
    pytest -m gpu                 # GPU tests; skipped without CUDA

Every change keeps `pytest` green. A test that fails because of a deliberate change is updated in the same change, never deleted or loosened to make it pass.

## Where a test goes

The tiers follow the call stack. Each tier tests its own logic and treats the tier below as a black box.

| tier | tests | builds what it tests |
|---|---|---|
| `test/cli/` | `user_scripts/parse_folder.py` as a subprocess: arguments, input listing, outputs, skipping, errors, exit codes, full runs | runs the script |
| `test/page_parser/` | what `PageParser` adds: which sections run and in which order, numbered sections, the contracts between stages (crop height, alphabets, CTC logits), what it does after the stages, pages in a row, production-shaped pipelines | `PageParser(config)` |
| `test/stages/` | one kind of config section (`[LAYOUT_PARSER]`, `[LINE_CROPPER]`, `[OCR]`, `[DECODER]`): which engine it builds (`METHOD`, `TYPE`, unknown values), the contract all its engines keep, the logic around the engine (categories, thresholds, failing lines, devices, relative paths, writing results to lines) | always through the factory (`layout_parser_factory`, `line_cropper_factory`, `ocr_factory`, `page_decoder_factory`) |
| `test/engines/` | one engine as a black box with **all** its options: a layout method, the line cropper, an OCR engine, a decoder, the language model | the engine directly (from its own config section where it reads one) |
| `test/engines/consistency/` | one engine against frozen model, input and output | `frozen_engines.RUNNERS` |
| `test/io/` (+ `consistency/`) | PAGE XML, ALTO, logits: export, import, round trips; golden IO files | hand-built pages, `test/io/reference_layout.py` |
| `test/unit/` | internals with no config section of their own: algorithms and data structures the engines are built from | the function or class |

Deciding where a test goes:

- It drives an engine through its public interface: `engines/`, even if it checks exact numbers. Internals of that engine: `unit/`.
- It is about what a config section means independently of the engine behind it, or about choosing the engine: `stages/`. Every kind of section has its stage file, even when the stage has one engine or no logic of its own yet.
- It is about how stages are wired or what one stage hands to the next: `page_parser/`.
- An engine option is tested in the engine's file, not in a higher tier. Higher tiers test their own logic and a few full runs.
- Files mirror each other: `engines/test_<engine>.py` and `engines/consistency/test_<engine>.py`; every test file starts with a docstring saying what it covers.

## Tiny models instead of real ones

The default run uses no real models and no external data. `test/tiny_models.py` has small torchscript modules with the calling contract of the real networks (ParseNet, CTC, transformer, reading-order transformer, brnolm LM); their output is a deterministic function of the input, which `test/synthetic.py` draws (colours encode text, doubt, line ends, separators, reading order). Models loaded by third-party libraries are replaced by fakes instead (`test/fake_yolo.py`).

When a test needs a code path a tiny model cannot reach, upgrade the model in place (one class per network, never an old and a new one side by side) and regenerate its frozen data as described in `test/engines/consistency/README.md`.

## Consistency checks

`test/engines/consistency/` runs every engine with a **frozen** model on a **frozen** input and requires exactly the **frozen** output (floats to 1e-5). It catches models exported by older code that no longer load, and any change of an engine's output caused by the code or a library (torch, numpy, opencv, scipy, shapely). `test/io/consistency/` does the same for the output formats.

- **Every new engine gets both its engine tests and a consistency check**, unless it cannot have a meaningful one: its model is loaded by a third-party library (YOLO), or it is trivial and uses no library whose results could change (`REGION_WHOLE_PAGE`). Say why in `test/engines/consistency/README.md` when an engine has none.
- Adding a check: a tiny model in `tiny_models.py` (if the engine has a network), a runner in `frozen_engines.RUNNERS`, the frozen input in `data/regenerate.py` (`INPUTS`, `MODELS`), `test_<engine>.py` with `check('<engine>')`, the rows in the README, then `python test/engines/consistency/data/regenerate.py <engine>`. Run the engine as production configures it, and choose inputs that reach every step production runs.
- Regenerate only after a deliberate change, only the engines it concerns: `--outputs-only` when the output changes on purpose (the frozen model stays old), completely when the model format changes on purpose or a tiny model is upgraded. Never regenerate to make a failing check pass; find out what changed first.
- Frozen data must be reproducible: if the engine uses randomness, seed it in the runner and say why.

## Known bugs and odd behaviour

- A known bug gets a test that describes the correct behaviour, marked `@pytest.mark.xfail(reason=..., strict=True)`; the reason says what is wrong. The fix removes the marker in the same change.
- Every bug or odd behaviour found is recorded on GitHub (DCGM/pero-ocr): in its own issue, or in the plan issue #96, under "Bugs found by the test suite" (pinned by an `xfail`) or "Weird behaviour to fix or decide". Keep the test paths there current when tests move.
- Behaviour that is odd but not decided to be wrong is pinned as current behaviour only when it matters for reproducibility, with a docstring pointing to #96.
