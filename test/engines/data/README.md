# Frozen tiny models

The files in this directory are small, fake networks saved as torchscript. They stand in for real pero-ocr models in the tests. They have no trained weights; their output is a deterministic function of the input image.

## Why tests use fake models at all

pero-ocr's engines are mostly plumbing around a network: reading the engine json, finding and loading the checkpoint, padding and batching line crops, decoding, turning outputs into logits and confidences, and building the `PageLayout`. That plumbing is what breaks when code changes, and it is what these tests exercise.

Real models are a poor fit for that:

- They weigh hundreds of MB, and many of them are not public.
- They need a GPU to run in reasonable time.
- Their output is hard to predict, so a test can only check "something came out".

pero-ocr loads every network as a torchscript file and only calls it through a fixed interface: `forward(...)`, or `encode` / `adapt` / `decode_step` for the transformer. So a tiny module that implements the same interface goes through exactly the same engine code as a real model. It fits in a few KB, runs on CPU in milliseconds, and gives an output the test can predict exactly.

Real models are still needed to check recognition quality. That is the job of the separate GPU regression run on real data, not of these tests.

## How the stand-ins behave

Defined in `test/tiny_models.py`:

| model | interface used by | behaviour |
|---|---|---|
| `ctc/` | `PytorchEngineLineOCR` | every horizontal run of bright pixels in the line crop becomes one character, `characters[1]`; with an embedding id `e` it becomes `characters[1 + e % (n - 1)]` |
| `transformer/` | `TransformerEngineLineOCR` (`exported: true`) | a line with *n* bright runs decodes to *n* characters `characters[1], characters[2], ...`, then the end-of-sequence token |
| `parsenet/` | `LayoutEngine` (ParseNet) | a baseline at the bottom row of every bright block, constant line heights, no line endpoints and no region separators |

"Ink" means bright pixels because the engines pad crops and pages with zeros, which is black. Bright-on-black test images (`test/synthetic.py`) keep that padding inert.

Each model is saved twice, as `X` and `X.cpu`, because the engines append `.cpu` to the checkpoint path when they run on CPU.

## Why there are two copies: fresh and frozen

The test session builds the stand-ins **fresh** from the current code (fixtures in `test/conftest.py`), and most tests use those.

A model built from the current code can never notice that the current code stopped accepting models built by an older one. For example, a change to how the engine reads the json, a renamed method the engine calls, or a torch upgrade that cannot load older torchscript files would all break models already deployed in production, yet every test using fresh models would still pass.

The copies in this directory close that gap. They were generated once and committed. The `test_frozen_model_still_works` test in each engine's test file (`test/engines/test_ctc_engine.py`, `test_transformer_engine.py`, `test_layout_engine.py`) loads them with the current code and checks that they give the same results as the fresh ones. If that test fails, a model exported with an older pero-ocr or torch no longer works.

Generated from:

- pero-ocr: develop @ 81b496e
- torch: 2.14.1

## Regenerating

**Do not regenerate these files to make a failing test pass.** A failure here is the warning the files exist for.

Regenerate them only on purpose: after a deliberate, announced break of the model format, or when you add a new kind of stand-in. Commit the new files together with that change and update the versions above.

    python -m test.tiny_models test/engines/data
