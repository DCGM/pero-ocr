Benchmark
=========

Three separate steps:

1. **Inference** (`infer.py`) runs the benchmark models on the benchmark pages with the code of the current commit and
   records what they produce in `$PERO_INFERENCE/<commit>/`. It is not a test. A run fills in only the units that are
   missing, so models can be added to any commit's records later (check out the commit, run `infer.py`).
2. **Benchmark** (`score.py`) turns one commit's records into the scores of the models: CER per model and test list,
   and how well the layout models find the annotated lines. For people who only care about the models.
3. **Consistency** (`test/inference/`) compares the records of two commits: the differences must stay within what two
   runs of the same commit differ by. This is the pero-ocr test.

Inputs, given as environment variables or options:

| variable | content |
|---|---|
| `PERO_TEST_DATA` | the benchmark sample: `metadata.tsv`, `<access>/images/`, `<access>/page_xml/` (the input layout), `test_lists/` (the reference texts) |
| `PERO_TEST_MODELS` | the models: `models.tsv` (name, kind, folder, ...), `pairings.tsv` (which LM decodes which OCR model, as which service does it, on which lists), one folder per model |
| `PERO_INFERENCE` | the records, one folder per commit |

The data and the models are not public; the benchmark's definition (lists, page metadata, how they were made) is kept
in the ScribbleSense repository, `backend/tools/ocr_production_data/tst/<date>/`.

Running it (in the basis environment, `requirements/lock.txt`; GPU):

    python benchmark/infer.py --list              # the units and which are done
    python benchmark/infer.py --jobs 6            # everything missing, 6 units at a time
    python benchmark/infer.py --units ocr/universal.ctc.2025-03-19 --jobs 2

See the docstring of `infer.py` for what each unit records.
