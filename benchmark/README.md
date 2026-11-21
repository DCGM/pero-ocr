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

Adding models to the records of a commit (check out the commit, then run `infer.py`; only the new units run):

- an OCR or layout model: a row in `models.tsv` and its folder;
- a language model: a row in `models.tsv`, its folder, and a row in `pairings.tsv` for every OCR model it is to decode
  (an LM alone is not benchmarked; the OCR model's own results are already there). The LM unit decodes the OCR
  output stored by the OCR unit; where it is not stored (pruned, or the OCR model had no LM pair when it ran) the OCR
  model runs again on the pages of the pairing and must give the recorded lines.

The OCR units of models with an LM pair keep their full output (PAGE XML and logits of every page) for the LM units.
`python benchmark/infer.py --prune-logits` deletes the logits once all LM pairs of a model are done.
