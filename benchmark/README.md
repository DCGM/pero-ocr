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
| `PERO_TEST_MODELS` | the registry: `models.tsv`, one row per model (name, kind, folder; for an LM also the OCR models whose results it decodes, how the service runs it, on which lists), and one folder per model |
| `PERO_INFERENCE` | the records, one folder per commit |

The data and the models are not public; the benchmark's definition (lists, page metadata, how they were made) is kept
in the ScribbleSense repository, `backend/tools/ocr_production_data/tst/<date>/`.

Running it (in the basis environment, `requirements/lock.txt`; GPU):

    python benchmark/infer.py --jobs 6            # everything in the registry that is not in the records yet
    python benchmark/infer.py --list              # the units and which are done
    python benchmark/score.py                     # the benchmark of the records

Adding a model is adding its row to `models.tsv` and running the same command; only the missing units run, on any
commit whose records are kept (check out the commit first). `--units` restricts a run to some units, which is rarely
needed.

- OCR or layout model: `name`, `kind`, `folder`. An OCR model runs with the `[LINE_CROPPER]` of its config; more cropper
  setups to try go to `croppers` (comma-separated, e.g. `interp2_scale1.25_height48`; the height must be the model's),
  each giving the unit `ocr/<name>@<setup>`.
- Language model: also `decodes` (the OCR runs whose results it decodes, comma-separated: `<name>` or
  `<name>@<setup>`), `lm_style` (`scribble`:
  a separate stage with the LM package's alphabet, as ScribbleSense runs it; `pero`: OCR and LM in one config with
  the OCR model's alphabet, as the pero-ocr web runs it) and `lm_lists` (`*`, `A,B` or `-A,B`: the pages it decodes).
  An LM is benchmarked on OCR results; the OCR results themselves are already in the records and are not computed
  again.

Nothing is computed twice: the OCR models read the crops of the crops units; a model that differs from another only
in a higher `FILTER_CONFIDENT_LINES_THRESHOLD` is derived from the other's run; the LMs decode the stored output of the
OCR runs. That output (PAGE XML and logits of every page) is kept for the OCR models an LM decodes;
`python benchmark/infer.py --prune-logits` deletes the logits once all LMs of a model are done (an LM added later then
runs its OCR model again on its pages, and must get the recorded lines).

See the docstring of `infer.py` for what each unit records.
