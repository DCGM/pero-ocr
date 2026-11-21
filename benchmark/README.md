Benchmark
=========

Three separate steps, each with its own output folder under the commit:

1. **Inference** (`infer.py`) runs the engines on the pages with the code of the current commit and records what they
   produce in `$PERO_INFERENCE/<commit>/inference/<dataset>/`. It is not a test. A run fills in only the units that
   are missing, so engines can be added to any commit's records later (check out the commit, run `infer.py`).
2. **Benchmark** (`score.py`) scores one commit's records against the test lists: CER per engine and test list, and
   how well the layout engines find the annotated lines. Output: `<commit>/score/<dataset>/`. For people who only care
   about the models.
3. **Consistency** (`test/inference/`) compares the records of two commits: the differences must stay within what two
   runs of the same commit differ by. This is the pero-ocr test. It needs no test lists.

Inputs, given as environment variables or options:

| variable | content |
|---|---|
| `PERO_TEST_DATA` | the data folder: one folder per dataset (below) |
| `PERO_TEST_MODELS` | the engines: `<STAGE>/<name>/`, a `config.ini` with the sections of its stage and the files they refer to (`STAGE`: `LAYOUT_PARSER`, `LINE_CROPPER`, `OCR`, `DECODER`) |
| `PERO_INFERENCE` | the records, one folder per commit |

The data and the engines are not public; the benchmark's definition (lists, page metadata, how they were made) is
kept in the ScribbleSense repository, `backend/tools/ocr_production_data/tst/<date>/`.

A dataset
---------

```
<dataset>/
  registry.yaml          which engines run on it
  images/<page>.<ext>    the page images
  page_xml/<page>.xml    the input layout: the croppers crop its lines
  test_lists/            the references for score.py (not read by infer.py)
  metadata.tsv           optional: where each page comes from (read by neither script)
```

A page is named by its file name. A test list has one line per row, `<page>-<line_id>.jpg <reference text>`.

The registry names the engines of each pero-ocr stage and what each one runs on:

```yaml
LAYOUT_PARSER:
  - universal.parsenet.2022-04-13

LINE_CROPPER:
  - interp2_scale1.15_height48

OCR:
  universal.ctc.2025-03-19:
    LINE_CROPPER: interp2_scale1.15_height48          # the croppers whose line crops it reads

DECODER:
  english.lstm.wikitext103.2021-06-28:
    OCR: universal.ctc.2025-03-19                     # the OCR engines whose output it decodes
```

- An engine's folder is `<STAGE>/<name>/` in the engines folder. The configs are used as they are; `infer.py` only
  sets the `RUN_*` switches of `[PAGE_PARSER]` to the engine's stage. Other `[PAGE_PARSER]` settings
  (`FILTER_CONFIDENT_LINES_THRESHOLD`) stay in the config.
- A value is one name or a list. Any subset of the stages is valid. Every engine runs on every page of the dataset.
- An OCR engine listed with several croppers runs with each of them; a decoder then names the run it decodes as
  `<ocr>@<cropper>`.
- The registry is checked before anything runs: every engine folder exists, a cropper gives lines of the height the
  OCR network takes, a decoder's `[OCR] OCR_JSON` alphabet is the alphabet of every OCR engine it decodes, and those
  are CTC engines (not transformers).

Running it
----------

In the basis environment (`requirements/lock.txt`); GPU:

    python benchmark/infer.py --jobs 6            # every dataset: everything in its registry not in the records yet
    python benchmark/infer.py --list              # the units and which are done
    python benchmark/score.py                     # the benchmark of the records

Adding an engine is adding it to a registry and running the same command; only the missing units run.
`--datasets` and `--units` restrict a run, which is rarely needed.

The records of a commit:

```
<commit>/
  inference/
    environment.json, report.md
    <dataset>/
      dataset.json                      fingerprint of the pages (file names and hashes)
      LAYOUT_PARSER/<layout>/           page_xml/
      LINE_CROPPER/<cropper>/           crops.lmdb
      OCR/<ocr>@<cropper>/              lines.tsv, logprobs.npz (+ page_xml/, logits/ when decoded)
      DECODER/<lm>@<ocr>@<cropper>/     lines.tsv
      logs/
  score/
    <dataset>/                          cer.tsv, layout.tsv, report.md
```

Nothing is computed twice: each cropper crops once and the OCR engines read its crops; an OCR engine that differs
from another only in a higher `FILTER_CONFIDENT_LINES_THRESHOLD` is derived from the other's run (its lines without
the lines of confidence at or below the threshold, as pero-ocr removes them); the decoders decode the stored output
of the OCR runs. That output (PAGE XML and logits of every page) is kept for the OCR runs a decoder decodes;
`python benchmark/infer.py --prune-logits` deletes the logits once all their decoders are done (a decoder added
later then runs the OCR engine again, and must get the recorded lines).

See the docstring of `infer.py` for what each unit records.
