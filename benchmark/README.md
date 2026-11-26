Benchmark
=========

Three separate steps. A benchmark is a folder of pages with its test lists; a registry names the engines to run on
one benchmark.

1. **Inference** (`infer.py --registry <registry>`) runs the registry's engines on its benchmark's pages with the code of
   the current commit and records what they produce in `$PERO_INFERENCE/<commit>/<benchmark>/`. It is not a test. A run fills in only the units that
   are missing, so engines can be added to any commit's records later (check out the commit, run `infer.py`).
2. **Benchmark** (`score.py`) scores one commit's records against the test lists: CER per engine and test list, and
   how well the layout engines find the annotated lines. Output: `<commit>/<benchmark>/score/`. For people who only care
   about the models.
3. **Consistency** (`test/inference/`) compares the records of two commits: the differences must stay within what two
   runs of the same commit differ by. This is the pero-ocr test. It needs no test lists.

Inputs, given as environment variables or options:

| variable | content |
|---|---|
| `--registry` | the registries to run (`.yaml` files, anywhere): each names its benchmark and the engines (below) |
| `PERO_TEST_DATA` | the data folder: one folder per benchmark (below) |
| `PERO_TEST_MODELS` | the engines: `<STAGE>/<name>/`, a `config.ini` with the sections of its stage and the files they refer to (`STAGE`: `LAYOUT_PARSER`, `LINE_CROPPER`, `OCR`, `DECODER`) |
| `PERO_INFERENCE` | the records, one folder per commit, one folder per benchmark in it |

The data, the engines and the registries are not public; the benchmarks' definition (lists, page metadata, how they
were made) and the registries are kept privately with the data.

A benchmark
-----------

```
<benchmark>/
  images/<page>.<ext>    the page images
  page_xml/<page>.xml    the input layout: the croppers crop its lines
  test_lists/            the references for score.py (not read by infer.py): <name>.test_list, one row per text line
  page_lists/            the pages of a test list, <name>.page_list, one row per page: a registry restricts a run
                         evaluated on a test list to the page list of the same name
  metadata.tsv           optional: where each page comes from (read by neither script)
```

A page is named by its file name. A test list has one line per row, `<page>-<line_id>.jpg <reference text>`.

A registry names its benchmark and the engines of each pero-ocr stage and what each one runs on. Registries record
which engines are benchmarked for a purpose (the official benchmark of every commit, a light variant, an experiment)
and are kept apart from the benchmarks and the engines; one benchmark can have several registries:

```yaml
BENCHMARK: example                                    # the benchmark: a folder of the data folder

LAYOUT_PARSER:
  - example.parsenet.2020-01-01

LINE_CROPPER:
  - interp2_scale1.15_height48

OCR:
  example.ctc.2020-01-01:
    LINE_CROPPER: interp2_scale1.15_height48          # the croppers whose line crops it reads

DECODER:
  example.lstm.2020-01-01:
    OCR: example.ctc.2020-01-01                       # the OCR engines whose output it decodes
```

- An engine's folder is `<STAGE>/<name>/` in the engines folder. The configs are used as they are; `infer.py` only
  sets the `RUN_*` switches of `[PAGE_PARSER]` to the engine's stage, and `TORCHSCRIPT_FUSER = no`: with
  TorchScript's fuser on (pero-ocr's default), a model's output on the GPU depends slightly (about 1e-5) on the pages
  processed before it in the same process, so a page would get different numbers in a full run, a resumed run and a
  top-up. A free run can leave it out (`--keep-torchscript-fuser`) to add to records made with the fuser on; a unit
  is only ever added to with the engine (files and config) it was recorded with; `--force-engine` adds pages anyway,
  for development, and records the change in the unit's `unit.json` (`engine_changes`: the files and config before, the
  pages added after). Other `[PAGE_PARSER]` settings (`FILTER_CONFIDENT_LINES_THRESHOLD`) stay in the config.
- A value is one name or a list. Any subset of the stages is valid.
- An OCR engine listed with several croppers runs with each of them; a decoder then names the run it decodes as
  `<ocr>@<cropper>`.
- A run is on every page, or on the pages of a page list given with its name (a path relative to the benchmark
  folder): a layout engine or cropper as `<name>: <page list>`, an OCR or decoder run as `<input>: <page list>` under
  its `LINE_CROPPER` or `OCR`:

  ```yaml
  DECODER:
    example.lstm.2020-01-01:
      OCR:
        printed.ctc.2020-01-01: page_lists/lm.printed.page_list   # only these pages
        handwritten.ctc.2020-01-01:                               # every page
  ```

  A run's pages must be among the pages of the run it reads (an OCR run's cropper, a decoder's OCR run, the run a
  threshold variant is derived from); otherwise the registry is refused. A unit records the pages it holds; giving it
  more pages later (a longer list, or no list) computes only the pages it does not hold and adds them.
- A unit saves its pages in chunks of `--chunk-pages` pages (25 by default: under ten minutes for the slowest units,
  the decoders): an interrupted run (a crash, a killed process) loses
  at most one chunk per unit, and running the same command again computes only the pages that are missing.
- The registry is checked before anything runs: every engine folder exists, every page list names pages of the
  benchmark, a run gets only pages its input run gets, a cropper gives lines of the height the
  OCR network takes, and the OCR engines a decoder decodes are CTC engines (not transformers).
- A decoder reads the logits of an OCR run with the OCR engine's alphabet (its `[OCR] OCR_JSON`), as it does in a
  config together with the OCR; its own `[OCR] OCR_JSON` is not used. `unit.json` records the OCR engine whose alphabet
  it used and its md5 (`alphabet_of`, `alphabet_md5`). Whether the language model fits that alphabet is not checked
  here: pero-ocr gives the language model the index of each OCR output, not the character, so the characters the
  language model knows must be the alphabet's first characters, in order.

Running it
----------

In the basis environment (`requirements/lock.txt`); GPU:

    python benchmark/infer.py --registry R.yaml --jobs 6   # everything of the registry not in the records yet
    python benchmark/infer.py --registry R.yaml --list     # the units and which are done
    python benchmark/score.py                              # score the records of the commit, every benchmark

Adding an engine (or pages of a run) is adding it to a registry and running the same command; only what is missing
runs. A unit whose engine was changed since (its files, e.g. a fixed model json) is not run again by itself: `--list`
marks it ENGINE CHANGED, and unit folders the registry no longer has as orphan; delete them and run again to fill
them in. A unit keeps one set of files per page, so computing some pages again is deleting their files
(`lines/<page>.*`, `logprobs/<page>.npz`, for a layout unit `page_xml/<page>.xml`) and running again; when the engine
changed since, add `--force-engine`, which `unit.json` records (`engine_changes`: which pages got the new engine).
`score.py` scores a unit on a test list when the unit holds all the list's pages, otherwise it reports NA.
Several registries (one per benchmark) can be run at once (`--registry R.yaml S.yaml`). `--units` restricts a run,
which is rarely needed.
`--jobs` layout and OCR units run at a time on a GPU, of them at most `--big-jobs` (1) big OCR units (models over 150 M
parameters, the mega transformer): several big models at once load the GPU to its limit (the machine of the first runs
crashed under three transformers). The other transformers (~110 M) and the CTC models (~17 M) run next to it.
`--decoder-jobs` (10) decoder units run at a time besides them, in all; each needs about one CPU core (the beam search)
and keeps a GPU busy too: its LM runs a small batch at every beam step, so one GPU serves only about two decoders at
full speed (an RTX A5000: one decoder keeps it 37 % busy, four 95 %). On a machine with several GPUs, `--gpus 0,1,2,3`
spreads the units over them (`--jobs` layout and OCR units on each, the decoders on the least loaded);
`--decoder-gpus` gives the decoders other GPUs. A unit process sees only its GPU (`CUDA_VISIBLE_DEVICES`).

Official and free runs:

- **Official** (the default): what a commit produces, for the published benchmark and the consistency tests. The
  working tree must be clean; the records go to `$PERO_INFERENCE/<commit>/`.
- **Free** (`--out <folder>`): the same run into any folder, not bound to the commit: another Python environment,
  uncommitted code, one's own engines or data, a trial on a few pages (`--pages N`). `<folder>/<benchmark>/run.json`
  records the commit, the uncommitted changes and the command of every run into it. Score it with
  `python benchmark/score.py --out <folder>`.

Within the records of one benchmark, official or free, results are added only from the same environment and the same
pages. Each benchmark's records are self-contained (nothing is stored at the level of the commit), so different
benchmarks of one commit can be made by different people on different machines and their folders put together.

- **Adopting units** (`--adopt-from <commit> --adopt LINE_CROPPER/ OCR/ DECODER/`): when a commit changes only some
  stages, its run copies the complete units of the other stages from the records of the earlier commit instead of
  computing them again, and computes only the rest. The source must have been inferred in the same environment and
  on the same pages (both checked, per benchmark); that the change does not touch the adopted stages is the caller's
  claim. `<benchmark>/adopted.json` records the source commit, the files changed in between and the units; the report
  marks them.

Every page is processed with numpy's and Python's random generators seeded from its name, so the layout engine's
random tie-breaks (issue #100) are the same in every run.

The records of a commit (a free run's folder has the same structure, plus `<benchmark>/run.json`):

```
<commit>/
  <benchmark>/
    environment.json                  the environment the benchmark's units were made in
    benchmark.json                    fingerprint of the pages (file names and hashes)
    adopted.json                      when units were adopted
    report.md                         the units
    LAYOUT_PARSER/<layout>/           page_xml/
    LINE_CROPPER/<cropper>/           crops.lmdb
    OCR/<ocr>@<cropper>/              lines/, logprobs/ (+ page_xml/, logits/ when decoded)
    DECODER/<lm>@<ocr>@<cropper>/     lines/
    logs/                             <unit>.log: every attempt of a unit appended, between a start and an end line
                                      (a start line without an end line: that attempt was killed or stopped)
    score/                            cer.tsv, layout.tsv, report.md
      lines/<OCR|DECODER>/<unit>/     <test list>.txt: the unit's output on the lines of the test list
```

Every unit has `unit.json` (the engine: its files and config; the runs). OCR and decoder units keep one set of files
per page (`benchmark/records.py`), the lines in the page's PAGE XML order:

- `lines/<page>.txt`: the lines in the format of the test lists, `<page>-<line_id>.jpg <text>`; a line the
  confidence filter removed has no text;
- `lines/<page>.tsv`: `line_id`, `confidence` (full precision), `kept` (0: the filter removed the line);
- `logprobs/<page>.npz` (OCR): the full log-probabilities of the page's lines in a fixed subset of lines (1 in 64, by
  a hash of `<page>/<line_id>`, the same in every unit), to compare the raw output of engines and environments; no file
  for a page with no line in the subset.

`score.py` writes `<benchmark>/score/lines/<kind>/<unit>/<list>.txt`, the unit's output on exactly the lines of a test
list in the list's order and format, so a test list and a unit (or two units, or two record sets) can be compared side
by side (e.g. `meld test_lists/<list>.test_list <benchmark>/score/lines/OCR/<unit>/<list>.txt`).

Nothing is computed twice: each cropper crops once and the OCR engines read its crops; an OCR engine that differs
from another only in a higher `FILTER_CONFIDENT_LINES_THRESHOLD` is derived from the other's run (its lines without
the lines of confidence at or below the threshold, as pero-ocr removes them); the decoders decode the stored output
of the OCR runs. That output (PAGE XML and logits of every page) is kept for the OCR runs a decoder decodes;
`python benchmark/infer.py --registry R.yaml --prune-logits` deletes the logits once all their decoders are done (a decoder added
later then runs the OCR engine again, and must get the recorded lines).

See the docstring of `infer.py` for what each unit records.
