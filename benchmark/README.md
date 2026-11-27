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
  is only ever added to with the engine (files and config) it was recorded with; `--force-model` adds pages anyway,
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
- A decoder reads the logits of an OCR run with the OCR engine's char set (its `[OCR] OCR_JSON`), as it does in a
  config together with the OCR; its own `[OCR] OCR_JSON` is not used (the OCR engine is in the decoder unit's name,
  its files in the OCR unit's `engine.json`). Whether the language model fits that char set is
  not checked here: pero-ocr gives the language model the index of each OCR output, not the character, so the characters the
  language model knows must be the char set's first characters, in order.

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
(`lines/<page>.tsv` and the page's outputs, for a layout unit `page_xml/<page>.xml`) and running again; when the engine
changed since, add `--force-model`, which `unit.json` records (`engine_changes`: which pages got the new engine).
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

Within the records of one benchmark, official or free, results are added only under the same conditions; each guard
can be forced by its flag, and every forced addition is noted in `<benchmark>/forced.json` (what differed, when,
where):

| guard | what must be the same | flag |
|---|---|---|
| environment | the machine (hostname) and the environment (Python, packages, CUDA, cuDNN, GPU, GPU driver) of `environment.json`, and the pages of `benchmark.json` | `--force-environment` |
| commit | official records: the folder is the commit (a clean checkout), so other code never adds to them; a free run: the commit and uncommitted changes of its first run (`run.json`) | `--force-commit` |
| model | a unit: its model files and config (`engine.json`) | `--force-model` |

A run holds `<benchmark>/.lock` while it adds to the records: a
second run on the same records (by mistake, or another person) stops and says who holds it; `--list` needs no lock.
The lock is the operating system's lock on that file, released however the run ends (a crash or a kill leaves no
stale lock); the file is deleted when the run ends normally, and one left by a crash only says who ran last.
A stopped or crashed run is resumed with the same command: the chunks it finished are kept, the chunk in progress is
computed again. Each benchmark's records are self-contained (nothing is stored at the level of the commit), so different
benchmarks of one commit can be made by different people on different machines and their folders put together.

- **Adopting units** (`--adopt-from <commit> --adopt LINE_CROPPER/ OCR/ DECODER/`): when a commit changes only some
  stages, its run copies the complete units of the other stages from the records of the earlier commit instead of
  computing them again, and computes only the rest. The source must have been inferred on the same machine, in the
  same environment and
  on the same pages (both checked, per benchmark); that the change does not touch the adopted stages is the caller's
  claim. `<benchmark>/adopted.json` records the source commit, the files changed in between and the units; the report
  marks them.

Every page is processed as if it were the first one: before each page, the state pero-ocr's engines keep from page to
page is reset (the attributes of pero-ocr's objects in the parser, not the models, back to their values after loading;
a container changed in place stops the run). In production the same parser carries such state over (the downsample
`LAYOUT_CNN` adapted to on the previous page; with `CARRY_H_OVER` the decoder's last line of the previous page, issue
#96), so a page's result there depends on the pages before it; in the benchmark it must not depend on their order, a
page list or a resumed run. This is a rule for every change of pero-ocr or of this code (`docs/benchmark.md`): state kept across pages is wiped out before each page, never by reloading the models. And
every page is processed with numpy's and Python's random generators seeded from its
name, so the layout engine's random tie-breaks (issue #100) are the same in every run.

The records of a commit (a free run's folder has the same structure, plus `<benchmark>/run.json`):

```
<commit>/
  <benchmark>/
    environment.json                  the environment the benchmark's units were made in
    benchmark.json                    fingerprint of the pages (file names and hashes)
    adopted.json                      when units were adopted
    report.md                         the units
    LAYOUT_PARSER/<layout>/           page_xml/
    LINE_CROPPER/<cropper>/           crops.png.lmdb (the line crops as PNG; its pages: pages.tsv)
    OCR/<ocr>@<cropper>/              lines/, page_xml/, logits/, alto/ (--sparse-output N: page_xml.1in<N>/,
                                      logits.1in<N>/, alto.1in<N>/, + page_xml/, logits/ of the pages its readers need)
    DECODER/<lm>@<ocr>@<cropper>/     lines/, page_xml/, alto/ (--sparse-output N: page_xml.1in<N>/, alto.1in<N>/)
    logs/                             <unit>.log: every attempt of a unit appended, between a start and an end line
                                      (a start line without an end line: that attempt was killed or stopped)
    score/                            cer.tsv, layout.tsv, report.md
      lines/<OCR|DECODER>/<unit>/     <test list>.txt: the unit's output on the lines of the test list
```

Every unit has `engine.json`, the engine it ran: `unit` (its id), `engine` (the engine's name), `files` (the md5 of
every file of the engine's folder) and `config` (the config it ran with); a unit is complete when it has it. Everything
else follows from the unit's name, its engine and its folders: a decoder's OCR run is in its name, a derived unit's
threshold in its config, a cropper's pages are the rows of its `pages.tsv` and its crops `crops.png.lmdb`, an OCR unit
keeps `page_xml/` and `logits/` only for the pages a decoder decodes. `unit.json` appears only when a unit was adopted
(`adopted_from`: the commit it was copied from) or extended with `--force-model` (`engine_changes`: what changed, which
pages). Every unit except a derived one also has
`pages.tsv`, the processing of every page, one row per page, in the columns of its stage:

| stage | columns |
|---|---|
| layout | `page`, `seconds`, `gpu_peak_mb`, `image_px` (the pixels of the page image), `pass1_px`, `pass2_px` (the pixels of the ParseNet's input in each pass, empty when there was no such pass) |
| cropper | `page`, `seconds`, `lines` (cropped; on the CPU) |
| OCR | `page`, `seconds`, `gpu_peak_mb`, `lines` (read), `width_px` (the total width of the crops it read: its input size), `chars` (written) |
| decoder | `page`, `seconds`, `gpu_peak_mb`, `lines` (decoded), `frames` (the logit frames of those lines, the padding included: its input size), `chars` (written in those lines) |

The counts of a stage come in the order input, then output. `seconds` is the wall time of the engine's call for the
page, without reading and writing; `gpu_peak_mb` the peak GPU memory; `chars` counts the text as the engine wrote it, before any NFC of the output. A decoder decodes only the lines
over its confidence threshold, the others keep the OCR text. The counts can be derived from the records too (except a
decoder's), and are kept with the inference as a check of them. With `--jobs` over 1 units share a GPU, so their
times include waiting for each other. LAYOUT_CNN runs its ParseNet on the image resized by its starting downsample
(further when the image is over `MAX_MEGAPIXELS`) and padded to a multiple of 64; when the lines it finds are too
small or too large for the network, it runs it a second time at another downsample, and keeps only the second
output. A unit derived by a higher threshold computes nothing and has no `pages.tsv`. OCR and decoder units keep one set of files
per page (`benchmark/records.py`):

- `lines/<page>.tsv`: every line the engine output, in its order: `line_id`, `confidence` (full precision), `text`
  (as the engine wrote it; a tab, newline or backslash in it escaped by a backslash). A line it did not output (the
  confidence filter removes lines) is not there; `score.py` counts it as an empty transcription;
- its outputs, as pero-ocr exports them, with the text as the engine wrote it (no `OUTPUT_NFC`):
  `page_xml/<page>.xml`, `logits/<page>.logits` (OCR; `PageLayout.save_logits_bytes`, read by
  `PageLayout.load_logits`) and `alto/<page>.xml`. A derived unit's are its base's without the lines its threshold
  removes, exported again.

Which outputs and how many lines:

- `--page-xml`, `--logits`, `--alto`: by default a run writes all three; with one or more of them, only those. A run
  refuses to add to a unit that holds pages without an output it writes (recorded without it, or pruned), so an
  output a unit has covers all its pages; run it with the outputs it has.
- `--sparse-output N`, for large benchmarks: the outputs of one line in N only (by a hash of `<page>/<line_id>`,
  the same lines in every unit), cut out of the page's export, in `page_xml.1in<N>/`, `logits.1in<N>/`,
  `alto.1in<N>/`; no file for a page with no line in the subset. The ALTO lines are exported with their neighbours
  in the region, which decide a word hyphenated across lines, so each line is exactly as in the full export (the page
  and block measures around it are not). They are the records to compare the raw output of engines and environments
  beyond the text. An OCR run then keeps its full `page_xml/` and `logits/` only for the pages its decoders and
  derived runs read. A unit keeps every line or one subset; adding to it the other way is refused.

`score.py` writes `<benchmark>/score/lines/<kind>/<unit>/<list>.txt`, the unit's output on exactly the lines of a test
list in the list's order and format, so a test list and a unit (or two units, or two record sets) can be compared side
by side (e.g. `meld test_lists/<list>.test_list <benchmark>/score/lines/OCR/<unit>/<list>.txt`).

Nothing is computed twice: each cropper crops once and the OCR engines read its crops; an OCR engine that differs
from another only in a higher `FILTER_CONFIDENT_LINES_THRESHOLD` is derived from the other's run (its lines without
the lines of confidence at or below the threshold, as pero-ocr removes them); the decoders decode the stored PAGE
XML and logits of the OCR runs. `--prune-page-xml`, `--prune-logits`, `--prune-alto` (all three: `--prune-outputs`)
delete the full outputs, an OCR run's PAGE XML and logits once its decoders and derived runs are done (one added
later then runs the OCR engine again, and must get the recorded lines); `--prune-sparse` deletes the sparse outputs
of every N.

See the docstring of `infer.py` for what each unit records.
