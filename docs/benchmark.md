# The benchmark: rules

For whoever works with the benchmark (`benchmark/infer.py`, `benchmark/score.py`) or changes anything in pero-ocr
that can influence what it records: an engine, the page parser, the line cropper, the decoders, the way models are
loaded. How the benchmark is run and what it records is in `benchmark/README.md`.

The benchmark records what one pero-ocr commit produces, page by page, so that the records of two commits can be
compared page by page and line by line, and a difference means a change of pero-ocr (or of its libraries), not of
how the benchmark happened to run. The rules below keep it so.

## Every page on its own

`benchmark/infer.py` must process every page as if the engine processed that page alone, right after loading. A
page's records must never depend on the pages processed before it: their order, a page list, a resumed or topped-up
run, another chunk size.

Models are not reloaded per page (that would make a run far too slow). Instead every dependency between pages is wiped
out before each page:

- `start_page()` (in `infer.py`) runs before every page of every engine: it restores the attributes of pero-ocr's
  objects in the parser to their values after loading (`PageState`), stops the run when a list, dict or array of
  theirs changed in place (a restore cannot undo that), and seeds numpy's and Python's random generators from the page
  name.
- A change to pero-ocr that keeps anything from one page to the next (a cache, an adapted setting, the text or state
  of the previous page, a counter that changes results), or a change of the benchmark code that does, must be reset
  there. `PageState` resets pero-ocr's object attributes by itself. State it cannot see (module globals, class
  attributes, settings or caches of torch, numpy, opencv or other libraries) is reset in `start_page()` explicitly,
  with a comment saying what and why.
- Check it when adding an engine or changing one: process the same pages once in a longer sequence and once each
  after other predecessors; their records must be identical (apart from the creation times in PAGE XML).

pero-ocr itself does keep state across pages in production, where one parser processes many pages: `LAYOUT_CNN`
starts every page at the downsample it adapted to on the page before, and a decoder with `CARRY_H_OVER` primes the
first line of a page with the last line of the page before. Such behaviour is recorded in issue #96 ("Weird behaviour
to fix or decide"); the benchmark resets it.

## The same conditions within the records of a benchmark

Pages are added to a benchmark's records (a resumed run, a new unit, more pages) only under the conditions they were
recorded with. Each guard can be forced by its flag, and every forced addition is noted in `<benchmark>/forced.json`:

| guard | what must be the same | flag |
|---|---|---|
| environment | the machine (hostname) and the environment (Python, packages, CUDA, cuDNN, GPU, GPU driver), and the pages | `--force-environment` |
| commit | official records: the folder is the commit (a clean checkout); a free run: the commit and uncommitted changes of its first run | `--force-commit` |
| model | a unit: its model files and config (`engine.json`) | `--force-model` |

A run holds `<benchmark>/.lock` while it adds to the records, so two runs never write the same records at once.

## What the records hold

A unit's records are its engine (`engine.json`), its outputs per page and the processing of every page (`pages.tsv`:
seconds, peak GPU memory and the counts of its stage). Keep them so:

- Everything a page produced is in that page's own files, written completely or not at all, so a page can be deleted
  and computed again.
- Nothing is recorded that the unit's name, its engine or its folders already say; a new piece of information goes
  where it belongs (a page's file, a column of `pages.tsv`, a folder name such as `crops.png.lmdb` or
  `alto.1in64/`), not into a general bag of fields.
- Counts in `pages.tsv` are taken where the engine works, even when they could be derived from the records later:
  they are the check of the records.
