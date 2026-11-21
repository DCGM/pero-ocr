# IO consistency checks: golden IO files

This directory checks that pero-ocr keeps **writing** the same files and can still **read** the files it wrote before. It holds the checks and the committed data they compare against.

| file | what |
|---|---|
| `test_pagexml.py`, `test_altoxml.py`, `test_logits.py` | checks: does the current code still produce exactly the committed PAGE XML, ALTO XML and logits? |
| `test_pagexml_input.py` | check: can the current code still read the committed PAGE XML files back? pero-ocr also takes PAGE XML as input, e.g. `parse_folder.py -x`. |
| `golden_io.py` | what the script and the checks share: `GOLDEN_IO_FILES`, the list of golden IO files and the exporter call that produces each one, plus the comparison code |
| `data/` | the committed golden IO files |
| `data/regenerate.py` | the script that recreates `data/` with the current code; all code used only for regenerating is in it |

Every file in `GOLDEN_IO_FILES` is both written by `data/regenerate.py` and compared by the checks, so the two cannot drift apart. The checks also fail if `data/` holds a file that is not in the list, or misses one.

The general IO tests (round trips, single features, known bugs) are one level up in `test/io/`.

## Why

The output files are pero-ocr's interface. Production services, downstream tools and users' scripts parse them, and they break on changes that look harmless in the code, for example:

- a renamed or added attribute
- a different key in the `custom` json
- another namespace or schema version
- a different rounding of coordinates
- a reordered element
- a new key in the logits pickle

Unit tests that assert a few chosen values miss most of these, because nobody thinks to assert the thing that changed by accident. A golden IO file covers the whole document at once, so any change to the output fails a check, including changes nobody anticipated. When a change is intended, the updated golden IO file shows up in the diff of the commit, so a reviewer sees exactly what consumers of the output will see.

## What is in `data/`

The files are grouped by format and named `<page>.<variant>.<extension>`:

| file | produced by |
|---|---|
| **`page_xml/`** | |
| `reference_page.2019.xml` | `to_pagexml_string()`; this is what `parse_folder.py` writes |
| `reference_page.2013.xml` | `to_pagexml_string(version=PAGEVersion.PAGE_2013_07_15)` |
| `reference_page.2019.validate_id.xml` | `to_pagexml_string(validate_id=True)` |
| `edge_page.2019.xml`, `edge_page.2013.xml` | the edge-case page in both versions |
| `empty_page.2019.xml` | a page without regions |
| **`alto/`** | |
| `reference_page.v2.xml` | `to_altoxml_string()`; this is what `parse_folder.py` writes |
| `reference_page.v4.xml` | `to_altoxml_string(version=ALTOVersion.ALTO_v4_4)` |
| `reference_page.v2.page_uuid.xml` | `to_altoxml_string(page_uuid=..., ocr_processing_element=...)` |
| `edge_page.v2.xml`, `edge_page.v4.xml` | the edge-case page in both versions |
| `empty_page.v2.xml` | a page without regions |
| **`logits/`** | |
| `reference_page.logits` | `save_logits()`; this is what `parse_folder.py` writes (pickle protocol 4) |
| `reference_page.bytes.logits` | `save_logits_bytes()` (pickle protocol 5) |
| `edge_page.bytes.logits` | `save_logits_bytes()` of the edge-case page |
| `generated_with.json` | the commit and library versions that produced the files |
| `regenerate.py` | recreates all of the above |

`parse_folder.py` and the other scripts always use the default versions: PAGE 2019 and ALTO v2. The other versions can only be chosen through the Python API.

## How the data is made

The input is not a model run; it is a set of pages built by hand in `test/io/reference_layout.py`. That keeps the files independent of models, the GPU and numeric noise: the same code always gives the same output.

The pages are designed to cover the features the exporters handle:

- `reference_page`: two text regions plus one partly outside the page, region type, category and detection confidence, a reading order, coordinates that need rounding and clamping, line heights, a line without text, a hyphenated word split across lines, a non-text category, diacritics, and CTC logits with confidences below 1.
- `edge_page`: region-level text, an empty transcription, a line without logits, an Arabic line, YOLO-style category regions, and a page name with characters that ALTO has to sanitize.
- `empty_page`: a page without any region.

Timestamps (`Created`, `LastChange`, `processingDateTime`) change on every run, so they are replaced by `MASKED`, both in the files and when comparing. XML is compared after parsing, so whitespace does not matter. Logits are compared by content (keys, matrix type, dtype, shape, values) and by pickle protocol. The code for this is in `golden_io.py`.

The files record **current** behaviour, not ideal behaviour, including known flaws. For example, ALTO coordinates can be negative, and words placed without logits run past the end of the line. Each known flaw also has a strict `xfail` test in `test/io/` that describes the correct behaviour. Fixing a flaw makes both that `xfail` test and the consistency check fail, so the fix has to remove the marker and regenerate the data. That makes the format change visible in review.

## Regenerating

Regenerate the data whenever the output format changes **on purpose**, and commit it together with the code change:

    python test/io/consistency/data/regenerate.py

The script runs every exporter call in `GOLDEN_IO_FILES` with the current code, rewrites `data/page_xml`, `data/alto` and `data/logits` (removing files no longer in the list), and records the commit and library versions in `data/generated_with.json`. It works from any directory and always uses the `pero_ocr` of the checkout it lives in. pytest itself never writes to `data/`. Run it in the basis environment (`requirements/lock.txt`, issue #94); it refuses otherwise unless given `--any-environment`. Note that logits pickled with scipy ≥ 1.8 cannot be read with an older scipy.

To add a golden IO file, add an entry to `GOLDEN_IO_FILES` (and, if needed, a page to `test/io/reference_layout.py`), then run the script. Review the diff of `data/` before committing, and mention the format change in the commit message or PR, since consumers of the output may need to adapt.

## When a check fails

1. Read the diff in the failure. It shows the committed file against the current output.
2. If the change is **not intended**, the code change broke the output format. Fix the code.
3. If the change **is intended**, regenerate the data as described above.

Never edit the files in `data/` by hand, and never regenerate them just to make a failing check pass without understanding the diff.
