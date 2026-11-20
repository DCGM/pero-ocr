# Golden files

The files in this directory are reference outputs of pero-ocr's exporters: PAGE XML, ALTO XML and the logits pickle. The tests in `test/io/` export the same input again and fail if the result differs from the file here.

## Why

The output files are pero-ocr's interface. Production services, downstream tools and users' scripts parse them, and they break on changes that look harmless in the code, for example:

- a renamed or added attribute
- a different key in the `custom` json
- another namespace or schema version
- a different rounding of coordinates
- a reordered element
- a new key in the logits pickle

Unit tests that assert a few chosen values miss most of these, because nobody thinks to assert the thing that changed by accident. A golden file covers the whole document at once, so any change to the output fails a test, including changes nobody anticipated. When a change is intended, the updated golden file shows up in the diff of the commit, so a reviewer sees exactly what consumers of the output will see.

## How the files are made

The input is not a model run; it is a page built by hand in `test/io/reference_layout.py`. That keeps the files independent of models, the GPU and numeric noise: the same code always gives the same output.

The pages are designed to cover the features the exporters handle:

- `reference_page`: two text regions plus one partly outside the page, region type, category and detection confidence, a reading order, coordinates that need rounding and clamping, line heights, a line without text, a hyphenated word split across lines, a non-text category, diacritics, and CTC logits with confidences below 1.
- `edge_page`: region-level text, an empty transcription, a line without logits, an Arabic line, YOLO-style category regions, and a page name with characters that ALTO has to sanitize.
- `empty_page`: a page without any region.

Timestamps (`Created`, `LastChange`, `processingDateTime`) change on every run, so they are replaced by `MASKED`, both in the files and when comparing. XML is compared after parsing, so whitespace does not matter. Logits are compared by content (keys, matrix type, dtype, shape, values) and by pickle protocol.

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

`parse_folder.py` and the other scripts always use the default versions: PAGE 2019 and ALTO v2. The other versions can only be chosen through the Python API.

## The files record current behaviour, not ideal behaviour

The files were generated from develop @ 81b496e and contain what pero-ocr produced at that point, including known flaws. For example, ALTO coordinates can be negative, and words placed without logits run past the end of the line. Each known flaw also has a strict `xfail` test in `test/io/` that describes the correct behaviour. Fixing the flaw makes both the golden comparison and the `xfail` test fail, so the fix has to update the golden file and remove the marker, which makes the format change visible in review.

## When a golden test fails

1. Read the diff in the test failure. It shows the expected file against the current output.
2. If the change is **not intended**, the code change broke the output format. Fix the code.
3. If the change **is intended**, regenerate the files and commit them together with the code change:

       pytest test/io --update-golden

   Review the diff of `test/io/data/` before committing. Mention the format change in the commit message or PR, since consumers of the output may need to adapt.

Never edit these files by hand, and never regenerate them just to make a failing test pass without understanding the diff.
