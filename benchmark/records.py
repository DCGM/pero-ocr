"""The per-page files of the OCR and DECODER units of the benchmark records (infer.py writes them, score.py and the
tools read them).

A unit holds a page when it holds its lines file:

    lines/<page>.tsv      every line of the engine's output, in its order: line_id, confidence (full precision, empty
                          when the line has none), text (as the engine wrote it); a line of the input the engine did
                          not output (e.g. removed by the PAGE_PARSER confidence filter) is not there; tab-separated,
                          a tab, newline or backslash in the text escaped by a backslash (csv, QUOTE_NONE,
                          escapechar '\\')

The other outputs of a unit (page_xml/, logits/, alto/ or their sparse versions <kind>.1in<N>/) are pero-ocr's exports,
written by infer.py; logits_of_lines cuts the lines of a sparse subset out of a page's logits.

Every file is written under a temporary name and renamed, so a page is either there completely or not at all; deleting
a page's files makes the next run compute it again.
"""
import csv
import io
import os
import pickle

LINES = 'lines'
HEADER = ['line_id', 'confidence', 'text']
DIALECT = dict(delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')


def line_key(page, line_id):
    """The key of a line in the test lists."""
    return f'{page}-{line_id}.jpg'


def write_file(path, data):
    """Write a file under a temporary name and rename it: it is there completely or not at all."""
    _write(path, data)


def _write(path, data):
    tmp = os.path.join(os.path.dirname(path), f'.{os.path.basename(path)}.{os.getpid()}.tmp')
    with open(tmp, 'wb' if isinstance(data, bytes) else 'w', **({} if isinstance(data, bytes) else
                                                                 dict(encoding='utf-8', newline=''))) as f:
        f.write(data)
    os.replace(tmp, path)


def write_page(folder, page, rows):
    """rows: (line_id, text, confidence) of every line of the engine's output, confidence a string ('' for none)."""
    os.makedirs(os.path.join(folder, LINES), exist_ok=True)
    tsv = io.StringIO()
    w = csv.writer(tsv, **DIALECT)
    w.writerow(HEADER)
    w.writerows([(line_id, conf, text) for line_id, text, conf in rows])
    _write(os.path.join(folder, LINES, page + '.tsv'), tsv.getvalue())


def read_page(folder, page):
    """[(line_id, text, confidence)] of a page the unit holds."""
    path = os.path.join(folder, LINES, page + '.tsv')
    with open(path, encoding='utf-8', newline='') as f:
        table = list(csv.reader(f, **DIALECT))
    if table[0] != HEADER:
        raise ValueError(f'{path}: unexpected header {table[0]}')
    return [(line_id, text, conf) for line_id, conf, text in table[1:]]


def held_pages(folder):
    """The pages a unit holds: their lines file there."""
    lines = os.path.join(folder, LINES)
    if not os.path.isdir(lines):
        return set()
    return {f[:-4] for f in os.listdir(lines) if f.endswith('.tsv') and not f.startswith('.')}


def logits_of_lines(data, line_ids):
    """pero-ocr's logits of a page (PageLayout.save_logits_bytes) limited to the lines line_ids, in the same format;
    None when none of them has logits."""
    page = pickle.loads(data)
    ids = [k for k in line_ids if page.get(k) is not None]
    if not ids:
        return None
    out = {k: page[k] for k in ids}
    for key in ('line_characters', 'logit_coords'):
        out[key] = {k: page.get(key, {}).get(k) for k in ids}
    return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
