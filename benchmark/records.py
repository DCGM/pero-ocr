"""The per-page files of the OCR and DECODER units of the benchmark records (infer.py writes them, score.py and the
tools read them).

A unit holds a page when it holds its lines file:

    lines/<page>.tsv      every line of the input PAGE XML in its order: line_id, confidence (full precision, empty
                          when the line has none), kept (0: the PAGE_PARSER filter removed the line; it has no text and
                          no confidence), text (as the engine wrote it); tab-separated, a tab, newline or backslash in
                          the text escaped by a backslash (csv, QUOTE_NONE, escapechar '\\')
    logits.1in64/<page>.logits   OCR units: the logits of the page's lines in the fixed subset, one line in 64 by a
                          hash of "<page>/<line_id>", the same in every unit, in pero-ocr's format
                          (PageLayout.save_logits_bytes: a pickled dict of the lines' sparse logits, their characters
                          and logit_coords; PageLayout.load_logits reads it); no file when none of its lines is in the
                          subset

Every file is written under a temporary name and renamed, so a page is either there completely or not at all; deleting
a page's files makes the next run compute it again.
"""
import csv
import io
import os
import pickle

LINES = 'lines'
LOGITS_SUBSET = 64  # one line in LOGITS_SUBSET keeps its logits in every OCR unit
LOGITS = f'logits.1in{LOGITS_SUBSET}'
HEADER = ['line_id', 'confidence', 'kept', 'text']
DIALECT = dict(delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')


def line_key(page, line_id):
    """The key of a line in the test lists."""
    return f'{page}-{line_id}.jpg'


def _write(path, data):
    tmp = os.path.join(os.path.dirname(path), f'.{os.path.basename(path)}.{os.getpid()}.tmp')
    with open(tmp, 'wb' if isinstance(data, bytes) else 'w', **({} if isinstance(data, bytes) else
                                                                 dict(encoding='utf-8', newline=''))) as f:
        f.write(data)
    os.replace(tmp, path)


def write_page(folder, page, rows):
    """rows: (line_id, text, confidence, kept) of every line of the page, confidence a string ('' for none), kept
    '1' or '0'."""
    os.makedirs(os.path.join(folder, LINES), exist_ok=True)
    tsv = io.StringIO()
    w = csv.writer(tsv, **DIALECT)
    w.writerow(HEADER)
    w.writerows([(line_id, conf, kept, text) for line_id, text, conf, kept in rows])
    _write(os.path.join(folder, LINES, page + '.tsv'), tsv.getvalue())


def read_page(folder, page):
    """[(line_id, text, confidence, kept)] of a page the unit holds."""
    path = os.path.join(folder, LINES, page + '.tsv')
    with open(path, encoding='utf-8', newline='') as f:
        table = list(csv.reader(f, **DIALECT))
    if table[0] != HEADER:
        raise ValueError(f'{path}: unexpected header {table[0]}')
    return [(line_id, text, conf, kept) for line_id, conf, kept, text in table[1:]]


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


def write_logits(folder, page, data):
    """The subset logits of a page (logits_of_lines); nothing when None."""
    if data is None:
        return
    os.makedirs(os.path.join(folder, LOGITS), exist_ok=True)
    _write(os.path.join(folder, LOGITS, page + '.logits'), data)


def read_logits(folder, page):
    """The subset logits of a page as bytes (PageLayout.load_logits takes them), None when it has none."""
    path = os.path.join(folder, LOGITS, page + '.logits')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return f.read()
