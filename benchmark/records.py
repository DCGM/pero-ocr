"""The per-page files of the OCR and DECODER units of the benchmark records (infer.py writes them, score.py and the
tools read them).

A unit holds a page when it holds both line files of the page:

    lines/<page>.txt      the page's lines in the format of the test lists, "<page>-<line_id>.jpg <text>" per line,
                          every line of the input PAGE XML in its order (a line the PAGE_PARSER filter removed: no
                          text); it can be compared with a test list line by line (e.g. meld)
    lines/<page>.tsv      the rest of each line, in the same order: line_id, confidence (full precision, empty when
                          the line has none), kept (0: the filter removed the line)
    logprobs.1in64/<page>.npz   OCR units: the full log-probabilities of the page's lines in the fixed subset, one
                          line in 64 by a hash of "<page>/<line_id>", the same in every unit (a free run can choose
                          another N, infer.py --logprobs-subset: logprobs.1in<N>/) (key: the line id); no
                          file when none of its lines is in the subset. The folder name says the subset, so records
                          made with another one do not look comparable.

Every file is written under a temporary name and renamed, so a page is either there completely or not at all; deleting
a page's files makes the next run compute it again.
"""
import csv
import io
import os

import numpy as np

LINES = 'lines'
LOGPROBS_SUBSET = 64  # one line in LOGPROBS_SUBSET keeps its log-probabilities (official runs: always 64)
LOGPROBS = f'logprobs.1in{LOGPROBS_SUBSET}'


def set_logprobs_subset(n):
    """Keep the log-probabilities of one line in n (infer.py --logprobs-subset, free runs only); the folder name
    logprobs.1in<n> follows."""
    global LOGPROBS_SUBSET, LOGPROBS
    LOGPROBS_SUBSET, LOGPROBS = n, f'logprobs.1in{n}'
HEADER = ['line_id', 'confidence', 'kept']


def line_key(page, line_id):
    """The key of a line in the test lists and in lines/<page>.txt."""
    return f'{page}-{line_id}.jpg'


def _write(path, data):
    tmp = os.path.join(os.path.dirname(path), f'.{os.path.basename(path)}.{os.getpid()}.tmp')
    with open(tmp, 'wb' if isinstance(data, bytes) else 'w', **({} if isinstance(data, bytes) else
                                                                 dict(encoding='utf-8', newline=''))) as f:
        f.write(data)
    os.replace(tmp, path)


def write_page(folder, page, rows):
    """rows: (line_id, text, confidence, kept) of every line of the page, confidence a string ('' for none), kept
    '1' or '0'. The .tsv first, the .txt last: a page is held once its .txt is there."""
    os.makedirs(os.path.join(folder, LINES), exist_ok=True)
    tsv = io.StringIO()
    w = csv.writer(tsv, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
    w.writerow(HEADER)
    w.writerows([(line_id, conf, kept) for line_id, _, conf, kept in rows])
    _write(os.path.join(folder, LINES, page + '.tsv'), tsv.getvalue())
    _write(os.path.join(folder, LINES, page + '.txt'),
           ''.join(f'{line_key(page, line_id)} {text}\n' for line_id, text, _, _ in rows))


def read_page(folder, page):
    """[(line_id, text, confidence, kept)] of a page the unit holds."""
    with open(os.path.join(folder, LINES, page + '.tsv'), encoding='utf-8', newline='') as f:
        meta = list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'))
    if meta[0] != HEADER:
        raise ValueError(f'{folder}/{LINES}/{page}.tsv: unexpected header {meta[0]}')
    with open(os.path.join(folder, LINES, page + '.txt'), encoding='utf-8', newline='') as f:
        texts = f.read().split('\n')
    if texts and texts[-1] == '':
        texts.pop()
    if len(texts) != len(meta) - 1:
        raise ValueError(f'{folder}/{LINES}/{page}: {len(texts)} lines in .txt, {len(meta) - 1} in .tsv')
    rows = []
    for (line_id, conf, kept), line in zip(meta[1:], texts):
        key = line_key(page, line_id)
        if not line.startswith(key + ' '):
            raise ValueError(f'{folder}/{LINES}/{page}: .txt and .tsv disagree at line {line_id}')
        rows.append((line_id, line[len(key) + 1:], conf, kept))
    return rows


def count_lines(folder, page):
    """The number of lines of a page the unit holds."""
    with open(os.path.join(folder, LINES, page + '.txt'), encoding='utf-8', newline='') as f:
        return f.read().count('\n')


def held_pages(folder):
    """The pages a unit holds: both line files there."""
    lines = os.path.join(folder, LINES)
    if not os.path.isdir(lines):
        return set()
    files = set(os.listdir(lines))
    return {f[:-4] for f in files if f.endswith('.txt') and not f.startswith('.') and f[:-4] + '.tsv' in files}


def write_logprobs(folder, page, logprobs):
    """{line_id: array} of a page's lines in the subset; nothing when empty."""
    if not logprobs:
        return
    os.makedirs(os.path.join(folder, LOGPROBS), exist_ok=True)
    data = io.BytesIO()
    np.savez(data, **logprobs)
    _write(os.path.join(folder, LOGPROBS, page + '.npz'), data.getvalue())


def read_logprobs(folder, page):
    """{line_id: array} of a page ({} when none of its lines is in the subset)."""
    path = os.path.join(folder, LOGPROBS, page + '.npz')
    if not os.path.exists(path):
        return {}
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
