#!/usr/bin/env python3
"""Compare the inference records of two commits: the consistency check of the GPU tier.

    python benchmark/consistency.py --base <commit>                     # the current commit against <commit>
    python benchmark/consistency.py --base <commit> --new <commit>
    python benchmark/consistency.py --base <commit> --new ~/bench/try1  # a free run (infer.py --out) against a commit
    python benchmark/consistency.py --base <commit> --expected 'DECODER/*'   # a deliberate change of the decoders

--base and --new are a commit with records in $PERO_INFERENCE, or the folder of a free run. Only the records are read
(never score/); the benchmark data, when given (--data, PERO_TEST_DATA), for its page lists, to say on which data a unit
differs, and its test lists, to measure the annotated lines separately.

For every benchmark both hold and every unit both hold, on the pages both units hold (new against base):

    LAYOUT_PARSER   lines matched by the overlap of their polygons (IoU >= 0.5): the share of lines without a match on
                    either side (of the base's lines), regions added or lost, pages whose ParseNet input
                    (pass1_px, pass2_px) changed
    LINE_CROPPER    the lines cropped (pages.tsv), the crops that differ and their largest pixel difference
    OCR, DECODER    text change: the edit distance between the texts of the lines both output, over the characters of
                    the base (a CER between the commits, the text as the engine wrote it); lines missing (output by
                    the base only) and new; the largest confidence difference; for OCR the largest difference of a
                    character log-probability both store in the logits (the full logits, or the sparse ones where only they are).
                    On the annotated lines (the lines of the test lists) separately: their text change (over the
                    reference characters; a line one side did not output counts as empty) and the CER of either side
                    against the references (as score.py computes it), with its change. The logits are compared only on
                    the lines whose text is the same (a transformer's logits follow its decoded text)
    all             seconds and peak GPU memory, new / base, summed over the pages both hold: reported, never judged

Every measure is computed per unit on all its compared pages, per unit and page list (the annotated measures of a
page list: on the lines of the test list of the same name) and per unit and page. A row's verdict is the worst of its
measures against TOLERANCES (its why: those over their limits); only the row of a whole unit can fail, a page list or a
page at most warns (on a small list one line of an unannotated region, or one line crossing a .tNN threshold, would
be several per cent), and shows where a unit differs. A unit's verdict is the worst of its rows. A unit matching --expected (a unit
name, a stage, or a glob over <STAGE>/<unit>) that fails is EXPECTED instead: a deliberate change, still listed, not a
failure. Units held by one side only, a changed engine (its files or config differ) and different environments are
reported, not judged.

Writes <new records>/<benchmark>/consistency/<base>/:
    summary.md        the text for the pull request: verdicts per stage, the largest differences, no unit names
    report.md         the verdict of every unit, worst first, and where a unit differs: its page lists and pages
    units.tsv         one row per unit, unit_lists.tsv per unit and page list, pages.tsv per unit and page
    lines/<STAGE>/<unit>.tsv   the lines that differ: page, line_id, change, base and new text and confidence
    run.json          both commits (or free runs' folder names), the environment differences, the tolerances, --expected
Exit code 1 when a unit fails (not EXPECTED), 0 otherwise.
"""
import argparse
import csv
import fnmatch
import json
import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
import Levenshtein

from records import DIALECT, held_pages, read_page
from score import clean, load_lists

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ('LAYOUT_PARSER', 'LINE_CROPPER', 'OCR', 'DECODER')
CROPS = 'crops.png.lmdb'
ALL = '*'  # the page list row of all compared pages

PASS, WARN, EXPECTED, FAIL = 'PASS', 'WARN', 'EXPECTED', 'FAIL'
RANK = {PASS: 0, WARN: 1, EXPECTED: 2, FAIL: 3}

# (warn above, fail above); None: not judged at that level. Set from what was measured on the develop benchmark
# (2026-11-27), the same commit in the lock environment and in newer libraries (torch 2.14/2.15, opencv 5.1, numpy 2):
# per unit a text change of at most 0.2 % (0.05 % on annotated lines, CER within 0.013 pp), at most 0.19 % lines
# missing or new (.tNN units), every crop different by up to 7 grey levels, layout identical, log-probabilities of
# lines with the same text different by up to 0.19 (transformers); Python 3.10 and 3.12 with the same libraries
# gave the same text. The same environment twice gave identical records. A broken character set changed 9 %.
TOLERANCES = {
    'text_change_pct': (0.1, 1.0),         # OCR, DECODER: edit distance / base characters
    'ann_text_change_pct': (0.1, 1.0),     # OCR, DECODER: the same on the annotated lines / reference characters
    'missing_new_lines_pct': (0.0, 1.0),   # OCR, DECODER: lines output by one side only / base lines
    'logprob_diff': (1e-3, None),          # OCR: the largest difference of a character log-probability
    'layout_unmatched_pct': (0.2, 2.0),    # LAYOUT_PARSER: lines without a match on either side / base lines
    'layout_input_changed_pages': (0, None),  # LAYOUT_PARSER: pages whose ParseNet input size changed
    'crop_lines_diff': (None, 0),          # LINE_CROPPER: lines cropped, difference
    'crops_differing': (0, None),          # LINE_CROPPER: crops that differ
}


def judge(row, can_fail=True):
    """(verdict, the measures over their limits) of a row; a row that cannot fail (a page list, a page) at most warns."""
    worst, why = PASS, []
    for key, (warn, fail) in TOLERANCES.items():
        value = row.get(key)
        if value is None or value == '':
            continue
        if fail is not None and value > fail:
            worst = FAIL if can_fail else max(worst, WARN, key=RANK.get)
            why.append(key)
        elif warn is not None and value > warn:
            worst = max(worst, WARN, key=RANK.get)
            why.append(key)
    return worst, ' '.join(why)


# Reading the records

def records_root(value, results):
    """(folder, label) of --base or --new: a free run's folder, or a commit's records in results."""
    if os.path.isdir(value):
        return value, os.path.basename(os.path.normpath(value))
    if not results:
        sys.exit(f'{value} is not a folder, and no --results (PERO_INFERENCE) to find a commit in')
    folder = os.path.join(results, value)
    if not os.path.isdir(folder):
        sys.exit(f'no records of {value} in {results}')
    return folder, value


def units_of(records):
    """{'<STAGE>/<unit>': folder} of the complete units (with engine.json)."""
    out = {}
    for stage in STAGES:
        d = os.path.join(records, stage)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if os.path.isfile(os.path.join(d, name, 'engine.json')):
                    out[f'{stage}/{name}'] = os.path.join(d, name)
    return out


def read_json(path):
    with open(path) as f:
        return json.load(f)


def pages_tsv(folder):
    """{page: row} of a unit's pages.tsv ({} when it has none, e.g. a derived unit)."""
    path = os.path.join(folder, 'pages.tsv')
    if not os.path.isfile(path):
        return {}
    with open(path, newline='') as f:
        return {r['page']: r for r in csv.DictReader(f, delimiter='\t')}


def annotated(dataset):
    """{page: {line_id: [(test list, reference)]}} of the benchmark's test lists, {} without them."""
    if not dataset or not os.path.isdir(os.path.join(dataset, 'test_lists')):
        return {}
    out = defaultdict(lambda: defaultdict(list))
    for name, refs in load_lists(dataset).items():
        for page, line_id, ref in refs:
            out[page][line_id].append((name, ref))
    return out


def page_lists(dataset):
    """{page list: set of pages} of the benchmark, {} without its data."""
    d = os.path.join(dataset, 'page_lists') if dataset else None
    if not d or not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('.page_list'):
            with open(os.path.join(d, f)) as fh:
                out[f[:-len('.page_list')]] = {s.strip() for s in fh if s.strip() and not s.startswith('#')}
    return out


# Per page comparisons: each returns the sums of one page, which are added up per page list

def xml_lines(path):
    """(regions, [(line_id, polygon)]) of a PAGE XML file."""
    from lxml import etree
    from shapely.geometry import Polygon
    root = etree.parse(path).getroot()
    regions, lines = 0, []
    for el in root.iter():
        tag = etree.QName(el).localname
        if tag == 'TextRegion':
            regions += 1
        elif tag == 'TextLine':
            coords = next((c for c in el if etree.QName(c).localname == 'Coords'), None)
            pts = [tuple(map(float, p.split(','))) for p in (coords.get('points') or '').split()] if coords is not None else []
            poly = Polygon(pts) if len(pts) >= 3 else None
            if poly is not None and not poly.is_valid:
                poly = poly.buffer(0)
            lines.append((el.get('id'), poly))
    return regions, lines


def unmatched(a, b, threshold=0.5):
    """The lines of a with no line of b overlapping it by IoU >= threshold."""
    from shapely.strtree import STRtree
    polys = [p for _, p in b if p is not None and not p.is_empty]
    tree = STRtree(polys) if polys else None
    out = []
    for line_id, p in a:
        best = 0.0
        if p is not None and not p.is_empty and tree is not None:
            for i in tree.query(p):
                q = polys[int(i)] if not hasattr(i, 'area') else i
                union = p.union(q).area
                if union > 0:
                    best = max(best, p.intersection(q).area / union)
        if best < threshold:
            out.append(line_id)
    return out


def compare_layout_page(base, new, page, base_rows, new_rows):
    rb, lb = xml_lines(os.path.join(base, 'page_xml', page + '.xml'))
    rn, ln = xml_lines(os.path.join(new, 'page_xml', page + '.xml'))
    lost, added = unmatched(lb, ln), unmatched(ln, lb)
    a, b = base_rows.get(page, {}), new_rows.get(page, {})
    input_changed = any(a.get(k, '') != b.get(k, '') for k in ('pass1_px', 'pass2_px')) if a and b else False
    sums = dict(base_lines=len(lb), new_lines=len(ln), lines_lost=len(lost), lines_added=len(added),
                regions_changed=abs(rb - rn), layout_input_changed_pages=int(input_changed))
    changes = [(page, i, 'lost', '', '', '', '') for i in lost] + [(page, i, 'added', '', '', '', '') for i in added]
    return sums, changes


def compare_crops_page(page, base_rows, new_rows, base_txn, new_txn):
    import cv2
    a, b = base_rows.get(page, {}), new_rows.get(page, {})
    sums = dict(base_lines=int(a.get('lines') or 0), new_lines=int(b.get('lines') or 0), crops_differing=0,
                crop_max_pixel_diff=0)
    changes = []
    if base_txn is None or new_txn is None:
        return sums, changes
    prefix = (page + '/').encode()
    keys_a = {k for k, _ in base_txn.cursor() if k.startswith(prefix)}
    keys_b = {k for k, _ in new_txn.cursor() if k.startswith(prefix)}
    for k in sorted(keys_a | keys_b):
        line_id = k.decode()[len(prefix):]
        if k not in keys_a or k not in keys_b:
            sums['crops_differing'] += 1
            changes.append((page, line_id, 'lost' if k in keys_a else 'added', '', '', '', ''))
            continue
        pa, pb = base_txn.get(k), new_txn.get(k)
        if pa == pb:
            continue
        ia = cv2.imdecode(np.frombuffer(pa, np.uint8), cv2.IMREAD_UNCHANGED)
        ib = cv2.imdecode(np.frombuffer(pb, np.uint8), cv2.IMREAD_UNCHANGED)
        if ia.shape != ib.shape:
            diff = 255
        else:
            diff = int(np.abs(ia.astype(np.int16) - ib.astype(np.int16)).max())
        if diff:
            sums['crops_differing'] += 1
            sums['crop_max_pixel_diff'] = max(sums['crop_max_pixel_diff'], diff)
            changes.append((page, line_id, f'crop differs by {diff}', '', '', '', ''))
    return sums, changes


def logits_folder(base, new):
    """The logits both units keep: 'logits', else a sparse 'logits.1in<N>' both have, else None."""
    def kinds(folder):
        return {f for f in os.listdir(folder) if re.fullmatch(r'logits(\.1in\d+)?', f)} if os.path.isdir(folder) else set()
    common = kinds(base) & kinds(new)
    if 'logits' in common:
        return 'logits'
    return sorted(common)[0] if common else None


class _Unpickler(pickle.Unpickler):
    """Reads logits pickled under numpy 2 (numpy._core) also under numpy 1 (numpy.core), so that records of two
    environments can be compared in either."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            try:
                __import__(module)
            except ImportError:
                module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def load_logits(folder, kind, page):
    path = os.path.join(folder, kind, page + '.logits')
    if not os.path.isfile(path):
        return {}
    with open(path, 'rb') as f:
        data = _Unpickler(f).load()
    return {k: v for k, v in data.items() if k not in ('line_characters', 'logit_coords') and v is not None}


def logprobs(sparse):
    """(log-probabilities, stored) of a line's stored logits, as pero-ocr computes them (the entries not stored, of
    probability below 1e-4, filled with -80)."""
    dense = sparse.toarray().astype(np.float64)
    stored = dense != 0
    dense[~stored] = -80
    dense -= dense.max(axis=1, keepdims=True)
    return dense - np.log(np.exp(dense).sum(axis=1, keepdims=True)), stored


def logprob_diff(a, b):
    """The largest difference of a log-probability both sides store; None when the shapes differ."""
    (la, sa), (lb, sb) = logprobs(a), logprobs(b)
    if la.shape != lb.shape:
        return None
    both = sa & sb
    return float(np.abs(la - lb)[both].max()) if both.any() else 0.0


ANN_KEYS = ('ann_lines', 'ann_lines_changed', 'ann_ref_chars', 'ann_edits', 'ann_errors_base', 'ann_errors_new')


def compare_annotated(a, b, refs):
    """{test list or ALL: sums} over the annotated lines of a page (ALL: every annotated line once)."""
    out = {}
    for line_id, entries in refs.items():
        ta, tb = (a.get(line_id) or ('', ''))[0], (b.get(line_id) or ('', ''))[0]
        ref = entries[0][1]
        one = dict(ann_lines=1, ann_lines_changed=int(ta != tb), ann_ref_chars=len(ref),
                   ann_edits=Levenshtein.distance(ta, tb), ann_errors_base=Levenshtein.distance(clean(ta), ref),
                   ann_errors_new=Levenshtein.distance(clean(tb), ref))
        for name in {n for n, _ in entries} | {ALL}:
            add(out.setdefault(name, {}), one)
    return out


def compare_text_page(base, new, page, kind, refs):
    a = {i: (t, c) for i, t, c in read_page(base, page)}
    b = {i: (t, c) for i, t, c in read_page(new, page)}
    sums = dict(base_lines=len(a), new_lines=len(b), lines_missing=0, lines_new=0, lines_changed=0, base_chars=0,
                edits=0, conf_max_diff=0.0, logprob_diff=None)
    changes = []
    for i in a:
        if i not in b:
            sums['lines_missing'] += 1
            changes.append((page, i, 'missing', a[i][0], '', a[i][1], ''))
    for i in b:
        if i not in a:
            sums['lines_new'] += 1
            changes.append((page, i, 'new', '', b[i][0], '', b[i][1]))
    for i in a:
        if i not in b:
            continue
        (ta, ca), (tb, cb) = a[i], b[i]
        sums['base_chars'] += len(ta)
        if ca and cb:
            sums['conf_max_diff'] = max(sums['conf_max_diff'], abs(float(ca) - float(cb)))
        elif ca != cb:
            sums['conf_max_diff'] = max(sums['conf_max_diff'], 1.0)
        if ta != tb:
            sums['edits'] += Levenshtein.distance(ta, tb)
            sums['lines_changed'] += 1
            changes.append((page, i, 'text', ta, tb, ca, cb))
        elif ca != cb:
            changes.append((page, i, 'confidence', ta, tb, ca, cb))
    if kind:
        # only lines with the same text: a transformer's logits follow its decoded text, and where the text changed
        # the text measures already tell
        la, lb = load_logits(base, kind, page), load_logits(new, kind, page)
        same = [i for i in la.keys() & lb.keys() if i in a and i in b and a[i][0] == b[i][0]]
        sums['logprob_lines'] = len(same)
        sums['logprob_shape_changed'] = sum(la[i].shape != lb[i].shape for i in same)
        diffs = [logprob_diff(la[i], lb[i]) for i in same if la[i].shape == lb[i].shape]
        sums['logprob_diff'] = max(diffs) if diffs else None
    return sums, changes, compare_annotated(a, b, refs) if refs else {}


# Adding up

SUM_MAX = {'conf_max_diff', 'logprob_diff', 'crop_max_pixel_diff'}


def add(total, sums):
    for k, v in sums.items():
        if v is None:
            total.setdefault(k, None)
        elif k in SUM_MAX:
            total[k] = v if total.get(k) is None else max(total[k], v)
        else:
            total[k] = total.get(k, 0) + v


def pct(a, b):
    return round(100.0 * a / b, 4) if b else (0.0 if not a else 100.0)


def measures(stage, t):
    """The judged and reported measures of a row from its sums."""
    if stage in ('OCR', 'DECODER'):
        out = dict(text_change_pct=pct(t.get('edits', 0), t.get('base_chars', 0)),
                   missing_new_lines_pct=pct(t.get('lines_missing', 0) + t.get('lines_new', 0), t.get('base_lines', 0)),
                   logprob_diff=t.get('logprob_diff'))
        if t.get('ann_lines'):
            chars = t['ann_ref_chars']
            out.update(ann_text_change_pct=pct(t['ann_edits'], chars), cer_base=pct(t['ann_errors_base'], chars),
                       cer_new=pct(t['ann_errors_new'], chars))
            out['cer_change_pp'] = round(out['cer_new'] - out['cer_base'], 4)
        return out
    if stage == 'LAYOUT_PARSER':
        return dict(layout_unmatched_pct=pct(t.get('lines_lost', 0) + t.get('lines_added', 0), t.get('base_lines', 0)),
                    layout_input_changed_pages=t.get('layout_input_changed_pages', 0))
    return dict(crop_lines_diff=abs(t.get('new_lines', 0) - t.get('base_lines', 0)),
                crops_differing=t.get('crops_differing', 0))


def resources(pages, base_rows, new_rows):
    """new / base of the seconds and the peak GPU memory over the pages both have rows for."""
    both = [p for p in pages if p in base_rows and p in new_rows]

    def total(rows, key, f):
        vals = [float(rows[p][key]) for p in both if rows[p].get(key) not in (None, '')]
        return f(vals) if vals else None
    out = {}
    for key, f in (('seconds', sum), ('gpu_peak_mb', max)):
        a, b = total(base_rows, key, f), total(new_rows, key, f)
        out[f'{key}_ratio'] = round(b / a, 3) if a and b is not None else None
    return out


def rounded(sums):
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sums.items()}


def compare_unit(uid, base, new, lists, refs):
    stage = uid.split('/')[0]
    base_rows, new_rows = pages_tsv(base), pages_tsv(new)
    if stage in ('OCR', 'DECODER'):
        pa, pb = held_pages(base), held_pages(new)
    elif stage == 'LAYOUT_PARSER':
        def xmls(f):
            d = os.path.join(f, 'page_xml')
            return {x[:-4] for x in os.listdir(d) if x.endswith('.xml') and not x.startswith('.')} if os.path.isdir(d) else set()
        pa, pb = xmls(base), xmls(new)
    else:
        pa, pb = set(base_rows), set(new_rows)
    pages = sorted(pa & pb)
    txns, kind = (None, None), None
    if stage == 'LINE_CROPPER':
        import lmdb
        envs = [lmdb.open(os.path.join(f, CROPS), readonly=True, lock=False, subdir=True)
                if os.path.isdir(os.path.join(f, CROPS)) else None for f in (base, new)]
        txns = tuple(e.begin() if e is not None else None for e in envs)
    if stage == 'OCR':
        kind = logits_folder(base, new)

    page_rows, changes, ann = [], [], {}
    for page in pages:
        if stage in ('OCR', 'DECODER'):
            sums, ch, ann[page] = compare_text_page(base, new, page, kind, refs.get(page))
        elif stage == 'LAYOUT_PARSER':
            sums, ch = compare_layout_page(base, new, page, base_rows, new_rows)
        else:
            sums, ch = compare_crops_page(page, base_rows, new_rows, *txns)
        page_rows.append((page, sums))
        changes += ch

    groups = {ALL: pages}
    for name, members in lists.items():
        sel = [p for p in pages if p in members]
        if sel:
            groups[name] = sel
    by_page = dict(page_rows)
    list_rows = []
    for name, sel in groups.items():
        total = {}
        for p in sel:
            add(total, by_page[p])
            if p in ann and name in ann[p]:
                add(total, ann[p][name])
        row = dict(dict(unit=uid, page_list=name, pages=len(sel)), **rounded(total))
        row.update(measures(stage, total), **resources(sel, base_rows, new_rows))
        row['verdict'], row['why'] = judge(row, can_fail=name == ALL)
        list_rows.append(row)
    pages_out = []
    for p, sums in page_rows:
        sums = dict(sums, **ann.get(p, {}).get(ALL, {}))
        row = dict(dict(unit=uid, page=p), **rounded(sums))
        row.update(measures(stage, sums))
        row['verdict'], row['why'] = judge(row, can_fail=False)
        pages_out.append(row)
    only = dict(pages_base_only=len(pa - pb), pages_new_only=len(pb - pa))
    return list_rows, pages_out, changes, only, (kind or '')


# Writing

def write_tsv(path, rows, first=()):
    if not rows:
        open(path, 'w').close()
        return
    keys = list(first) + [k for r in rows for k in r if k not in first]
    keys = list(dict.fromkeys(keys))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in keys})


def fmt(v, digits=3):
    if v is None or v == '':
        return '–'
    if isinstance(v, float):
        return f'{v:.{digits}g}' if v and abs(v) < 0.01 else f'{v:.{digits}f}'.rstrip('0').rstrip('.') or '0'
    return str(v)


def env_diff(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        if k == 'packages':
            pa, pb = a.get(k, {}), b.get(k, {})
            for p in sorted(set(pa) | set(pb)):
                if pa.get(p) != pb.get(p):
                    out.append(f'package {p}: {pa.get(p, "–")} -> {pb.get(p, "–")}')
        elif a.get(k) != b.get(k):
            out.append(f'{k}: {a.get(k, "–")} -> {b.get(k, "–")}')
    return out


def registries(records):
    try:
        runs = read_json(os.path.join(records, 'run.json'))
    except (OSError, ValueError):
        return []
    out = []
    for r in runs:
        argv = r.get('argv', [])
        out += [os.path.basename(argv[i + 1]) for i, a in enumerate(argv[:-1]) if a == '--registry']
    return sorted(set(out))


def compare_benchmark(name, base_rec, new_rec, base_label, new_label, dataset, expected):
    lists, refs = page_lists(dataset), annotated(dataset)
    ub, un = units_of(base_rec), units_of(new_rec)
    unit_rows, list_rows, page_rows = [], [], []
    out = os.path.join(new_rec, 'consistency', base_label)
    os.makedirs(out, exist_ok=True)
    for uid in sorted(ub.keys() & un.keys(), key=lambda u: (STAGES.index(u.split('/')[0]), u)):
        stage = uid.split('/')[0]
        print(f'{name}: {uid}', file=sys.stderr)
        lr, pr, changes, only, kind = compare_unit(uid, ub[uid], un[uid], lists, refs)
        ea, eb = read_json(os.path.join(ub[uid], 'engine.json')), read_json(os.path.join(un[uid], 'engine.json'))
        engine_changed = (ea.get('files'), ea.get('config')) != (eb.get('files'), eb.get('config'))
        verdict = max((r['verdict'] for r in lr), key=RANK.get, default=PASS)
        if verdict == FAIL and any(fnmatch.fnmatch(uid, p) or p in (stage, uid.split('/', 1)[1]) for p in expected):
            verdict = EXPECTED
        total = next(r for r in lr if r['page_list'] == ALL)
        unit_rows.append(dict(total, stage=stage, unit=uid.split('/', 1)[1], verdict=verdict,
                              engine_changed=engine_changed, logits=kind, **only))
        list_rows += lr
        page_rows += pr
        if changes:
            d = os.path.join(out, 'lines', stage)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, uid.split('/', 1)[1] + '.tsv'), 'w', newline='') as f:
                w = csv.writer(f, **DIALECT)
                w.writerow(['page', 'line_id', 'change', 'base_text', 'new_text', 'base_confidence', 'new_confidence'])
                w.writerows(changes)
    env_a = read_json(os.path.join(base_rec, 'environment.json'))
    env_b = read_json(os.path.join(new_rec, 'environment.json'))
    pages_a = read_json(os.path.join(base_rec, 'benchmark.json')).get('pages')
    pages_b = read_json(os.path.join(new_rec, 'benchmark.json')).get('pages')
    info = dict(benchmark=name, base=base_label, new=new_label, environment_differences=env_diff(env_a, env_b),
                benchmark_pages_differ=pages_a != pages_b, units_base_only=sorted(ub.keys() - un.keys()),
                units_new_only=sorted(un.keys() - ub.keys()), registries_new=registries(new_rec),
                tolerances=TOLERANCES, expected=expected)
    with open(os.path.join(out, 'run.json'), 'w') as f:
        json.dump(info, f, indent=1)
    first = ('stage', 'unit', 'verdict', 'pages')
    write_tsv(os.path.join(out, 'units.tsv'), unit_rows, first)
    write_tsv(os.path.join(out, 'unit_lists.tsv'), list_rows, ('unit', 'page_list', 'verdict', 'pages'))
    write_tsv(os.path.join(out, 'pages.tsv'), page_rows, ('unit', 'page', 'verdict'))
    write_report(os.path.join(out, 'report.md'), info, unit_rows, list_rows, page_rows)
    write_summary(os.path.join(out, 'summary.md'), info, unit_rows)
    print(os.path.join(out, 'report.md'))
    return unit_rows


MEASURE_COLS = {
    'OCR': [('text_change_pct', 'text change %'), ('lines_changed', 'lines changed'),
            ('missing_new_lines_pct', 'missing+new %'), ('conf_max_diff', 'max conf diff'),
            ('logprob_diff', 'max logprob diff (same text)'), ('ann_text_change_pct', 'annotated: text change %'),
            ('cer_base', 'CER base %'), ('cer_new', 'CER new %'), ('cer_change_pp', 'CER change pp')],
    'DECODER': [('text_change_pct', 'text change %'), ('lines_changed', 'lines changed'),
                ('missing_new_lines_pct', 'missing+new %'), ('conf_max_diff', 'max conf diff'),
                ('ann_text_change_pct', 'annotated: text change %'), ('cer_base', 'CER base %'), ('cer_new', 'CER new %'),
                ('cer_change_pp', 'CER change pp')],
    'LAYOUT_PARSER': [('layout_unmatched_pct', 'unmatched lines %'), ('lines_lost', 'lost'), ('lines_added', 'added'),
                      ('regions_changed', 'regions +-'), ('layout_input_changed_pages', 'input changed (pages)')],
    'LINE_CROPPER': [('crop_lines_diff', 'lines diff'), ('crops_differing', 'crops differing'),
                     ('crop_max_pixel_diff', 'max pixel diff')],
}


def overall(unit_rows):
    return max((r['verdict'] for r in unit_rows), key=RANK.get, default=PASS)


def write_report(path, info, unit_rows, list_rows, page_rows):
    L = [f"# Consistency: {info['benchmark']}, {info['new']} against {info['base']}", '',
         f"Verdict: **{overall(unit_rows)}** ({', '.join(f'{v} {n}' for v, n in counts(unit_rows))}).", '',
         f"Records: base `{info['base']}/{info['benchmark']}`, new `{info['new']}/{info['benchmark']}` (commits, or free "
         f"runs by their folder name).", '']
    if info['environment_differences']:
        L += ['Environments differ (differences may come from them):', ''] + [f'- {d}' for d in info['environment_differences']] + ['']
    else:
        L += ['Same environment.', '']
    if info['benchmark_pages_differ']:
        L += ['The benchmark\'s pages differ between the records (benchmark.json).', '']
    for k, what in (('units_base_only', 'Units only in the base'), ('units_new_only', 'Units only in the new records')):
        if info[k]:
            L += [f'{what} (not compared):', ''] + [f'- `{u}`' for u in info[k]] + ['']
    for stage in STAGES:
        rows = sorted([r for r in unit_rows if r['stage'] == stage], key=lambda r: (-RANK[r['verdict']], r['unit']))
        if not rows:
            continue
        cols = MEASURE_COLS[stage]
        L += [f'## {stage}', '', '| verdict | why | unit | pages | ' + ' | '.join(c for _, c in cols)
              + ' | time new/base | GPU mem new/base |', '|' + '---|' * (len(cols) + 6)]
        for r in rows:
            notes = []
            if r['engine_changed']:
                notes.append('engine changed')
            if r['pages_base_only'] or r['pages_new_only']:
                notes.append(f"pages only in base {r['pages_base_only']}, only in new {r['pages_new_only']}")
            unit = f"`{r['unit']}`" + (f" ({'; '.join(notes)})" if notes else '')
            L.append(f"| {r['verdict']} | {r.get('why') or '–'} | {unit} | {r['pages']} | " + ' | '.join(fmt(r.get(k)) for k, _ in cols)
                     + f" | {fmt(r.get('seconds_ratio'))} | {fmt(r.get('gpu_peak_mb_ratio'))} |")
        L.append('')
    bad = [r for r in unit_rows if r['verdict'] != PASS]
    if bad:
        L += ['## Where the units differ', '']
        for r in sorted(bad, key=lambda r: (-RANK[r['verdict']], r['stage'], r['unit'])):
            uid = f"{r['stage']}/{r['unit']}"
            cols = MEASURE_COLS[r['stage']]
            L += [f"### {r['verdict']}: `{uid}`", '', '| verdict | why | page list | pages | ' + ' | '.join(c for _, c in cols)
                  + ' |', '|' + '---|' * (len(cols) + 4)]
            for lr in sorted([x for x in list_rows if x['unit'] == uid and x['page_list'] != ALL],
                             key=lambda x: (-RANK[x['verdict']], x['page_list'])):
                L.append(f"| {lr['verdict']} | {lr.get('why') or '–'} | {lr['page_list']} | {lr['pages']} | " + ' | '.join(fmt(lr.get(k)) for k, _ in cols) + ' |')
            worst = sorted([p for p in page_rows if p['unit'] == uid and p['verdict'] != PASS],
                           key=lambda p: (-RANK[p['verdict']], p['page']))[:10]
            if worst:
                L += ['', 'Pages: ' + ', '.join(f"`{p['page']}` ({p['verdict']})" for p in worst)]
            L += ['', f"Lines: `lines/{r['stage']}/{r['unit']}.tsv`", '']
    L += ['## Tolerances (warn above, fail above)', '']
    L += [f'- {k}: {w if w is not None else "–"}, {f if f is not None else "–"}' for k, (w, f) in TOLERANCES.items()]
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def counts(rows):
    c = defaultdict(int)
    for r in rows:
        c[r['verdict']] += 1
    return [(v, c[v]) for v in (FAIL, EXPECTED, WARN, PASS) if c[v]]


def write_summary(path, info, unit_rows):
    """The text for the pull request: no unit names, pages or paths, only verdicts and the largest differences."""
    L = [f"**GPU tier consistency, benchmark `{info['benchmark']}`: {overall(unit_rows)}**", '',
         f"- commits: new `{info['new'][:12]}` against base `{info['base'][:12]}`",
         f"- registries (new): {', '.join(f'`{r}`' for r in info['registries_new']) or '–'}",
         f"- environment: {'same' if not info['environment_differences'] else 'differs (' + str(len(info['environment_differences'])) + ' items)'}"]
    if info['units_base_only'] or info['units_new_only']:
        L.append(f"- units not compared: {len(info['units_base_only'])} only in base, {len(info['units_new_only'])} only in new")
    L += ['', '| stage | units | verdicts | largest difference |', '|---|---|---|---|']
    for stage in STAGES:
        rows = [r for r in unit_rows if r['stage'] == stage]
        if not rows:
            continue
        if stage in ('OCR', 'DECODER'):
            diff = f"text change {fmt(max(r['text_change_pct'] for r in rows))} %, missing+new lines " \
                   f"{fmt(max(r['missing_new_lines_pct'] for r in rows))} %"
            probs_ = [r['logprob_diff'] for r in rows if r.get('logprob_diff') is not None]
            if probs_:
                diff += f', log-probability {fmt(max(probs_))}'
            ann_rows = [r for r in rows if r.get('cer_change_pp') is not None]
            if ann_rows:
                worst = max(ann_rows, key=lambda r: abs(r['cer_change_pp']))
                diff += f"; annotated lines: text change {fmt(max(r['ann_text_change_pct'] for r in ann_rows))} %, " \
                        f"CER change {worst['cer_change_pp']:+.3f} pp (largest)"
        elif stage == 'LAYOUT_PARSER':
            diff = f"unmatched lines {fmt(max(r['layout_unmatched_pct'] for r in rows))} %, input size changed on " \
                   f"{max(r['layout_input_changed_pages'] for r in rows)} pages"
        else:
            diff = f"lines {max(r['crop_lines_diff'] for r in rows)}, crops differing {max(r['crops_differing'] for r in rows)}"
        L.append(f"| {stage} | {len(rows)} | {', '.join(f'{v} {n}' for v, n in counts(rows))} | {diff} |")
    changed = sum(1 for r in unit_rows if r['engine_changed'])
    if changed:
        L += ['', f'{changed} units have a changed engine (model files or config).']
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', required=True, help='a commit with records in --results, or a free run\'s folder')
    p.add_argument('--new', help='the same (default: the current commit)')
    p.add_argument('--results', default=os.environ.get('PERO_INFERENCE'))
    p.add_argument('--data', default=os.environ.get('PERO_TEST_DATA'), help='the benchmarks (only for page lists)')
    p.add_argument('--benchmarks', nargs='*', help='only these benchmarks')
    p.add_argument('--expected', nargs='*', default=[],
                   help='units whose failure is a deliberate change: a unit name, a stage, or a glob over <STAGE>/<unit>')
    args = p.parse_args()
    new = args.new or subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], check=True, capture_output=True,
                                     text=True).stdout.strip()
    base_root, base_label = records_root(args.base, args.results)
    new_root, new_label = records_root(new, args.results)

    def benchmarks(root):
        return {n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, 'benchmark.json'))}
    found = sorted(benchmarks(base_root) & benchmarks(new_root))
    if args.benchmarks:
        missing = sorted(set(args.benchmarks) - set(found))
        if missing:
            sys.exit(f'not in both records: {", ".join(missing)}')
        found = [b for b in found if b in args.benchmarks]
    if not found:
        sys.exit(f'no benchmark in both {base_root} and {new_root}')
    failed = False
    for name in found:
        rows = compare_benchmark(name, os.path.join(base_root, name), os.path.join(new_root, name), base_label,
                                 new_label, os.path.join(args.data, name) if args.data else None, args.expected)
        print(f'{name}: {overall(rows)}')
        failed |= overall(rows) == FAIL
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
