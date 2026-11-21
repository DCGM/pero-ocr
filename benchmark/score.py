#!/usr/bin/env python3
"""Score the inference records of one commit: the benchmark.

    python benchmark/score.py                     # the records of the current commit
    python benchmark/score.py --commit <hash>     # any commit with records in $PERO_INFERENCE

Writes <records>/benchmark/: cer.tsv, layout.tsv and report.md. Run it again after more units were inferred; it scores
whatever is there.

Text (ocr/ and lm/ units): on every line of every test list, CER = edit distance / reference length, summed over the
list's lines. Both texts are NFC with single spaces and no space at the start or end (the references are stored that
way). A line the model removed (the PAGE_PARSER confidence filter) counts as an empty transcription.

Layout (layout/ units): the annotated lines of the test lists are the reference (the service's layout; only their
baselines are known to be right, regions are not annotated). A reference baseline is sampled at 20 points; a point is
covered by a detected line if its distance to the detected baseline is below half the reference line's height and it
projects inside the detected baseline. The best detected line of a reference line is the one that covers most points.
    found       the share of reference lines whose best detected line covers at least half of them
    coverage    the mean share of points the best detected line covers
    offset      the mean distance of the covered points to the detected baseline, in reference line heights
Lines detected where no annotated line is are not scored: most pages have lines nobody annotated.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import unicodedata

import numpy as np
import Levenshtein

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTS = 20


def read_tsv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'))


def clean(text):
    return ' '.join(unicodedata.normalize('NFC', text or '').split())


def load_lists(data):
    """{list name: [(sample_id, line_id, reference text)]}."""
    pages = read_tsv(os.path.join(data, 'metadata.tsv'))
    by_image = {p['image_id']: p['sample_id'] for p in pages}
    lists = {}
    folder = os.path.join(data, 'test_lists')
    for f in sorted(os.listdir(folder)):
        rows = []
        with open(os.path.join(folder, f), encoding='utf-8') as lines:
            for line in lines:
                key, _, text = line.rstrip('\n').partition(' ')
                key = key[:-len('.jpg')]
                image_id = key[:36] if key[:36] in by_image else next(i for i in by_image if key.startswith(i + '-'))
                rows.append((by_image[image_id], key[len(image_id) + 1:], text))
        lists[f[:-len('.test_list')]] = rows
    return lists, pages


def score_text(records, lists):
    rows = []
    for kind in ('ocr', 'lm'):
        folder = os.path.join(records, kind)
        if not os.path.isdir(folder):
            continue
        for unit in sorted(os.listdir(folder)):
            if not os.path.exists(os.path.join(folder, unit, 'unit.json')):
                continue
            hyp = {(r['sample_id'], r['line_id']): r for r in read_tsv(os.path.join(folder, unit, 'lines.tsv'))}
            for name, refs in lists.items():
                n = chars = errors = removed = 0
                for sample_id, line_id, ref in refs:
                    r = hyp.get((sample_id, line_id))
                    if r is None:  # an LM pair that does not decode this list's pages
                        continue
                    n += 1
                    removed += r['kept'] == '0'
                    chars += len(ref)
                    errors += Levenshtein.distance(clean(r['text']), ref)
                if n:
                    rows.append(dict(kind=kind, unit=unit, list=name, lines=n, chars=chars, errors=errors,
                                     cer=round(100 * errors / max(1, chars), 3), removed_lines=removed))
    return rows


def baselines(path):
    from pero_ocr.core.layout import PageLayout
    layout = PageLayout(file=path)
    return {line.id: (np.asarray(line.baseline, dtype=float), max(1.0, float(sum(line.heights))))
            for line in layout.lines_iterator() if line.baseline is not None and len(line.baseline) > 1}


def sample_polyline(points, n=POINTS):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, points[:, 0]), np.interp(t, cum, points[:, 1])], axis=1)


def coverage(points, height, detected):
    """(share of points covered, mean distance of covered points / height) for the best detected line."""
    best = (0.0, None)
    for base in detected:
        a, b = base[:-1], base[1:]
        d = b - a
        length2 = np.maximum((d ** 2).sum(1), 1e-9)
        t = ((points[:, None, :] - a[None]) * d[None]).sum(2) / length2[None]  # points x segments
        tc = np.clip(t, 0, 1)
        dist = np.linalg.norm(points[:, None, :] - (a[None] + tc[..., None] * d[None]), axis=2)
        nearest = dist.argmin(1)
        inside = (t[np.arange(len(points)), nearest] >= -1e-6) & (t[np.arange(len(points)), nearest] <= 1 + 1e-6)
        covered = inside & (dist.min(1) < height / 2)
        share = covered.mean()
        if share > best[0]:
            best = (share, float(dist.min(1)[covered].mean() / height))
    return best


def score_layout(records, lists, pages, data):
    folder = os.path.join(records, 'layout')
    if not os.path.isdir(folder):
        return []
    xml = {p['sample_id']: os.path.join(data, p['page_xml']) for p in pages}
    by_page = {}
    for name, refs in lists.items():
        for sample_id, line_id, _ in refs:
            by_page.setdefault(sample_id, set()).add(line_id)
    reference = {s: baselines(xml[s]) for s in by_page}
    rows = []
    for unit in sorted(os.listdir(folder)):
        if not os.path.exists(os.path.join(folder, unit, 'unit.json')):
            continue
        result = {}
        for sample_id, line_ids in by_page.items():
            path = os.path.join(folder, unit, 'page_xml', sample_id + '.xml')
            if not os.path.exists(path):  # records of a trial (--pages)
                continue
            detected = [b for b, _ in baselines(path).values()]
            for line_id in line_ids:
                if line_id in reference[sample_id]:
                    base, height = reference[sample_id][line_id]
                    result[(sample_id, line_id)] = coverage(sample_polyline(base), height, detected)
        for name, refs in lists.items():
            scored = [result[(s, l)] for s, l, _ in refs if (s, l) in result]
            if not scored:
                continue
            offsets = [o for c, o in scored if o is not None]
            rows.append(dict(unit=unit, list=name, lines=len(scored),
                             found=round(100 * float(np.mean([c >= 0.5 for c, _ in scored])), 2),
                             coverage=round(100 * float(np.mean([c for c, _ in scored])), 2),
                             offset=round(float(np.mean(offsets)), 4) if offsets else ''))
    return rows


def write_tsv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def table(rows, row_key, value, lists):
    names = sorted({r[row_key] for r in rows})
    cols = [n for n in lists if any(r['list'] == n for r in rows)]
    cell = {(r[row_key], r['list']): r[value] for r in rows}
    out = ['| ' + ' | '.join([row_key] + cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for n in names:
        out.append('| ' + ' | '.join([n] + [f'{cell[(n, c)]}' if (n, c) in cell else '' for c in cols]) + ' |')
    return out


def write_report(path, records, text_rows, layout_rows, lists):
    with open(os.path.join(records, 'environment.json')) as f:
        env = json.load(f)
    lines = [f'# Benchmark of pero-ocr {os.path.basename(records)}', '',
             f"Python {env['python']}, torch {env['packages'].get('torch')}, GPU {env['gpu']}. "
             f"{len(lists)} test lists, {sum(len(v) for v in lists.values())} lines.", '',
             '## CER [%] of the OCR models', '']
    lines += table([dict(r, model=r['unit']) for r in text_rows if r['kind'] == 'ocr'], 'model', 'cer', lists)
    if any(r['kind'] == 'lm' for r in text_rows):
        lines += ['', '## CER [%] with a language model (on the pages of the pairing\'s lists)', '']
        lines += table([dict(r, pair=r['unit']) for r in text_rows if r['kind'] == 'lm'], 'pair', 'cer', lists)
    removed = [r for r in text_rows if r['removed_lines']]
    if removed:
        lines += ['', '## Lines removed by the confidence filter (counted as empty)', '']
        lines += table([dict(r, model=r['unit']) for r in removed], 'model', 'removed_lines', lists)
    if layout_rows:
        lines += ['', '## Layout: annotated lines found [%]', '']
        lines += table([dict(r, model=r['unit']) for r in layout_rows], 'model', 'found', lists)
        lines += ['', '## Layout: mean coverage of annotated baselines [%]', '']
        lines += table([dict(r, model=r['unit']) for r in layout_rows], 'model', 'coverage', lists)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=os.environ.get('PERO_TEST_DATA'))
    p.add_argument('--results', default=os.environ.get('PERO_INFERENCE'))
    p.add_argument('--commit', help='records folder name (default: the current commit)')
    args = p.parse_args()
    if not args.data or not args.results:
        sys.exit('--data and --results (or PERO_TEST_DATA and PERO_INFERENCE) are required')
    commit = args.commit or subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], check=True, capture_output=True,
                                           text=True).stdout.strip()
    records = os.path.join(args.results, commit)
    if not os.path.isdir(records):
        sys.exit(f'no records in {records}')
    lists, pages = load_lists(args.data)
    text_rows = score_text(records, lists)
    layout_rows = score_layout(records, lists, pages, args.data)
    out = os.path.join(records, 'benchmark')
    os.makedirs(out, exist_ok=True)
    write_tsv(os.path.join(out, 'cer.tsv'), text_rows)
    write_tsv(os.path.join(out, 'layout.tsv'), layout_rows)
    write_report(os.path.join(out, 'report.md'), records, text_rows, layout_rows, lists)
    print(os.path.join(out, 'report.md'))


if __name__ == '__main__':
    main()
