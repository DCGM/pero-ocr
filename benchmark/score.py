#!/usr/bin/env python3
"""Score the inference records of one commit against the test lists of each benchmark.

    python benchmark/score.py                          # the records of the current commit, every benchmark
    python benchmark/score.py --commit <hash>          # any commit with records in $PERO_INFERENCE
    python benchmark/score.py --benchmarks main        # only some benchmarks
    python benchmark/score.py --out ~/bench/try1       # the records of a free run (infer.py --out)

Reads the records of each benchmark, <commit>/<benchmark>/ (or <out>/<benchmark>/), and <data>/<benchmark>/test_lists/,
writes <commit>/<benchmark>/score/: cer.tsv,
layout.tsv and report.md, and lines/<OCR or DECODER>/<unit>/<test list>.txt: the unit's output on exactly the lines
of the test list, in its order and format, to compare with the test list (or another unit's) line by line, e.g. in
meld. Run it again after more units were inferred; it scores whatever is there.

A test list has one line per row, "<page>-<line_id>.jpg <reference text>", where <page> is the page's file name in
the benchmark.

A unit is scored on a test list when it holds every page of the list (the pages whose files it has); otherwise its
score on that list is NA. Which pages were inferred is up to infer.py and its registry; this script does not read it.

Text (OCR/ and DECODER/ units): on every line of every test list, CER = edit distance / reference length, summed over
the list's lines. Both texts are NFC with single spaces and no space at the start or end (the references are stored
that way). A line the engine did not output (e.g. removed by the PAGE_PARSER confidence filter) counts as an empty
transcription.
Two versions: CER against the references as transcribed, and modernized CER, where both the reference and the output
are rewritten by the modern German rules of the training data (a vowel with a combining small e above, U+0364, to the
vowel with umlaut or diaeresis, and long s to s; the "modern_german" test sets of the PERO training scripts). Models
that write long s and models that write s are then scored alike; the modernized CER is what the older benchmarks
reported.

Layout (LAYOUT_PARSER/ units): the annotated lines of the test lists are the reference (the benchmark's PAGE XML; only
their baselines are known to be right, regions are not annotated). A reference baseline is sampled at 20 points; a
point is covered by a detected line if its distance to the detected baseline is below half the reference line's
height and it projects inside the detected baseline. The best detected line of a reference line is the one that
covers most points.
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

from records import held_pages, line_key, read_page

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTS = 20


def clean(text):
    return ' '.join(unicodedata.normalize('NFC', text or '').split())


MODERN = [(v + '\u0364', u) for v, u in zip('AEIOUaeiou', 'ÄËÏÖÜäëïöü')] + [('ſ', 's')]


def modernize(text):
    """The modern German rules of the training data's modern_german test sets."""
    for old, new in MODERN:
        text = text.replace(old, new)
    return text


def load_lists(dataset):
    """{list name: [(page, line_id, reference text)]} of <benchmark>/test_lists."""
    pages = sorted({os.path.splitext(f)[0] for f in os.listdir(os.path.join(dataset, 'page_xml'))}, key=len,
                   reverse=True)
    folder = os.path.join(dataset, 'test_lists')
    lists = {}
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.test_list'):
            continue
        rows = []
        with open(os.path.join(folder, f), encoding='utf-8') as lines:
            for line in lines:
                key, _, text = line.rstrip('\n').partition(' ')
                key = key[:-len('.jpg')] if key.endswith('.jpg') else key
                page = next((p for p in pages if key.startswith(p + '-')), None)
                if page is None:
                    sys.exit(f'{os.path.join(folder, f)}: {key}: no page of the benchmark')
                rows.append((page, key[len(page) + 1:], text))
        lists[f[:-len('.test_list')]] = rows
    return lists


def list_pages(lists):
    return {name: {page for page, _, _ in refs} for name, refs in lists.items()}


def score_text(records, lists, out):
    """A row for every text unit and test list; NA (cer None) when the unit does not hold every page of the list.
    Writes the unit's output on each list it holds to <out>/lines/<kind>/<unit>/<list>.txt."""
    rows = []
    pages_of = list_pages(lists)
    for kind in ('OCR', 'DECODER'):
        folder = os.path.join(records, kind)
        if not os.path.isdir(folder):
            continue
        for unit in sorted(os.listdir(folder)):
            if not os.path.exists(os.path.join(folder, unit, 'engine.json')):
                continue
            held = held_pages(os.path.join(folder, unit))
            hyp = {}
            for page in set().union(*pages_of.values()) & held:
                for line_id, text, conf in read_page(os.path.join(folder, unit), page):
                    hyp[(page, line_id)] = text
            for name, refs in lists.items():
                if not pages_of[name] <= held:
                    rows.append(dict(kind=kind, unit=unit, list=name, lines=0, chars='', errors='', cer=None,
                                     chars_modern='', errors_modern='', cer_modern=None, missing_lines=''))
                    continue
                n = chars = errors = chars_m = errors_m = missing = 0
                compare = []
                for page, line_id, ref in refs:
                    hyp_text = hyp.get((page, line_id))
                    missing += hyp_text is None
                    hyp_text = hyp_text or ''
                    one_line = hyp_text.replace('\r', ' ').replace('\n', ' ')  # the test-list format: a line per line
                    compare.append(f"{line_key(page, line_id)} {one_line}\n")
                    n += 1
                    text = clean(hyp_text)
                    chars += len(ref)
                    errors += Levenshtein.distance(text, ref)
                    chars_m += len(modernize(ref))
                    errors_m += Levenshtein.distance(modernize(text), modernize(ref))
                lines_folder = os.path.join(out, 'lines', kind, unit)
                os.makedirs(lines_folder, exist_ok=True)
                with open(os.path.join(lines_folder, name + '.txt'), 'w', encoding='utf-8', newline='') as f:
                    f.write(''.join(compare))
                rows.append(dict(kind=kind, unit=unit, list=name, lines=n, chars=chars, errors=errors,
                                 cer=round(100 * errors / max(1, chars), 3), chars_modern=chars_m,
                                 errors_modern=errors_m, cer_modern=round(100 * errors_m / max(1, chars_m), 3),
                                 missing_lines=missing))
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


def score_layout(records, lists, dataset):
    folder = os.path.join(records, 'LAYOUT_PARSER')
    if not os.path.isdir(folder):
        return []
    by_page = {}
    for refs in lists.values():
        for page, line_id, _ in refs:
            by_page.setdefault(page, set()).add(line_id)
    reference = {p: baselines(os.path.join(dataset, 'page_xml', p + '.xml')) for p in by_page}
    rows = []
    pages_of = list_pages(lists)
    for unit in sorted(os.listdir(folder)):
        if not os.path.exists(os.path.join(folder, unit, 'engine.json')):
            continue
        xml = os.path.join(folder, unit, 'page_xml')
        held = {f[:-4] for f in os.listdir(xml) if f.endswith('.xml') and not f.startswith('.')} if os.path.isdir(
            xml) else set()
        result = {}
        for page, line_ids in by_page.items():
            if page not in held:
                continue
            detected = [b for b, _ in baselines(os.path.join(xml, page + '.xml')).values()]
            for line_id in line_ids:
                if line_id in reference[page]:
                    base, height = reference[page][line_id]
                    result[(page, line_id)] = coverage(sample_polyline(base), height, detected)
        for name, refs in lists.items():
            if not pages_of[name] <= held:
                rows.append(dict(unit=unit, list=name, lines=0, found=None, coverage=None, offset=''))
                continue
            scored = [result[(p, l)] for p, l, _ in refs if (p, l) in result]
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
        w.writerows([{k: 'NA' if v is None else v for k, v in r.items()} for r in rows])


def table(rows, row_key, value, lists):
    """A unit per row, a test list per column; NA where the unit does not hold the list's pages. Lists no unit
    scores are left out."""
    names = sorted({r[row_key] for r in rows})
    cols = [n for n in lists if any(r['list'] == n and r[value] is not None for r in rows)]
    cell = {(r[row_key], r['list']): r[value] for r in rows}
    out = ['| ' + ' | '.join([row_key] + cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for n in names:
        out.append('| ' + ' | '.join([n] + ['NA' if cell.get((n, c), '') is None else f"{cell.get((n, c), '')}"
                                            for c in cols]) + ' |')
    return out


def write_report(path, commit, dataset_name, env, text_rows, layout_rows, lists):
    lines = [f'# Benchmark {commit}: {dataset_name}', '',
             f"Python {env['python']}, torch {env['packages'].get('torch')}, GPU {env['gpu']}. "
             f"{len(lists)} test lists, {sum(len(v) for v in lists.values())} lines.", '',
             ]
    for value, title in (('cer', 'CER [%]'), ('cer_modern', 'Modernized CER [%] (long s as s, combining e as umlaut)')):
        lines += [f'## {title} of the OCR engines', '']
        lines += table([dict(r, ocr=r['unit']) for r in text_rows if r['kind'] == 'OCR'], 'ocr', value, lists)
        if any(r['kind'] == 'DECODER' for r in text_rows):
            lines += ['', f'## {title} with a decoder', '']
            lines += table([dict(r, decoder=r['unit']) for r in text_rows if r['kind'] == 'DECODER'], 'decoder',
                           value, lists)
        lines.append('')
    missing = [r for r in text_rows if r['missing_lines']]
    if missing:
        lines += ['', '## Lines of the test lists the engine did not output (counted as empty)', '']
        lines += table([dict(r, unit=f"{r['kind']}/{r['unit']}") for r in missing], 'unit', 'missing_lines', lists)
    if layout_rows:
        lines += ['', '## Layout: annotated lines found [%]', '']
        lines += table([dict(r, layout=r['unit']) for r in layout_rows], 'layout', 'found', lists)
        lines += ['', '## Layout: mean coverage of annotated baselines [%]', '']
        lines += table([dict(r, layout=r['unit']) for r in layout_rows], 'layout', 'coverage', lists)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=os.environ.get('PERO_TEST_DATA'))
    p.add_argument('--results', default=os.environ.get('PERO_INFERENCE'))
    p.add_argument('--commit', help='records folder name (default: the current commit)')
    p.add_argument('--out', help='the folder of a free run (infer.py --out) instead of a commit')
    p.add_argument('--benchmarks', nargs='*', help='only these benchmarks')
    args = p.parse_args()
    if not args.data or not (args.results or args.out):
        sys.exit('--data and --results or --out (or PERO_TEST_DATA and PERO_INFERENCE) are required')
    if args.out:
        root, label = args.out, os.path.basename(os.path.normpath(args.out))
    else:
        label = args.commit or subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], check=True,
                                              capture_output=True, text=True).stdout.strip()
        root = os.path.join(args.results, label)
    if not os.path.isdir(root):
        sys.exit(f'no records in {root}')
    found = sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, 'benchmark.json')))
    missing = sorted(set(args.benchmarks or []) - set(found))
    if missing:
        sys.exit(f'no records of {", ".join(missing)} in {root}')
    if not found:
        sys.exit(f'no records in {root} (a benchmark\'s records are a folder with benchmark.json)')
    for name in found:
        records, dataset = os.path.join(root, name), os.path.join(args.data, name)
        if args.benchmarks and name not in args.benchmarks:
            continue
        if not os.path.isdir(os.path.join(dataset, 'test_lists')):
            print(f'{name}: no test lists in {dataset}, not scored')
            continue
        with open(os.path.join(records, 'environment.json')) as f:
            env = json.load(f)
        lists = load_lists(dataset)
        out = os.path.join(records, 'score')
        os.makedirs(out, exist_ok=True)
        text_rows = score_text(records, lists, out)
        layout_rows = score_layout(records, lists, dataset)
        write_tsv(os.path.join(out, 'cer.tsv'), text_rows)
        write_tsv(os.path.join(out, 'layout.tsv'), layout_rows)
        write_report(os.path.join(out, 'report.md'), label, name, env, text_rows, layout_rows, lists)
        print(os.path.join(out, 'report.md'))


if __name__ == '__main__':
    main()
