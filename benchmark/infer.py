#!/usr/bin/env python3
"""Run the benchmark models on the benchmark pages and store what they produce, under the pero-ocr commit.

    python benchmark/infer.py                        # every unit that is not in <results>/<commit>/ yet
    python benchmark/infer.py --units ocr/universal.ctc.2025-03-19 lm/
    python benchmark/infer.py --list                 # what would run

Inference is not a test: it only records what the code at this commit produces. benchmark/score.py turns the records
into the benchmark, and test/inference/ compares the records of two commits.

Inputs (environment variables or options):
    PERO_TEST_DATA     the benchmark sample: metadata.tsv, <access>/images, <access>/page_xml, test_lists/
    PERO_TEST_MODELS   the models: models.tsv (name, kind, folder), pairings.tsv (lm, ocr, style), the model folders
    PERO_INFERENCE     where the records go: <PERO_INFERENCE>/<commit>/

Units (one folder each, complete when it holds unit.json; a run only fills in the missing ones):
    layout/<model>        the layout model on every page image: page_xml/<sample_id>.xml
    crops/<setup>         the line cropper on the sample's PAGE XML, one LMDB per [LINE_CROPPER] setup of the OCR
                          models: key "<sample_id>/<line_id>", value the crop as PNG
    ocr/<model>           the OCR model (with its line cropper and PAGE_PARSER settings) on every line of the sample's
                          PAGE XML: lines.tsv, logprobs.npz (full log-probabilities of a fixed subset of lines, the
                          same subset for every model)
    lm/<lm>+<ocr>         the LM decoder on the logits of the OCR model, as the service that pairs them runs it
                          (style "scribble": a separate stage with the LM package's alphabet; "pero": OCR and LM in one
                          config, alphabet of the OCR model), on the pages of the pairing's lists (an Arabic LM decodes
                          the Arabic pages only): lines.tsv

Every page is processed by a fresh numpy random state seeded from its sample id, so a rerun gives the same result
(the layout engine breaks ties with np.random, issue #100).
"""
import argparse
import configparser
import copy
import csv
import datetime
import hashlib
import io
import json
import logging
import multiprocessing
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger('infer')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The pero web adds these to every OCR model config (its config_base_ocr.ini); the scribble packages carry their own.
PERO_WEB_PAGE_PARSER = {'RUN_LAYOUT_PARSER': 'no', 'RUN_LINE_CROPPER': 'yes', 'RUN_OCR': 'yes', 'RUN_DECODER': 'no'}
LOGITS_SUBSET = 64  # one line in LOGITS_SUBSET keeps its logits (chosen by a hash of its key, the same for every model)
LMDB_MAP_SIZE = 1 << 40


# ----------------------------------------------------------------------------------------------------------------------
# Inputs


def read_tsv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE))


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def page_seed(sample_id):
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)


def in_logits_subset(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % LOGITS_SUBSET == 0


class Sample:
    def __init__(self, root):
        self.root = root
        self.all_pages = read_tsv(os.path.join(root, 'metadata.tsv'))
        self.pages = self.all_pages  # the pages to process (a trial takes the first few)

    def image(self, page):
        return os.path.join(self.root, page['image'])

    def page_xml(self, page):
        return os.path.join(self.root, page['page_xml'])

    def page_lists(self):
        """{sample_id: set of test lists with a line of the page}."""
        by_image = {p['image_id']: p['sample_id'] for p in self.all_pages}
        result = {p['sample_id']: set() for p in self.all_pages}
        folder = os.path.join(self.root, 'test_lists')
        for f in sorted(os.listdir(folder)):
            with open(os.path.join(folder, f), encoding='utf-8') as lines:
                for line in lines:
                    key = line.split(' ', 1)[0]
                    image_id = key[:36] if key[:36] in by_image else next(i for i in by_image if key.startswith(i + '-'))
                    result[by_image[image_id]].add(f[:-len('.test_list')])
        return result

    def in_scope(self, spec):
        """Sample ids of the pages a pairing's "lists" column selects: "*" every page, "A,B" pages in list A or B,
        "-A,B" pages in neither."""
        lists = self.page_lists()
        if spec == '*':
            return set(lists)
        names = set(spec.lstrip('-').split(','))
        return {s for s, l in lists.items() if bool(l & names) != spec.startswith('-')}

    def fingerprint(self):
        files = ['metadata.tsv'] + sorted(os.path.join('test_lists', f) for f in os.listdir(
            os.path.join(self.root, 'test_lists')))
        return {f: md5_file(os.path.join(self.root, f)) for f in files}


def read_config(folder, add_page_parser=True):
    config = configparser.ConfigParser()
    config.read(os.path.join(folder, 'config.ini'))
    if add_page_parser and not config.has_section('PAGE_PARSER'):
        config['PAGE_PARSER'] = PERO_WEB_PAGE_PARSER
    return config


def config_text(config):
    out = io.StringIO()
    config.write(out)
    return out.getvalue()


def cropper_setup(config):
    """The [LINE_CROPPER] settings as a name, e.g. interp2_scale1.15_height48."""
    section = config['LINE_CROPPER']
    return (f"interp{section.getint('INTERP')}_scale{section.getfloat('LINE_SCALE'):g}"
            f"_height{section.getint('LINE_HEIGHT')}")


class Models:
    def __init__(self, root):
        self.root = root
        self.models = {r['name']: r for r in read_tsv(os.path.join(root, 'models.tsv'))}
        self.pairings = read_tsv(os.path.join(root, 'pairings.tsv'))

    def folder(self, name):
        return os.path.join(self.root, self.models[name]['folder'])

    def of_kind(self, kind):
        return sorted(n for n, m in self.models.items() if m['kind'] == kind)

    def weights(self, name):
        """{file: md5} of the files of a model folder (configs included)."""
        folder = self.folder(name)
        return {f: md5_file(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
                if os.path.isfile(os.path.join(folder, f))}


# ----------------------------------------------------------------------------------------------------------------------
# Commit and environment


def git(*args):
    return subprocess.run(['git', '-C', REPO] + list(args), check=True, capture_output=True, text=True).stdout.strip()


def commit_folder_name(allow_dirty):
    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty and not allow_dirty:
        sys.exit('The working tree has uncommitted changes; the records would not belong to the commit. Commit them '
                 'or pass --allow-dirty (the records then go to <commit>-dirty).\n' + dirty)
    return commit + ('-dirty' if dirty else ''), commit


def environment():
    import importlib.metadata as metadata
    import torch
    env = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {d.metadata['Name'].lower(): d.version for d in metadata.distributions() if d.metadata['Name']},
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    env['packages'].pop('pero-ocr', None)  # the commit identifies pero-ocr
    return env


ENV_KEYS = ('python', 'packages', 'torch_cuda', 'cudnn', 'gpu')


def check_environment(out, env, sample_fingerprint, force):
    path = os.path.join(out, 'environment.json')
    record = dict(env, sample=sample_fingerprint)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(record, f, indent=1, sort_keys=True)
        return
    with open(path) as f:
        old = json.load(f)
    differ = [k for k in ENV_KEYS + ('sample',) if old.get(k) != record.get(k)]
    if differ and not force:
        details = []
        for k in differ:
            if k == 'packages':
                a, b = old.get(k, {}), record[k]
                details += [f'  {p}: {a.get(p)} -> {b.get(p)}' for p in sorted(set(a) | set(b)) if a.get(p) != b.get(p)]
            else:
                details.append(f'  {k}: {old.get(k)} -> {record.get(k)}')
        sys.exit(f'{out} was inferred in another environment or on other data; adding to it would mix them '
                 f'(--force-environment to do it anyway):\n' + '\n'.join(details))


# ----------------------------------------------------------------------------------------------------------------------
# Units


class Unit:
    def __init__(self, kind, name, out):
        self.kind, self.name = kind, name
        self.final = os.path.join(out, kind, name)
        self.partial = self.final + '.partial'

    @property
    def id(self):
        return f'{self.kind}/{self.name}'

    def done(self):
        return os.path.exists(os.path.join(self.final, 'unit.json'))

    def start(self):
        shutil.rmtree(self.partial, ignore_errors=True)
        os.makedirs(self.partial)
        self.started = time.time()
        self.started_utc = utc_now()

    def finish(self, info):
        info = dict(unit=self.id, started_utc=self.started_utc, finished_utc=utc_now(),
                    seconds=round(time.time() - self.started, 1), **info)
        with open(os.path.join(self.partial, 'unit.json'), 'w') as f:
            json.dump(info, f, indent=1, ensure_ascii=False)
        shutil.rmtree(self.final, ignore_errors=True)
        os.rename(self.partial, self.final)
        logger.info(f'{self.id} done in {info["seconds"]:.0f} s')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def plan(models, out):
    """All units, in run order."""
    units = [Unit('layout', n, out) for n in models.of_kind('layout')]
    setups = sorted({cropper_setup(read_config(models.folder(n))) for n in models.of_kind('ocr')})
    units += [Unit('crops', s, out) for s in setups]
    units += [Unit('ocr', n, out) for n in models.of_kind('ocr')]
    units += [Unit('lm', f"{p['lm']}+{p['ocr']}", out) for p in models.pairings]
    return units


# ----------------------------------------------------------------------------------------------------------------------
# Page reading (a thread reads the next pages while the GPU works)


def read_pages(sample, pages, with_layout=True, depth=3):
    from pero_ocr.core.layout import PageLayout
    q = queue.Queue(maxsize=depth)

    def worker():
        for page in pages:
            image = cv2.imread(sample.image(page), 1)
            layout = PageLayout(file=sample.page_xml(page)) if with_layout else None
            q.put((page, image, layout))
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


# ----------------------------------------------------------------------------------------------------------------------
# Layout


def run_layout(unit, models, sample):
    from pero_ocr.core.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import PageParser
    folder = models.folder(unit.name)
    config = read_config(folder)
    parser = PageParser(config, config_path=folder)
    os.makedirs(os.path.join(unit.partial, 'page_xml'))
    n_lines = 0
    for page, image, _ in read_pages(sample, sample.pages, with_layout=False):
        np.random.seed(page_seed(page['sample_id']))
        layout = PageLayout(id=page['image_id'], page_size=(image.shape[0], image.shape[1]))
        layout = parser.process_page(image, layout)
        layout.to_pagexml(os.path.join(unit.partial, 'page_xml', page['sample_id'] + '.xml'))
        n_lines += sum(1 for _ in layout.lines_iterator())
    del parser
    return dict(model=unit.name, weights=models.weights(unit.name), config=config_text(config),
                pages=len(sample.pages), lines=n_lines)


# ----------------------------------------------------------------------------------------------------------------------
# Crops (CPU, a process pool; runs while the GPU units run)


def _crop_page(args):
    image_path, xml_path, sample_id, setups = args
    from pero_ocr.core.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import line_cropper_factory
    np.random.seed(page_seed(sample_id))
    image = cv2.imread(image_path, 1)
    result = {}
    for setup, section in setups.items():
        config = configparser.ConfigParser()
        config['LINE_CROPPER'] = section
        cropper = line_cropper_factory(config['LINE_CROPPER'])
        layout = PageLayout(file=xml_path)
        layout = cropper.process_page(image, layout)
        result[setup] = [(f'{sample_id}/{line.id}', cv2.imencode('.png', line.crop)[1].tobytes())
                         for line in layout.lines_iterator()]
    return result


def run_crops(units, models, sample, workers):
    import lmdb
    sections = {}
    for n in models.of_kind('ocr'):
        config = read_config(models.folder(n))
        sections.setdefault(cropper_setup(config), dict(config['LINE_CROPPER']))
    setups = {u.name: sections[u.name] for u in units}
    envs = {}
    for u in units:
        u.start()
        envs[u.name] = lmdb.open(os.path.join(u.partial, 'crops.lmdb'), map_size=LMDB_MAP_SIZE, subdir=True)
    jobs = [(sample.image(p), sample.page_xml(p), p['sample_id'], setups) for p in sample.pages]
    counts = {u.name: 0 for u in units}
    with multiprocessing.get_context('spawn').Pool(workers, initializer=os.nice, initargs=(10,)) as pool:
        for result in pool.imap(_crop_page, jobs, chunksize=2):
            for setup, items in result.items():
                with envs[setup].begin(write=True) as txn:
                    for key, png in items:
                        txn.put(key.encode(), png)
                counts[setup] += len(items)
    for u in units:
        envs[u.name].close()
        u.finish(dict(setup=dict(setups[u.name]), pages=len(sample.pages), lines=counts[u.name], format='png'))


# ----------------------------------------------------------------------------------------------------------------------
# OCR and LM


class LineWriter:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8', newline='')
        self.w = csv.writer(self.f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        self.w.writerow(['sample_id', 'line_id', 'text', 'confidence', 'kept'])
        self.n = 0

    def page(self, sample_id, line_ids, layout):
        """Every line of the input; kept=0 for lines the PAGE_PARSER filter removed (no text)."""
        lines = {line.id: line for line in layout.lines_iterator()}
        for line_id in line_ids:
            line = lines.get(line_id)
            if line is None:
                self.w.writerow([sample_id, line_id, '', '', 0])
            else:
                text = (line.transcription or '').replace('\t', ' ').replace('\n', ' ')
                conf = '' if line.transcription_confidence is None else f'{line.transcription_confidence:.6f}'
                self.w.writerow([sample_id, line_id, text, conf, 1])
            self.n += 1

    def close(self):
        self.f.close()


def decoder_config(models, pairing, ocr_folder):
    lm_folder = models.folder(pairing['lm'])
    config = read_config(lm_folder, add_page_parser=False)
    config['PAGE_PARSER'] = {'RUN_LAYOUT_PARSER': 'no', 'RUN_LINE_CROPPER': 'no', 'RUN_OCR': 'no',
                             'RUN_DECODER': 'yes'}
    if pairing['style'] == 'pero':
        config['OCR'] = {'OCR_JSON': os.path.join(ocr_folder, read_config(ocr_folder)['OCR']['OCR_JSON'])}
    elif pairing['style'] != 'scribble' or not config.has_option('OCR', 'OCR_JSON'):
        raise ValueError(f'pairing {pairing}: style must be "pero", or "scribble" with an LM package that has '
                         f'[OCR] OCR_JSON')
    return config, lm_folder


def run_ocr(ocr_name, ocr_unit, lm_units, models, sample):
    """One OCR model on every page; the LM pairs of it decode its logits on the way."""
    import scipy.sparse
    from pero_ocr.document_ocr.page_parser import PageParser, prepare_dense_logits
    folder = models.folder(ocr_name)
    config = read_config(folder)
    parser = PageParser(config, config_path=folder)
    decoders = []
    for unit, pairing in lm_units:
        dconfig, lm_folder = decoder_config(models, pairing, folder)
        scope = sample.in_scope(pairing['lists'])
        decoders.append((unit, pairing, dconfig, scope, PageParser(dconfig, config_path=lm_folder)))
        unit.start()
    if ocr_unit:
        ocr_unit.start()
        writer = LineWriter(os.path.join(ocr_unit.partial, 'lines.tsv'))
        logits = {}
    lm_writers = [LineWriter(os.path.join(u.partial, 'lines.tsv')) for u, *_ in decoders]
    pages = sample.pages if ocr_unit else [p for p in sample.pages if any(p['sample_id'] in d[3] for d in decoders)]
    for page, image, layout in read_pages(sample, pages):
        sid = page['sample_id']
        np.random.seed(page_seed(sid))
        line_ids = [line.id for line in layout.lines_iterator()]
        for line in layout.lines_iterator():
            line.transcription = None
        layout = parser.process_page(image, layout)
        if ocr_unit:
            writer.page(sid, line_ids, layout)
            for line in layout.lines_iterator():
                if line.logits is not None and in_logits_subset(f'{sid}/{line.id}'):
                    logits[f'{sid}/{line.id}'] = prepare_dense_logits(line).astype(np.float32)
        for (unit, pairing, _, scope, decoder), lm_writer in zip(decoders, lm_writers):
            if sid not in scope:
                continue
            decoded = copy.deepcopy(layout)
            if pairing['style'] == 'scribble':  # the worker stores the logits and the LM stage loads them
                decoded.load_logits(layout.save_logits_bytes(missing_line_logits_ok=True))
            decoded = decoder.process_page(image, decoded)
            lm_writer.page(sid, line_ids, decoded)
    del parser
    if ocr_unit:
        writer.close()
        np.savez(os.path.join(ocr_unit.partial, 'logprobs.npz'), **logits)
        ocr_unit.finish(dict(model=ocr_name, weights=models.weights(ocr_name), config=config_text(config),
                             cropper_setup=cropper_setup(config), pages=len(sample.pages), lines=writer.n,
                             logprobs_lines=len(logits),
                             logprobs_subset=f'1/{LOGITS_SUBSET} by md5 of "<sample_id>/<line_id>"'))
    for (unit, pairing, dconfig, scope, _), lm_writer in zip(decoders, lm_writers):
        lm_writer.close()
        unit.finish(dict(lm=pairing['lm'], ocr=ocr_name, style=pairing['style'], lists=pairing['lists'],
                         lm_weights=models.weights(pairing['lm']), ocr_weights=models.weights(ocr_name),
                         decoder_config=config_text(dconfig), pages=len(scope & {p['sample_id'] for p in sample.pages}),
                         lines=lm_writer.n))


# ----------------------------------------------------------------------------------------------------------------------
# Report


def write_report(out, commit):
    rows = []
    for kind in ('layout', 'crops', 'ocr', 'lm'):
        folder = os.path.join(out, kind)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name, 'unit.json')
            if os.path.exists(path):
                with open(path) as f:
                    info = json.load(f)
                rows.append((kind, name, info.get('pages', ''), info.get('lines', ''), info['seconds'],
                             info['finished_utc']))
    with open(os.path.join(out, 'environment.json')) as f:
        env = json.load(f)
    lines = [f'# Inference of pero-ocr {commit}', '',
             f"Python {env['python']}, torch {env['packages'].get('torch')} (CUDA {env['torch_cuda']}), "
             f"numpy {env['packages'].get('numpy')}, opencv {env['packages'].get('opencv-python')}, GPU {env['gpu']}.",
             '', '| kind | unit | pages | lines | seconds | finished (UTC) |', '|---|---|---|---|---|---|']
    lines += [f'| {k} | {n} | {p} | {l} | {s:.0f} | {t} |' for k, n, p, l, s, t in rows]
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------------------------------------------------


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    return float('inf')


def groups(todo, models, sample):
    """Units that run in one process, most expensive first: a layout unit, an OCR unit, or an LM pair (which runs its
    OCR model again, on the pages of its lists only)."""
    result = []
    n = len(sample.pages)
    for u in todo:
        if u.kind == 'layout':
            result.append(([u], 0.5 * n))
        elif u.kind == 'ocr':
            result.append(([u], (1.6 if 'transformer' in u.name else 0.6) * n))
        elif u.kind == 'lm':
            pairing = next(p for p in models.pairings if f"{p['lm']}+{p['ocr']}" == u.name)
            result.append(([u], 6.0 * len(sample.in_scope(pairing['lists']) & {p['sample_id'] for p in sample.pages})))
    return [g for g, _ in sorted(result, key=lambda x: -x[1])]


def run_inline(todo, models, sample, out, folder_name, crop_workers):
    crops = [u for u in todo if u.kind == 'crops']
    crop_thread = None
    if crops:
        crop_thread = threading.Thread(target=run_crops, args=(crops, models, sample, crop_workers))
        crop_thread.start()
    for unit in (u for u in todo if u.kind == 'layout'):
        unit.start()
        unit.finish(run_layout(unit, models, sample))
    lm_todo = {u.id: u for u in todo if u.kind == 'lm'}
    for ocr_name in models.of_kind('ocr'):
        ocr_unit = next((u for u in todo if u.id == f'ocr/{ocr_name}'), None)
        lm_units = [(lm_todo[f'lm/{p["lm"]}+{ocr_name}'], p) for p in models.pairings
                    if p['ocr'] == ocr_name and f'lm/{p["lm"]}+{ocr_name}' in lm_todo]
        if ocr_unit or lm_units:
            logger.info(f'ocr {ocr_name}' + (f' with {len(lm_units)} LM' if lm_units else ''))
            run_ocr(ocr_name, ocr_unit, lm_units, models, sample)
    if crop_thread:
        crop_thread.join()


def run_parallel(todo, models, sample, out, args, folder_name):
    """Crops in this process (a CPU pool), every other unit in a process of its own, args.jobs at a time, at low CPU
    priority; a new process starts only when enough memory is free."""
    crops = [u for u in todo if u.kind == 'crops']
    crop_thread = None
    if crops:
        crop_thread = threading.Thread(target=run_crops, args=(crops, models, sample, args.crop_workers))
        crop_thread.start()
    pending = groups([u for u in todo if u.kind != 'crops'], models, sample)
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    base = [sys.executable, os.path.abspath(__file__), '--exact', '--data', args.data, '--models', args.models,
            '--results', args.results, '--crop-workers', '0'] + (['--pages', str(args.pages)] if args.pages else []) + \
           (['--allow-dirty'] if args.allow_dirty else []) + (['--force-environment'] if args.force_environment else [])
    lock, failed = threading.Lock(), []

    def worker():
        while True:
            with lock:
                if not pending:
                    return
                while mem_available_gb() < args.min_free_gb:
                    time.sleep(30)
                group = pending.pop(0)
                name = '_'.join(u.id.replace('/', '__') for u in group)
                logger.info(f'start {" ".join(u.id for u in group)}')
                log = open(os.path.join(out, 'logs', name + '.log'), 'w')
                proc = subprocess.Popen(base + ['--units'] + [u.id for u in group], stdout=log, stderr=subprocess.STDOUT,
                                        preexec_fn=lambda: os.nice(10))
                time.sleep(args.stagger)  # let it load its models before the memory check of the next one
            if proc.wait() != 0:
                failed.append(name)
                logger.error(f'{name} failed, see logs/{name}.log')
            log.close()
            write_report(out, folder_name)

    threads = [threading.Thread(target=worker) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if crop_thread:
        crop_thread.join()
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=os.environ.get('PERO_TEST_DATA'))
    p.add_argument('--models', default=os.environ.get('PERO_TEST_MODELS'))
    p.add_argument('--results', default=os.environ.get('PERO_INFERENCE'))
    p.add_argument('--units', nargs='*', default=[''], help='unit id prefixes to run, e.g. ocr/ lm/ layout/universal')
    p.add_argument('--exact', action='store_true', help='--units are exact unit ids; run them in this process')
    p.add_argument('--pages', type=int, help='only the first N pages (a trial; needs --results of its own)')
    p.add_argument('--jobs', type=int, default=1, help='units running at the same time, each in its own process')
    p.add_argument('--crop-workers', type=int, default=3)
    p.add_argument('--min-free-gb', type=float, default=5.0, help='start a unit process only with this much free RAM')
    p.add_argument('--stagger', type=float, default=45.0, help='seconds between starting two unit processes')
    p.add_argument('--list', action='store_true', help='show the units and whether they are done, run nothing')
    p.add_argument('--allow-dirty', action='store_true')
    p.add_argument('--force-environment', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    for name in ('data', 'models', 'results'):
        if not getattr(args, name):
            sys.exit(f'--{name} (or its environment variable) is required')

    sample, models = Sample(args.data), Models(args.models)
    if args.pages:
        if os.path.abspath(args.results) == os.path.abspath(os.environ.get('PERO_INFERENCE', '')):
            sys.exit('--pages makes incomplete records: give them a --results folder of their own')
        sample.pages = sample.pages[:args.pages]
    folder_name, commit = commit_folder_name(args.allow_dirty)
    out = os.path.join(args.results, folder_name)
    if args.exact:
        units = [u for u in plan(models, out) if u.id in set(args.units)]
    else:
        units = [u for u in plan(models, out) if any(u.id.startswith(prefix) for prefix in args.units)]
    if args.list:
        for u in units:
            print('done   ' if u.done() else 'missing', u.id)
        return
    todo = [u for u in units if not u.done()]
    if not todo:
        print('nothing to do')
        return
    os.makedirs(out, exist_ok=True)
    check_environment(out, environment(), sample.fingerprint(), args.force_environment)
    logger.info(f'{out}: {len(todo)} units to run')
    failed = []
    if args.exact or args.jobs <= 1:
        run_inline(todo, models, sample, out, folder_name, args.crop_workers)
    else:
        failed = run_parallel(todo, models, sample, out, args, folder_name)
    write_report(out, folder_name)
    if failed:
        sys.exit(f'failed: {", ".join(failed)}')
    logger.info('all units done')


if __name__ == '__main__':
    main()
