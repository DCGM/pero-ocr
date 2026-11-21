#!/usr/bin/env python3
"""Run the benchmark models on the benchmark pages and store what they produce, under the pero-ocr commit.

    python benchmark/infer.py                        # every unit that is not in <results>/<commit>/ yet
    python benchmark/infer.py --units ocr/universal.ctc.2025-03-19 lm/   # only these (rarely needed)
    python benchmark/infer.py --list                 # what would run
    python benchmark/infer.py --prune-logits         # delete the stored logits no LM pair still needs

Inference is not a test: it only records what the code at this commit produces. benchmark/score.py turns the records
into the benchmark, and test/inference/ compares the records of two commits.

Inputs (environment variables or options):
    PERO_TEST_DATA     the benchmark sample: metadata.tsv, <access>/images, <access>/page_xml, test_lists/
    PERO_TEST_MODELS   the registry: models.tsv (one row per model, see Models) and the model folders
    PERO_INFERENCE     where the records go: <PERO_INFERENCE>/<commit>/

Registering a model is adding its row to models.tsv; the next run of infer.py fills in what is missing. An LM row
names the OCR models whose results it decodes ("decodes"); the OCR results are not computed again for it.

Units (one folder each, complete when it holds unit.json; a run only fills in the missing ones):
    layout/<model>        the layout model on every page image: page_xml/<sample_id>.xml
    crops/<setup>         the line cropper on the sample's PAGE XML, one LMDB per [LINE_CROPPER] setup of the OCR
                          models: key "<sample_id>/<line_id>", value the crop as PNG (lossless)
    ocr/<model>[@<setup>] the OCR model (its PAGE_PARSER settings, on the stored crops of its cropper setup, or of
                          another setup listed in its registry row) on every line of
                          the sample's PAGE XML: lines.tsv (text, confidence, kept), logprobs.npz (full
                          log-probabilities of a fixed subset of lines, the same subset for every model). A model
                          that an LM decodes also keeps its full output of every page, as the ScribbleSense worker
                          stores it for the LM stage: page_xml/<sample_id>.xml and logits/<sample_id>.logits
                          (PageLayout.save_logits_bytes); --prune-logits deletes logits/ once all its LMs are done.
                          A model that differs from another only in a higher FILTER_CONFIDENT_LINES_THRESHOLD (same
                          files, same config otherwise) is not run: its unit is the other's lines with the lines of
                          confidence <= threshold removed, as PageParser removes them.
    lm/<lm>+<ocr>         the LM decoder on the stored output of ocr/<ocr> (which must be done; where that output is
                          not stored, pruned or never kept, the unit runs the OCR model again on its pages, requires
                          the same lines as ocr/<ocr>/lines.tsv and stores the output), as the service runs the LM
                          (lm_style "scribble": the LM package's alphabet; "pero": the alphabet of the OCR model,
                          [OCR] OCR_JSON), on the pages of its lm_lists (an Arabic LM decodes the Arabic pages
                          only): lines.tsv

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
import re
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


def cropper_section(setup):
    """The [LINE_CROPPER] settings of a setup name (the inverse of cropper_setup)."""
    m = re.fullmatch(r'interp(\d+)_scale([\d.]+)_height(\d+)(?:_categories(\S+))?', setup)
    if not m:
        sys.exit(f'{setup!r} is not a cropper setup (interp<INTERP>_scale<LINE_SCALE>_height<LINE_HEIGHT>, e.g. '
                 f'interp2_scale1.15_height48)')
    section = {'INTERP': m[1], 'LINE_SCALE': m[2], 'LINE_HEIGHT': m[3]}
    if m[4]:
        section['CATEGORIES'] = m[4]
    return section


def cropper_setup(config):
    """The [LINE_CROPPER] settings as a name, e.g. interp2_scale1.15_height48."""
    section = config['LINE_CROPPER']
    name = (f"interp{section.getint('INTERP')}_scale{section.getfloat('LINE_SCALE'):g}"
            f"_height{section.getint('LINE_HEIGHT')}")
    categories = section.get('CATEGORIES', '').replace(' ', '')
    return name + (f'_categories{categories}' if categories else '')


THRESHOLD = 'FILTER_CONFIDENT_LINES_THRESHOLD'
LM_STYLES = ('scribble', 'pero')


class Models:
    """The registry, models.tsv: one row per model.
        name, kind (layout, ocr, lm), folder (relative to the registry)
        croppers    ocr rows: more line cropper setups to run the model with, comma-separated setup names
                    (interp<INTERP>_scale<LINE_SCALE>_height<LINE_HEIGHT>); the model always runs with its own
                    [LINE_CROPPER] as ocr/<name>, with another setup as ocr/<name>@<setup>
        decodes     lm rows: the OCR runs whose results the LM decodes, comma-separated (<name> or <name>@<setup>)
        lm_style    lm rows: how the service runs the LM: "scribble" (a separate stage, the LM package's alphabet) or
                    "pero" (OCR and LM in one config, the OCR model's alphabet)
        lm_lists    lm rows: the pages it decodes: "*", "A,B" (pages in list A or B) or "-A,B" (in neither)
    Other columns are notes."""

    def __init__(self, root):
        self.root = root
        self.models = {r['name']: r for r in read_tsv(os.path.join(root, 'models.tsv'))}
        self.pairings = []
        self.ocr_runs = []
        for name in sorted(n for n, m in self.models.items() if m['kind'] == 'ocr'):
            own = cropper_setup(read_config(self.folder(name)))
            self.ocr_runs.append(name)
            for setup in (self.models[name].get('croppers') or '').split(','):
                setup = setup.strip()
                if setup and setup != own:
                    if cropper_section(setup)['LINE_HEIGHT'] != cropper_section(own)['LINE_HEIGHT']:
                        sys.exit(f'models.tsv: {name}: cropper {setup} has another LINE_HEIGHT than the model\'s '
                                 f'{own}; the OCR network takes crops of its own height only')
                    self.ocr_runs.append(f'{name}@{setup}')
        for name, m in sorted(self.models.items()):
            if m['kind'] not in ('layout', 'ocr', 'lm'):
                sys.exit(f'models.tsv: {name}: unknown kind {m["kind"]!r}')
            if m['kind'] != 'lm':
                continue
            if m.get('lm_style') not in LM_STYLES or not m.get('lm_lists') or not m.get('decodes'):
                sys.exit(f'models.tsv: LM {name} needs decodes, lm_style ({" or ".join(LM_STYLES)}) and lm_lists')
            for ocr in m['decodes'].split(','):
                ocr = ocr.strip()
                if ocr not in self.ocr_runs:
                    sys.exit(f'models.tsv: LM {name} decodes {ocr!r}, which is not an OCR model of the registry (or '
                             f'a setup in its croppers)')
                self.pairings.append(dict(lm=name, ocr=ocr, style=m['lm_style'], lists=m['lm_lists']))
        self._base = None

    def folder(self, name):
        """The model folder (of an OCR run <name>@<setup>: of the model)."""
        return os.path.join(self.root, self.models[name.split('@')[0]]['folder'])

    def ocr_config(self, name):
        """The config of an OCR run: the model's, for <name>@<setup> with that [LINE_CROPPER]."""
        config = read_config(self.folder(name))
        if '@' in name:
            config['LINE_CROPPER'] = cropper_section(name.split('@', 1)[1])
        return config

    def of_kind(self, kind):
        return sorted(n for n, m in self.models.items() if m['kind'] == kind)

    def weights(self, name):
        """{file: md5} of the files of a model folder (configs included)."""
        folder = self.folder(name)
        return {f: md5_file(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
                if os.path.isfile(os.path.join(folder, f))}

    def threshold(self, name):
        return self.ocr_config(name)['PAGE_PARSER'].getfloat(THRESHOLD, fallback=0.0)

    def base(self, name):
        """The OCR model whose run gives this model's output: itself, or for a model that differs from another only in
        FILTER_CONFIDENT_LINES_THRESHOLD (same files, same config otherwise) the one with the lowest threshold; the
        output of a higher threshold is that run with more lines removed."""
        if self._base is None:
            groups = {}
            for n in self.ocr_runs:
                config = self.ocr_config(n)
                config['PAGE_PARSER'].pop(THRESHOLD, None)
                sections = {k: dict(v) for k, v in config.items() if k != 'DEFAULT'}
                files = {f: h for f, h in self.weights(n).items() if f != 'config.ini'}
                groups.setdefault(json.dumps([sections, files], sort_keys=True), []).append(n)
            self._base = {}
            for members in groups.values():
                first = min(members, key=lambda n: (self.threshold(n), n))
                self._base.update({n: first for n in members})
        return self._base[name]

    def paired_ocr(self):
        """The OCR models that run for an LM: the bases of the decoded models."""
        return {self.base(p['ocr']) for p in self.pairings}


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
    setups = sorted({cropper_setup(models.ocr_config(n)) for n in models.ocr_runs})
    units += [Unit('crops', s, out) for s in setups]
    units += [Unit('ocr', n, out) for n in models.ocr_runs]
    units += [Unit('lm', f"{p['lm']}+{p['ocr']}", out) for p in models.pairings]
    return units


# ----------------------------------------------------------------------------------------------------------------------
# Page reading (a thread reads the next pages while the GPU works)


def read_pages(sample, pages, with_layout=True, crops=None, depth=3):
    """(page, image, layout) of every page; with crops (a crops LMDB) the layout's lines carry their stored crops."""
    from pero_ocr.core.layout import PageLayout
    q = queue.Queue(maxsize=depth)

    def worker():
        try:
            txn = crops.begin() if crops is not None else None
            for page in pages:
                image = cv2.imread(sample.image(page), 1)
                layout = PageLayout(file=sample.page_xml(page)) if with_layout else None
                if txn is not None:
                    for line in layout.lines_iterator():
                        png = txn.get(f"{page['sample_id']}/{line.id}".encode())
                        if png is None:
                            raise KeyError(f"no stored crop of line {page['sample_id']}/{line.id}")
                        line.crop = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
                q.put((page, image, layout))
            q.put(None)
        except Exception as e:
            q.put(e)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
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
    for n in models.ocr_runs:
        config = models.ocr_config(n)
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
        self.f = path if isinstance(path, io.StringIO) else open(path, 'w', encoding='utf-8', newline='')
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
                conf = '' if line.transcription_confidence is None else repr(float(line.transcription_confidence))
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


def crops_env(out, config):
    """The crops LMDB of the model's [LINE_CROPPER] setup (the crops unit must be done)."""
    import lmdb
    setup = cropper_setup(config)
    folder = os.path.join(out, 'crops', setup)
    if not os.path.exists(os.path.join(folder, 'unit.json')):
        sys.exit(f'crops/{setup} is not done: the OCR models read their line crops from it')
    return lmdb.open(os.path.join(folder, 'crops.lmdb'), subdir=True, readonly=True, lock=False), setup


def ocr_parser(models, name):
    """PageParser of an OCR model without its line cropper: the crops come from the crops unit."""
    from pero_ocr.document_ocr.page_parser import PageParser
    folder = models.folder(name)
    config = models.ocr_config(name)
    run = copy.deepcopy(config)
    run['PAGE_PARSER']['RUN_LINE_CROPPER'] = 'no'
    return PageParser(run, config_path=folder), config


def run_ocr(unit, models, sample, out):
    """One OCR model on every page, on the stored line crops of its setup."""
    from pero_ocr.document_ocr.page_parser import prepare_dense_logits
    parser, config = ocr_parser(models, unit.name)
    env, setup = crops_env(out, config)
    keep = unit.name in models.paired_ocr()
    unit.start()
    if keep:
        os.makedirs(os.path.join(unit.partial, 'page_xml'))
        os.makedirs(os.path.join(unit.partial, 'logits'))
    writer = LineWriter(os.path.join(unit.partial, 'lines.tsv'))
    logits = {}
    for page, image, layout in read_pages(sample, sample.pages, crops=env):
        sid = page['sample_id']
        np.random.seed(page_seed(sid))
        line_ids = [line.id for line in layout.lines_iterator()]
        for line in layout.lines_iterator():
            line.transcription = None
        layout = parser.process_page(image, layout)
        writer.page(sid, line_ids, layout)
        for line in layout.lines_iterator():
            if line.logits is not None and in_logits_subset(f'{sid}/{line.id}'):
                logits[f'{sid}/{line.id}'] = prepare_dense_logits(line).astype(np.float32)
        if keep:
            layout.to_pagexml(os.path.join(unit.partial, 'page_xml', sid + '.xml'))
            with open(os.path.join(unit.partial, 'logits', sid + '.logits'), 'wb') as f:
                f.write(layout.save_logits_bytes(missing_line_logits_ok=True))
    del parser
    env.close()
    writer.close()
    np.savez(os.path.join(unit.partial, 'logprobs.npz'), **logits)
    unit.finish(dict(model=unit.name, weights=models.weights(unit.name), config=config_text(config),
                     cropper_setup=setup, pages=len(sample.pages), lines=writer.n, logprobs_lines=len(logits),
                     logprobs_subset=f'1/{LOGITS_SUBSET} by md5 of "<sample_id>/<line_id>"', page_output=keep))


def read_lines(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'))


def run_derived(unit, models, sample, out):
    """An OCR model that differs from its base only in a higher FILTER_CONFIDENT_LINES_THRESHOLD: the base's lines,
    with the lines the filter removes (confidence <= threshold, as PageParser filters) marked kept=0."""
    base = models.base(unit.name)
    base_folder = os.path.join(out, 'ocr', base)
    if not os.path.exists(os.path.join(base_folder, 'unit.json')):
        sys.exit(f'ocr/{base} is not done: ocr/{unit.name} is derived from it')
    threshold = models.threshold(unit.name)
    unit.start()
    rows = read_lines(os.path.join(base_folder, 'lines.tsv'))
    removed = set()
    with open(os.path.join(unit.partial, 'lines.tsv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        w.writerow(rows[0])
        for sid, line_id, text, conf, kept in rows[1:]:
            if kept == '1' and conf != '' and float(conf) <= threshold:
                sid, line_id, text, conf, kept = sid, line_id, '', '', '0'
                removed.add(f'{sid}/{line_id}')
            w.writerow([sid, line_id, text, conf, kept])
    with np.load(os.path.join(base_folder, 'logprobs.npz')) as base_logits:
        logits = {k: base_logits[k] for k in base_logits.files if k not in removed}
    np.savez(os.path.join(unit.partial, 'logprobs.npz'), **logits)
    with open(os.path.join(base_folder, 'unit.json')) as f:
        base_info = json.load(f)
    config = models.ocr_config(unit.name)
    unit.finish(dict(model=unit.name, weights=models.weights(unit.name), config=config_text(config),
                     derived_from=base, base_finished_utc=base_info['finished_utc'], threshold=threshold,
                     pages=len(sample.pages), lines=len(rows) - 1, removed_lines=len(removed),
                     logprobs_lines=len(logits)))


def ocr_output(ocr_name, pages, models, sample, out):
    """The folder with the stored output (page_xml/, logits/) of the run of an OCR model (its base) for these pages.
    Pages whose output is not stored (pruned, or the model had no LM when it ran) are run again; their lines must
    equal the unit's lines.tsv (same code, same seeds), otherwise the records do not belong together."""
    folder = os.path.join(out, 'ocr', ocr_name)
    if not os.path.exists(os.path.join(folder, 'unit.json')):
        sys.exit(f'ocr/{ocr_name} is not done: the LMs that decode it need its output')
    missing = [p for p in pages if not os.path.exists(os.path.join(folder, 'logits', p['sample_id'] + '.logits'))]
    if not missing:
        return folder
    logger.info(f'ocr/{ocr_name}: output of {len(missing)} pages is not stored, running the OCR model again')
    recorded = {}
    for row in read_lines(os.path.join(folder, 'lines.tsv'))[1:]:
        recorded.setdefault(row[0], []).append(row)
    parser, config = ocr_parser(models, ocr_name)
    env, _ = crops_env(out, config)
    for sub in ('page_xml', 'logits'):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    for page, image, layout in read_pages(sample, missing, crops=env):
        sid = page['sample_id']
        np.random.seed(page_seed(sid))
        line_ids = [line.id for line in layout.lines_iterator()]
        for line in layout.lines_iterator():
            line.transcription = None
        layout = parser.process_page(image, layout)
        check = io.StringIO()
        LineWriter(check).page(sid, line_ids, layout)
        rows = list(csv.reader(io.StringIO(check.getvalue()), delimiter='\t', quoting=csv.QUOTE_NONE,
                               escapechar='\\'))[1:]
        if rows != recorded.get(sid, []):
            sys.exit(f'ocr/{ocr_name}: page {sid} run again differs from its lines.tsv; the OCR unit was recorded '
                     f'differently (another environment?): delete ocr/{ocr_name} and run it again')
        for sub, name, data in (('page_xml', sid + '.xml', None),
                                ('logits', sid + '.logits', layout.save_logits_bytes(missing_line_logits_ok=True))):
            tmp = os.path.join(folder, sub, f'.{name}.{os.getpid()}')
            if data is None:
                layout.to_pagexml(tmp)
            else:
                with open(tmp, 'wb') as f:
                    f.write(data)
            os.rename(tmp, os.path.join(folder, sub, name))
    del parser
    env.close()
    return folder


def run_lm(unit, models, sample, out):
    """The LM decoder on the stored output of the OCR model it decodes (PAGE XML and logits, as the worker stores
    them); for a model derived by a higher threshold, without the lines that threshold removes."""
    from pero_ocr.core.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import PageParser
    pairing = next(p for p in models.pairings if f"{p['lm']}+{p['ocr']}" == unit.name)
    ocr_unit = os.path.join(out, 'ocr', pairing['ocr'])
    if not os.path.exists(os.path.join(ocr_unit, 'unit.json')):
        sys.exit(f'ocr/{pairing["ocr"]} is not done: {unit.id} decodes it')
    scope = sample.in_scope(pairing['lists'])
    pages = [p for p in sample.pages if p['sample_id'] in scope]
    base = models.base(pairing['ocr'])
    ocr_folder = ocr_output(base, pages, models, sample, out)
    removed = set()
    if base != pairing['ocr']:
        removed = {(r[0], r[1]) for r in read_lines(os.path.join(ocr_unit, 'lines.tsv'))[1:] if r[4] == '0'}
    config, lm_folder = decoder_config(models, pairing, models.folder(pairing['ocr']))
    decoder = PageParser(config, config_path=lm_folder)
    unit.start()
    writer = LineWriter(os.path.join(unit.partial, 'lines.tsv'))
    for page in pages:
        sid = page['sample_id']
        np.random.seed(page_seed(sid))
        line_ids = [line.id for line in PageLayout(file=sample.page_xml(page)).lines_iterator()]
        layout = PageLayout(file=os.path.join(ocr_folder, 'page_xml', sid + '.xml'))
        if removed:
            for region in layout.regions:
                region.lines = [line for line in region.lines if (sid, line.id) not in removed]
        layout.load_logits(os.path.join(ocr_folder, 'logits', sid + '.logits'))
        layout = decoder.process_page(None, layout)
        writer.page(sid, line_ids, layout)
    writer.close()
    with open(os.path.join(ocr_unit, 'unit.json')) as f:
        ocr_info = json.load(f)
    unit.finish(dict(lm=pairing['lm'], ocr=pairing['ocr'], style=pairing['style'], lists=pairing['lists'],
                     lm_weights=models.weights(pairing['lm']), ocr_weights=models.weights(pairing['ocr']),
                     ocr_unit_finished_utc=ocr_info['finished_utc'], decoder_config=config_text(config),
                     pages=len(pages), lines=writer.n))


def prune_logits(models, out):
    """Delete ocr/<model>/logits/ of the OCR runs whose LMs are all done (an LM added later runs the OCR model again
    on its pages)."""
    for base in sorted(models.paired_ocr()):
        folder = os.path.join(out, 'ocr', base, 'logits')
        pairs = [f"{p['lm']}+{p['ocr']}" for p in models.pairings if models.base(p['ocr']) == base]
        if os.path.isdir(folder) and all(os.path.exists(os.path.join(out, 'lm', n, 'unit.json')) for n in pairs):
            shutil.rmtree(folder)
            logger.info(f'pruned the logits of ocr/{base}')


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


def needs(unit, models):
    """The unit ids a unit waits for."""
    if unit.kind == 'ocr':
        base = models.base(unit.name)
        if base != unit.name:
            return [f'ocr/{base}']
        return [f'crops/{cropper_setup(models.ocr_config(unit.name))}']
    if unit.kind == 'lm':
        return [f"ocr/{unit.name.split('+', 1)[1]}"]
    return []


def groups(todo, models, sample):
    """Units that run in a process of their own, most expensive first; a unit counts with the units that wait for it."""
    n = len(sample.pages)
    ids = {p['sample_id'] for p in sample.pages}
    own = {}
    for u in todo:
        if u.kind == 'layout':
            own[u.id] = 0.5 * n
        elif u.kind == 'ocr':
            own[u.id] = 0.01 * n if models.base(u.name) != u.name else (1.6 if 'transformer' in u.name else 0.6) * n
        elif u.kind == 'lm':
            pairing = next(p for p in models.pairings if f"{p['lm']}+{p['ocr']}" == u.name)
            own[u.id] = 5.0 * len(sample.in_scope(pairing['lists']) & ids)
    waiting = {}
    for u in todo:
        for dep in needs(u, models):
            waiting.setdefault(dep, []).append(u.id)

    def cost(uid):
        return own.get(uid, 0) + max((cost(w) for w in waiting.get(uid, [])), default=0)
    return [[u] for u in sorted((u for u in todo if u.kind != 'crops'), key=lambda u: -cost(u.id))]


def run_inline(todo, models, sample, out, folder_name, crop_workers):
    crops = [u for u in todo if u.kind == 'crops']
    crop_thread = None
    if crops:
        crop_thread = threading.Thread(target=run_crops, args=(crops, models, sample, crop_workers))
        crop_thread.start()
    for unit in (u for u in todo if u.kind == 'layout'):
        unit.start()
        unit.finish(run_layout(unit, models, sample))
    if crop_thread:
        crop_thread.join()
    ocr = [u for u in todo if u.kind == 'ocr']
    for unit in sorted(ocr, key=lambda u: models.base(u.name) != u.name):
        (run_ocr if models.base(unit.name) == unit.name else run_derived)(unit, models, sample, out)
    for unit in (u for u in todo if u.kind == 'lm'):
        run_lm(unit, models, sample, out)


def run_parallel(todo, models, sample, out, args, folder_name):
    """Crops in this process (a CPU pool), every other unit in a process of its own, args.jobs at a time, at low CPU
    priority; a unit starts when the units it needs are done and enough memory is free."""
    crops = [u for u in todo if u.kind == 'crops']
    lock, failed, finished = threading.Lock(), [], set()

    def crop_job():
        try:
            run_crops(crops, models, sample, args.crop_workers)
        except Exception:
            logger.exception('crops failed')
            with lock:
                failed.extend(u.id for u in crops)
        with lock:
            finished.update(u.id for u in crops)

    crop_thread = None
    if crops:
        crop_thread = threading.Thread(target=crop_job)
        crop_thread.start()
    pending = groups(todo, models, sample)
    todo_ids = {u.id for u in todo}
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    base = [sys.executable, os.path.abspath(__file__), '--exact', '--data', args.data, '--models', args.models,
            '--results', args.results, '--crop-workers', '0'] + (['--pages', str(args.pages)] if args.pages else []) + \
           (['--allow-dirty'] if args.allow_dirty else []) + (['--force-environment'] if args.force_environment else [])

    def done(uid):
        return os.path.exists(os.path.join(out, uid, 'unit.json'))

    def next_group():
        """The first pending group that can start, 'wait', or None when nothing is left."""
        for i, group in enumerate(pending):
            deps = needs(group[0], models)
            if all(done(d) for d in deps):
                return pending.pop(i)
            lost = [d for d in deps if not done(d) and (d in failed or d not in todo_ids or d in finished)]
            if lost:
                pending.pop(i)
                failed.append(group[0].id)
                finished.add(group[0].id)
                logger.error(f'{group[0].id} not run: {", ".join(lost)} failed')
                return 'wait'
        return 'wait' if pending else None

    def worker():
        while True:
            with lock:
                group = next_group()
                if group is None:
                    return
                if group != 'wait':
                    while mem_available_gb() < args.min_free_gb:
                        time.sleep(30)
                    name = '_'.join(u.id.replace('/', '__') for u in group)
                    logger.info(f'start {" ".join(u.id for u in group)}')
                    log = open(os.path.join(out, 'logs', name + '.log'), 'w')
                    proc = subprocess.Popen(base + ['--units'] + [u.id for u in group], stdout=log,
                                            stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(10))
                    time.sleep(args.stagger)  # let it load its models before the memory check of the next one
            if group == 'wait':
                time.sleep(10)
                continue
            code = proc.wait()
            log.close()
            with lock:
                finished.update(u.id for u in group)
                if code != 0:
                    failed.append(group[0].id)
                    logger.error(f'{name} failed, see logs/{name}.log')
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
    p.add_argument('--prune-logits', action='store_true',
                   help='delete the stored logits of the OCR units whose LM pairs are all done, run nothing')
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
    if args.prune_logits:
        prune_logits(models, out)
        return
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
