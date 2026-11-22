#!/usr/bin/env python3
"""Run the engines of each dataset's registry on the dataset's pages and store what they produce, under the pero-ocr
commit.

    python benchmark/infer.py --jobs 6               # every dataset: everything not in the records yet
    python benchmark/infer.py --list                 # the units and which are done
    python benchmark/infer.py --datasets arabic      # only some datasets (rarely needed)
    python benchmark/infer.py --units OCR/           # only some units (rarely needed)
    python benchmark/infer.py --prune-logits         # delete the stored OCR logits no decoder still needs

Inference is not a test: it only records what the code at this commit produces. benchmark/score.py turns the records
into the benchmark, and test/inference/ compares the records of two commits.

Inputs (environment variables or options):
    PERO_TEST_DATA     the data folder: one folder per dataset, <dataset>/registry.yaml (which engines run on it),
                       <dataset>/images/<page>.<ext>, <dataset>/page_xml/<page>.xml (the input layout; the croppers
                       crop its lines); a page is named by its file name. Other files (test lists) are not read.
    PERO_TEST_MODELS   the engines: <STAGE>/<name>/ with a config.ini holding the stage's section and the files it
                       refers to (STAGE: LAYOUT_PARSER, LINE_CROPPER, OCR, DECODER)
    PERO_INFERENCE     the records: <PERO_INFERENCE>/<commit>/inference/<dataset>/

The configs are used as they are; the code only sets the RUN_* switches of [PAGE_PARSER] to the engine's stage. Adding
an engine to a registry and running infer.py again computes only what is missing.

Units (one folder each in <commit>/inference/<dataset>/, complete when it holds unit.json):
    LAYOUT_PARSER/<layout>          the layout engine on every page image: page_xml/<page>.xml
    LINE_CROPPER/<cropper>          the line cropper on the lines of the dataset's PAGE XML: crops.lmdb, key
                                    "<page>/<line_id>", the crop as PNG (lossless)
    OCR/<ocr>@<cropper>             the OCR engine on the stored crops of the cropper: lines.tsv (text, confidence,
                                    kept), logprobs.npz (full log-probabilities of a fixed subset of lines, the same
                                    subset for every engine). An OCR run a decoder decodes also keeps its full output
                                    of every page, as the ScribbleSense worker stores it for the LM stage:
                                    page_xml/<page>.xml and logits/<page>.logits (PageLayout.save_logits_bytes);
                                    --prune-logits deletes logits/ once all its decoders are done.
                                    An OCR engine that differs from another only in a higher [PAGE_PARSER]
                                    FILTER_CONFIDENT_LINES_THRESHOLD (same files, same config otherwise, same cropper)
                                    is not run: its unit is the other's lines without the lines of confidence <= the
                                    threshold, as PageParser removes them.
    DECODER/<lm>@<ocr>@<cropper>    the decoder on the stored output of the OCR run (where that output is not stored,
                                    the OCR engine runs again and must give the recorded lines): lines.tsv

Every line of every page is inferred. Every page is processed by a fresh numpy random state seeded from its name, so
a rerun gives the same result (the layout engine breaks ties with np.random, issue #100).
"""
import argparse
import configparser
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
STAGES = ('LAYOUT_PARSER', 'LINE_CROPPER', 'OCR', 'DECODER')
INPUT = {'OCR': 'LINE_CROPPER', 'DECODER': 'OCR'}  # the stage whose results an engine of this stage runs on
RUN_SWITCH = {'LAYOUT_PARSER': 'RUN_LAYOUT_PARSER', 'LINE_CROPPER': 'RUN_LINE_CROPPER', 'OCR': 'RUN_OCR',
              'DECODER': 'RUN_DECODER'}
THRESHOLD = 'FILTER_CONFIDENT_LINES_THRESHOLD'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp', '.jp2')
LOGITS_SUBSET = 64  # one line in LOGITS_SUBSET keeps its logits (chosen by a hash of its key, the same for every model)
LMDB_MAP_SIZE = 1 << 40


# ----------------------------------------------------------------------------------------------------------------------
# Inputs


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def page_seed(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def in_logits_subset(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % LOGITS_SUBSET == 0


class Dataset:
    """<folder>/images/<page>.<ext> and <folder>/page_xml/<page>.xml; a page is {'name', 'image', 'page_xml'}."""

    def __init__(self, folder):
        self.folder = folder
        self.name = os.path.basename(os.path.normpath(folder))
        images, xmls = os.path.join(folder, 'images'), os.path.join(folder, 'page_xml')
        if not os.path.isdir(images) or not os.path.isdir(xmls):
            sys.exit(f'{folder}: a dataset needs images/ and page_xml/')
        by_name = {}
        for f in sorted(os.listdir(images)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            if stem in by_name:
                sys.exit(f'{folder}: two images of page {stem}')
            by_name[stem] = os.path.join(images, f)
        xml_names = {os.path.splitext(f)[0] for f in os.listdir(xmls) if f.endswith('.xml')}
        if set(by_name) != xml_names:
            only_images, only_xml = sorted(set(by_name) - xml_names), sorted(xml_names - set(by_name))
            sys.exit(f'{folder}: images and page_xml do not match; without PAGE XML: {only_images[:5]}, without an '
                     f'image: {only_xml[:5]}')
        self.all_pages = [dict(name=n, image=p, page_xml=os.path.join(xmls, n + '.xml')) for n, p in by_name.items()]
        self.pages = self.all_pages  # the pages to process (a trial takes the first few)

    def fingerprint(self):
        return {os.path.relpath(f, self.folder): md5_file(f)
                for p in self.all_pages for f in (p['image'], p['page_xml'])}


def config_text(config):
    out = io.StringIO()
    config.write(out)
    return out.getvalue()


# ----------------------------------------------------------------------------------------------------------------------
# Registry


def names(value, where):
    """A registry value: one name or a list of names."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return value
    sys.exit(f'registry: {where}: expected a name or a list of names, got {value!r}')


class Engine:
    def __init__(self, stage, name, folder, inputs):
        self.stage, self.name, self.folder = stage, name, folder
        self.inputs = inputs  # names of engines of the INPUT stage (OCR: croppers; DECODER: OCR runs)

    def config(self):
        """The engine's config.ini as it is, with the RUN_* switches of [PAGE_PARSER] set to its stage."""
        config = configparser.ConfigParser()
        config.read(os.path.join(self.folder, 'config.ini'))
        if not config.has_section('PAGE_PARSER'):
            config.add_section('PAGE_PARSER')
        for stage, switch in RUN_SWITCH.items():
            config['PAGE_PARSER'][switch] = 'yes' if stage == self.stage else 'no'
        return config

    def files(self):
        """{file: md5} of the engine folder (config included)."""
        return {f: md5_file(os.path.join(self.folder, f)) for f in sorted(os.listdir(self.folder))
                if os.path.isfile(os.path.join(self.folder, f))}

    def ocr_json(self):
        with open(os.path.join(self.folder, self.config()['OCR']['OCR_JSON'])) as f:
            return json.load(f)


class Registry:
    """<dataset>/registry.yaml: for every stage (LAYOUT_PARSER, LINE_CROPPER, OCR, DECODER) its engines, by name; an
    engine's folder is <STAGE>/<name>/ in the engines folder. An engine is a list item, or a key with what it runs on:
        LINE_CROPPER: <names>    OCR engines: the croppers whose line crops it reads (required)
        OCR: <names>             DECODER engines: the OCR runs whose output it decodes (required); <ocr>, or
                                 <ocr>@<cropper> for an OCR engine listed with several croppers
    <names> is one name or a list. Any subset of the stages is valid."""

    def __init__(self, path, engines_root):
        self.path, self.engines_root = path, engines_root
        try:
            import yaml
        except ImportError:
            sys.exit('reading the registry needs PyYAML (pip install pyyaml)')
        if not os.path.isfile(path):
            sys.exit(f'no registry {path}')
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            sys.exit(f'{path}: expected the stages as keys')
        unknown = set(data) - set(STAGES)
        if unknown:
            sys.exit(f'{path}: unknown stages {sorted(unknown)}; the stages are {", ".join(STAGES)}')
        self.engines = {stage: {} for stage in STAGES}
        for stage in STAGES:
            entries = data.get(stage)
            if entries is None:
                continue
            if isinstance(entries, (str, list)):
                entries = {e: None for e in names(entries, stage)}
            if not isinstance(entries, dict):
                sys.exit(f'{path}: {stage}: expected a list of engines or engines with what they run on')
            for name, settings in entries.items():
                self.engines[stage][name] = self._engine(stage, name, settings or {})
        self._check()
        self._base = None

    def _engine(self, stage, name, settings):
        where = f'{stage}: {name}'
        if '@' in name:
            sys.exit(f'registry: {where}: "@" separates the parts of a run name; an engine name must not contain it')
        folder = os.path.join(self.engines_root, stage, name)
        if not os.path.isfile(os.path.join(folder, 'config.ini')):
            sys.exit(f'registry: {where}: no {os.path.join(folder, "config.ini")}')
        allowed = {INPUT[stage]} if stage in INPUT else set()
        if not isinstance(settings, dict) or set(settings) - allowed:
            sys.exit(f'registry: {where}: ' + (f'a {stage} engine takes {INPUT[stage]} (what it runs on)' if allowed
                                                else f'a {stage} engine takes no settings'))
        inputs = []
        if stage in INPUT:
            if INPUT[stage] not in settings:
                sys.exit(f'registry: {where}: {INPUT[stage]} (what it runs on) is required')
            inputs = names(settings[INPUT[stage]], f'{where}: {INPUT[stage]}')
        return Engine(stage, name, folder, inputs)

    def _check(self):
        for e in self.engines['OCR'].values():
            for cropper in e.inputs:
                if cropper not in self.engines['LINE_CROPPER']:
                    sys.exit(f'registry: OCR: {e.name}: LINE_CROPPER {cropper} is not in the registry')
                height = self.engines['LINE_CROPPER'][cropper].config()['LINE_CROPPER'].getint('LINE_HEIGHT')
                if height != e.ocr_json()['line_px_height']:
                    sys.exit(f'registry: OCR: {e.name} takes lines of {e.ocr_json()["line_px_height"]} px, '
                             f'LINE_CROPPER {cropper} gives {height} px')
        runs = {r: self.ocr_engine(r) for r in self.ocr_runs()}
        for e in self.engines['DECODER'].values():
            resolved = []
            for ref in e.inputs:
                if ref in runs:
                    resolved.append(ref)
                    continue
                ocr = self.engines['OCR'].get(ref)
                if ocr is None:
                    sys.exit(f'registry: DECODER: {e.name}: OCR {ref} is not in the registry')
                if len(ocr.inputs) > 1:
                    sys.exit(f'registry: DECODER: {e.name}: OCR {ref} runs with several croppers; name one as '
                             f'{ref}@<cropper>')
                resolved.append(f'{ref}@{ocr.inputs[0]}')
            e.inputs = resolved
            if not e.config().has_option('OCR', 'OCR_JSON'):
                sys.exit(f'registry: DECODER: {e.name}: its config has no [OCR] OCR_JSON (the alphabet it decodes)')
            alphabet = e.ocr_json()['characters']
            for run in e.inputs:
                ocr = runs[run]
                if ocr.config()['OCR'].get('METHOD') == 'pytorch_ocr-transformer':
                    sys.exit(f'registry: DECODER: {e.name}: OCR {ocr.name} is a transformer; a decoder needs CTC '
                             f'logits')
                if ocr.ocr_json()['characters'] != alphabet:
                    sys.exit(f'registry: DECODER: {e.name}: its alphabet is not the alphabet of OCR {ocr.name}')

    def ocr_runs(self):
        """<ocr>@<cropper> of every OCR engine and cropper it is listed with."""
        return [f'{e.name}@{c}' for e in self.engines['OCR'].values() for c in e.inputs]

    def ocr_engine(self, run):
        return self.engines['OCR'][run.split('@', 1)[0]]

    def decoder_runs(self):
        """(decoder, OCR run) of every decoder and OCR run it decodes."""
        return [(e, run) for e in self.engines['DECODER'].values() for run in e.inputs]

    def threshold(self, run):
        return self.ocr_engine(run).config()['PAGE_PARSER'].getfloat(THRESHOLD, fallback=0.0)

    def base(self, run):
        """The OCR run that gives this run's output: itself, or for runs that differ only in FILTER_CONFIDENT_LINES_-
        THRESHOLD (same cropper, same files, same config otherwise) the one with the lowest threshold; the output of
        a higher threshold is that run with more lines removed."""
        if self._base is None:
            groups = {}
            for r in self.ocr_runs():
                engine = self.ocr_engine(r)
                config = engine.config()
                config['PAGE_PARSER'].pop(THRESHOLD, None)
                sections = {k: dict(v) for k, v in config.items() if k != 'DEFAULT'}
                files = {f: h for f, h in engine.files().items() if f != 'config.ini'}
                key = json.dumps([r.split('@', 1)[1], sections, files], sort_keys=True)
                groups.setdefault(key, []).append(r)
            self._base = {}
            for members in groups.values():
                first = min(members, key=lambda r: (self.threshold(r), r))
                self._base.update({r: first for r in members})
        return self._base[run]

    def kept_output(self):
        """The OCR runs whose full page output is kept: the bases of the decoded runs."""
        return {self.base(run) for _, run in self.decoder_runs()}


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


def check_record(path, record, keys, what, force):
    """Write the record, or require that an existing one has the same keys."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(record, f, indent=1, sort_keys=True)
        return
    with open(path) as f:
        old = json.load(f)
    differ = [k for k in keys if old.get(k) != record.get(k)]
    if differ and not force:
        details = []
        for k in differ:
            a, b = old.get(k), record.get(k)
            if isinstance(a, dict) and isinstance(b, dict):
                details += [f'  {k}: {p}: {a.get(p)} -> {b.get(p)}' for p in sorted(set(a) | set(b))
                            if a.get(p) != b.get(p)][:20]
            else:
                details.append(f'  {k}: {a} -> {b}')
        sys.exit(f'{os.path.dirname(path)} was inferred {what}; adding to it would mix them (--force-environment to '
                 f'do it anyway):\n' + '\n'.join(details))


# ----------------------------------------------------------------------------------------------------------------------
# Units


class Unit:
    def __init__(self, stage, name, out):
        self.stage, self.name = stage, name
        self.final = os.path.join(out, stage, name)
        self.partial = self.final + '.partial'

    @property
    def id(self):
        return f'{self.stage}/{self.name}'

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


def plan(registry, out):
    """All units, in run order."""
    units = [Unit('LAYOUT_PARSER', n, out) for n in registry.engines['LAYOUT_PARSER']]
    units += [Unit('LINE_CROPPER', n, out) for n in registry.engines['LINE_CROPPER']]
    units += [Unit('OCR', r, out) for r in registry.ocr_runs()]
    units += [Unit('DECODER', f'{e.name}@{run}', out) for e, run in registry.decoder_runs()]
    return units


def needs(unit, registry):
    """The unit ids a unit waits for."""
    if unit.stage == 'OCR':
        base = registry.base(unit.name)
        return [f'OCR/{base}'] if base != unit.name else [f"LINE_CROPPER/{unit.name.split('@', 1)[1]}"]
    if unit.stage == 'DECODER':
        return [f"OCR/{unit.name.split('@', 1)[1]}"]
    return []


# ----------------------------------------------------------------------------------------------------------------------
# Page reading (a thread reads the next pages while the GPU works)


def read_pages(pages, with_layout=True, crops=None, depth=3):
    """(page, image, layout) of every page; with crops (a crops LMDB) the layout's lines carry their stored crops."""
    from pero_ocr.core.layout import PageLayout
    q = queue.Queue(maxsize=depth)

    def worker():
        try:
            txn = crops.begin() if crops is not None else None
            for page in pages:
                image = cv2.imread(page['image'], 1)
                layout = PageLayout(file=page['page_xml']) if with_layout else None
                if txn is not None:
                    for line in layout.lines_iterator():
                        png = txn.get(f"{page['name']}/{line.id}".encode())
                        if png is None:
                            raise KeyError(f"no stored crop of line {page['name']}/{line.id}")
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


def run_layout(unit, registry, dataset):
    from pero_ocr.core.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import PageParser
    engine = registry.engines['LAYOUT_PARSER'][unit.name]
    config = engine.config()
    parser = PageParser(config, config_path=engine.folder)
    unit.start()
    os.makedirs(os.path.join(unit.partial, 'page_xml'))
    n_lines = 0
    for page, image, _ in read_pages(dataset.pages, with_layout=False):
        np.random.seed(page_seed(page['name']))
        layout = PageLayout(id=page['name'], page_size=(image.shape[0], image.shape[1]))
        layout = parser.process_page(image, layout)
        layout.to_pagexml(os.path.join(unit.partial, 'page_xml', page['name'] + '.xml'))
        n_lines += sum(1 for _ in layout.lines_iterator())
    del parser
    unit.finish(dict(engine=unit.name, files=engine.files(), config=config_text(config), pages=len(dataset.pages),
                     lines=n_lines))


# ----------------------------------------------------------------------------------------------------------------------
# Line crops (CPU, a process pool; runs while the GPU units run)


_croppers = {}


def _crop_page(args):
    page, configs = args
    from pero_ocr.core.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import PageParser
    image = cv2.imread(page['image'], 1)
    result = {}
    for name, (text, folder) in configs.items():
        if name not in _croppers:
            config = configparser.ConfigParser()
            config.read_string(text)
            _croppers[name] = PageParser(config, config_path=folder)
        np.random.seed(page_seed(page['name']))
        layout = _croppers[name].process_page(image, PageLayout(file=page['page_xml']))
        result[name] = [(f"{page['name']}/{line.id}", cv2.imencode('.png', line.crop)[1].tobytes())
                        for line in layout.lines_iterator() if line.crop is not None]
    return result


def run_crops(units, registry, dataset, workers):
    import lmdb
    engines = {u.name: registry.engines['LINE_CROPPER'][u.name] for u in units}
    configs = {n: (config_text(e.config()), e.folder) for n, e in engines.items()}
    envs = {}
    for u in units:
        u.start()
        envs[u.name] = lmdb.open(os.path.join(u.partial, 'crops.lmdb'), map_size=LMDB_MAP_SIZE, subdir=True)
    counts = {u.name: 0 for u in units}
    jobs = [(page, configs) for page in dataset.pages]
    with multiprocessing.get_context('spawn').Pool(workers, initializer=os.nice, initargs=(10,)) as pool:
        for result in pool.imap(_crop_page, jobs, chunksize=2):
            for name, items in result.items():
                with envs[name].begin(write=True) as txn:
                    for key, png in items:
                        txn.put(key.encode(), png)
                counts[name] += len(items)
    for u in units:
        envs[u.name].close()
        u.finish(dict(engine=u.name, files=engines[u.name].files(), config=configs[u.name][0],
                      pages=len(dataset.pages), lines=counts[u.name], format='png'))


# ----------------------------------------------------------------------------------------------------------------------
# OCR and decoders


class LineWriter:
    def __init__(self, path):
        self.f = path if isinstance(path, io.StringIO) else open(path, 'w', encoding='utf-8', newline='')
        self.w = csv.writer(self.f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        self.w.writerow(['page', 'line_id', 'text', 'confidence', 'kept'])
        self.n = 0

    def page(self, page, line_ids, layout):
        """Every line of the input; kept=0 for lines the PAGE_PARSER filter removed (no text)."""
        lines = {line.id: line for line in layout.lines_iterator()}
        for line_id in line_ids:
            line = lines.get(line_id)
            if line is None:
                self.w.writerow([page, line_id, '', '', 0])
            else:
                text = (line.transcription or '').replace('\t', ' ').replace('\n', ' ')
                conf = '' if line.transcription_confidence is None else repr(float(line.transcription_confidence))
                self.w.writerow([page, line_id, text, conf, 1])
            self.n += 1

    def close(self):
        self.f.close()


def read_lines(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'))


def crops_env(out, cropper):
    import lmdb
    folder = os.path.join(out, 'LINE_CROPPER', cropper)
    if not os.path.exists(os.path.join(folder, 'unit.json')):
        sys.exit(f'LINE_CROPPER/{cropper} is not done: the OCR runs on it read their line crops from it')
    return lmdb.open(os.path.join(folder, 'crops.lmdb'), subdir=True, readonly=True, lock=False)


def ocr_parser(registry, run):
    from pero_ocr.document_ocr.page_parser import PageParser
    engine = registry.ocr_engine(run)
    config = engine.config()
    return PageParser(config, config_path=engine.folder), config


def run_ocr(unit, registry, dataset, out):
    """The OCR engine on the stored line crops of its cropper."""
    from pero_ocr.document_ocr.page_parser import prepare_dense_logits
    parser, config = ocr_parser(registry, unit.name)
    env = crops_env(out, unit.name.split('@', 1)[1])
    keep = unit.name in registry.kept_output()
    unit.start()
    if keep:
        os.makedirs(os.path.join(unit.partial, 'page_xml'))
        os.makedirs(os.path.join(unit.partial, 'logits'))
    writer = LineWriter(os.path.join(unit.partial, 'lines.tsv'))
    logits = {}
    for page, image, layout in read_pages(dataset.pages, crops=env):
        name = page['name']
        np.random.seed(page_seed(name))
        line_ids = [line.id for line in layout.lines_iterator()]
        for line in layout.lines_iterator():
            line.transcription = None
        layout = parser.process_page(image, layout)
        writer.page(name, line_ids, layout)
        for line in layout.lines_iterator():
            if line.logits is not None and in_logits_subset(f'{name}/{line.id}'):
                logits[f'{name}/{line.id}'] = prepare_dense_logits(line).astype(np.float32)
        if keep:
            layout.to_pagexml(os.path.join(unit.partial, 'page_xml', name + '.xml'))
            with open(os.path.join(unit.partial, 'logits', name + '.logits'), 'wb') as f:
                f.write(layout.save_logits_bytes(missing_line_logits_ok=True))
    del parser
    env.close()
    writer.close()
    np.savez(os.path.join(unit.partial, 'logprobs.npz'), **logits)
    engine = registry.ocr_engine(unit.name)
    unit.finish(dict(engine=engine.name, files=engine.files(), config=config_text(config), pages=len(dataset.pages),
                     lines=writer.n, logprobs_lines=len(logits),
                     logprobs_subset=f'1/{LOGITS_SUBSET} by md5 of "<page>/<line_id>"', page_output=keep))


def run_derived(unit, registry, dataset, out):
    """An OCR run that differs from its base only in a higher FILTER_CONFIDENT_LINES_THRESHOLD: the base's lines,
    with the lines the filter removes (confidence <= threshold, as PageParser filters) marked kept=0."""
    base = registry.base(unit.name)
    base_folder = os.path.join(out, 'OCR', base)
    if not os.path.exists(os.path.join(base_folder, 'unit.json')):
        sys.exit(f'OCR/{base} is not done: OCR/{unit.name} is derived from it')
    threshold = registry.threshold(unit.name)
    pages = {p['name'] for p in dataset.pages}
    unit.start()
    rows = read_lines(os.path.join(base_folder, 'lines.tsv'))
    removed, n = set(), 0
    with open(os.path.join(unit.partial, 'lines.tsv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        w.writerow(rows[0])
        for name, line_id, text, conf, kept in rows[1:]:
            if name not in pages:
                continue
            if kept == '1' and conf != '' and float(conf) <= threshold:
                text, conf, kept = '', '', '0'
                removed.add(f'{name}/{line_id}')
            w.writerow([name, line_id, text, conf, kept])
            n += 1
    with np.load(os.path.join(base_folder, 'logprobs.npz')) as base_logits:
        logits = {k: base_logits[k] for k in base_logits.files if k not in removed and k.split('/', 1)[0] in pages}
    np.savez(os.path.join(unit.partial, 'logprobs.npz'), **logits)
    with open(os.path.join(base_folder, 'unit.json')) as f:
        base_info = json.load(f)
    engine = registry.ocr_engine(unit.name)
    unit.finish(dict(engine=engine.name, files=engine.files(), config=config_text(engine.config()),
                     derived_from=base, base_finished_utc=base_info['finished_utc'], threshold=threshold,
                     pages=len(pages), lines=n, removed_lines=len(removed), logprobs_lines=len(logits)))


def ocr_output(run, registry, dataset, out):
    """The folder with the stored output (page_xml/, logits/) of an OCR run. Pages whose output is not stored
    (pruned, or no decoder decoded the run when it ran) are run again; their lines must equal the unit's lines.tsv
    (same code, same seeds), otherwise the records do not belong together."""
    folder = os.path.join(out, 'OCR', run)
    if not os.path.exists(os.path.join(folder, 'unit.json')):
        sys.exit(f'OCR/{run} is not done: the decoders of it need its output')
    missing = [p for p in dataset.pages if not os.path.exists(os.path.join(folder, 'logits', p['name'] + '.logits'))]
    if not missing:
        return folder
    logger.info(f'OCR/{run}: output of {len(missing)} pages is not stored, running the OCR engine again')
    recorded = {}
    for row in read_lines(os.path.join(folder, 'lines.tsv'))[1:]:
        recorded.setdefault(row[0], []).append(row)
    parser, _ = ocr_parser(registry, run)
    env = crops_env(out, run.split('@', 1)[1])
    for sub in ('page_xml', 'logits'):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    for page, image, layout in read_pages(missing, crops=env):
        name = page['name']
        np.random.seed(page_seed(name))
        line_ids = [line.id for line in layout.lines_iterator()]
        for line in layout.lines_iterator():
            line.transcription = None
        layout = parser.process_page(image, layout)
        check = io.StringIO()
        LineWriter(check).page(name, line_ids, layout)
        rows = list(csv.reader(io.StringIO(check.getvalue()), delimiter='\t', quoting=csv.QUOTE_NONE,
                               escapechar='\\'))[1:]
        if rows != recorded.get(name, []):
            sys.exit(f'OCR/{run}: page {name} run again differs from its lines.tsv; the OCR unit was recorded '
                     f'differently (another environment?): delete OCR/{run} and run it again')
        for sub, file, data in (('page_xml', name + '.xml', None),
                                ('logits', name + '.logits', layout.save_logits_bytes(missing_line_logits_ok=True))):
            tmp = os.path.join(folder, sub, f'.{file}.{os.getpid()}')
            if data is None:
                layout.to_pagexml(tmp)
            else:
                with open(tmp, 'wb') as f:
                    f.write(data)
            os.rename(tmp, os.path.join(folder, sub, file))
    del parser
    env.close()
    return folder


def run_decoder(unit, registry, dataset, out):
    """The decoder on the stored output of the OCR run (PAGE XML and logits, as the ScribbleSense worker stores them
    for the LM stage); for a run derived by a higher threshold, without the lines that threshold removes."""
    from pero_ocr.core.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import PageParser
    lm, run = unit.name.split('@', 1)
    engine = registry.engines['DECODER'][lm]
    ocr_unit = os.path.join(out, 'OCR', run)
    if not os.path.exists(os.path.join(ocr_unit, 'unit.json')):
        sys.exit(f'OCR/{run} is not done: {unit.id} decodes it')
    base = registry.base(run)
    ocr_folder = ocr_output(base, registry, dataset, out)
    removed = set()
    if base != run:
        removed = {(r[0], r[1]) for r in read_lines(os.path.join(ocr_unit, 'lines.tsv'))[1:] if r[4] == '0'}
    config = engine.config()
    decoder = PageParser(config, config_path=engine.folder)
    unit.start()
    writer = LineWriter(os.path.join(unit.partial, 'lines.tsv'))
    for page in dataset.pages:
        name = page['name']
        np.random.seed(page_seed(name))
        line_ids = [line.id for line in PageLayout(file=page['page_xml']).lines_iterator()]
        layout = PageLayout(file=os.path.join(ocr_folder, 'page_xml', name + '.xml'))
        if removed:
            for region in layout.regions:
                region.lines = [line for line in region.lines if (name, line.id) not in removed]
        layout.load_logits(os.path.join(ocr_folder, 'logits', name + '.logits'))
        layout = decoder.process_page(None, layout)
        writer.page(name, line_ids, layout)
    writer.close()
    with open(os.path.join(ocr_unit, 'unit.json')) as f:
        ocr_info = json.load(f)
    unit.finish(dict(engine=lm, files=engine.files(), config=config_text(config), ocr_run=run,
                     ocr_unit_finished_utc=ocr_info['finished_utc'], pages=len(dataset.pages), lines=writer.n))


def prune_logits(registry, out):
    """Delete OCR/<run>/logits/ of the OCR runs whose decoders are all done (a decoder added later runs the OCR
    engine again)."""
    for base in sorted(registry.kept_output()):
        folder = os.path.join(out, 'OCR', base, 'logits')
        decoders = [f'{e.name}@{run}' for e, run in registry.decoder_runs() if registry.base(run) == base]
        if os.path.isdir(folder) and all(os.path.exists(os.path.join(out, 'DECODER', d, 'unit.json'))
                                         for d in decoders):
            shutil.rmtree(folder)
            logger.info(f'pruned the logits of OCR/{base}')


# ----------------------------------------------------------------------------------------------------------------------
# Report


def write_report(inference, commit):
    """<commit>/inference/report.md: the units of every dataset."""
    rows = []
    for dataset in sorted(os.listdir(inference)):
        for stage in STAGES:
            folder = os.path.join(inference, dataset, stage)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                path = os.path.join(folder, name, 'unit.json')
                if os.path.exists(path):
                    with open(path) as f:
                        info = json.load(f)
                    rows.append((dataset, stage, name, info.get('pages', ''), info.get('lines', ''), info['seconds'],
                                 info['finished_utc']))
    with open(os.path.join(inference, 'environment.json')) as f:
        env = json.load(f)
    lines = [f'# Inference of pero-ocr {commit}', '',
             f"Python {env['python']}, torch {env['packages'].get('torch')} (CUDA {env['torch_cuda']}), "
             f"numpy {env['packages'].get('numpy')}, opencv {env['packages'].get('opencv-python')}, GPU {env['gpu']}.",
             '', '| dataset | stage | unit | pages | lines | seconds | finished (UTC) |', '|---|---|---|---|---|---|---|']
    lines += [f'| {d} | {k} | {n} | {p} | {l} | {s:.0f} | {t} |' for d, k, n, p, l, s, t in rows]
    with open(os.path.join(inference, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------------------------------------------------
# Running


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    return float('inf')


def run_unit(unit, registry, dataset, out):
    if unit.stage == 'LAYOUT_PARSER':
        run_layout(unit, registry, dataset)
    elif unit.stage == 'OCR':
        (run_ocr if registry.base(unit.name) == unit.name else run_derived)(unit, registry, dataset, out)
    elif unit.stage == 'DECODER':
        run_decoder(unit, registry, dataset, out)


def order(todo, registry, dataset):
    """The units that run in processes of their own: layout and OCR first, decoders last (their results come after
    every OCR result); within that, most expensive first, a unit counting with the units that wait for it."""
    n = len(dataset.pages)
    own = {}
    for u in todo:
        if u.stage == 'LAYOUT_PARSER':
            own[u.id] = 0.5 * n
        elif u.stage == 'OCR':
            transformer = registry.ocr_engine(u.name).config()['OCR'].get('METHOD') == 'pytorch_ocr-transformer'
            own[u.id] = 0.01 * n if registry.base(u.name) != u.name else (1.6 if transformer else 0.6) * n
        elif u.stage == 'DECODER':
            own[u.id] = 5.0 * n
    waiting = {}
    for u in todo:
        for dep in needs(u, registry):
            waiting.setdefault(dep, []).append(u.id)

    def cost(uid):
        return own.get(uid, 0) + max((cost(w) for w in waiting.get(uid, [])), default=0)
    return sorted((u for u in todo if u.stage != 'LINE_CROPPER'), key=lambda u: (u.stage == 'DECODER', -cost(u.id)))


def run_inline(todo, registry, dataset, out, crop_workers):
    crops = [u for u in todo if u.stage == 'LINE_CROPPER']
    crop_thread = None
    if crops:
        crop_thread = threading.Thread(target=run_crops, args=(crops, registry, dataset, crop_workers))
        crop_thread.start()
    for unit in (u for u in todo if u.stage == 'LAYOUT_PARSER'):
        run_unit(unit, registry, dataset, out)
    if crop_thread:
        crop_thread.join()
    ocr = [u for u in todo if u.stage == 'OCR']
    for unit in sorted(ocr, key=lambda u: registry.base(u.name) != u.name):
        run_unit(unit, registry, dataset, out)
    for unit in (u for u in todo if u.stage == 'DECODER'):
        run_unit(unit, registry, dataset, out)


def run_parallel(todo, registry, dataset, out, args, inference, folder_name):
    """Crops in this process (a CPU pool), every other unit in a process of its own, args.jobs at a time, at low CPU
    priority; a unit starts when the units it needs are done and enough memory is free."""
    crops = [u for u in todo if u.stage == 'LINE_CROPPER']
    lock, failed, finished = threading.Lock(), [], set()
    running = {'gpu': 0}

    def crop_job():
        try:
            run_crops(crops, registry, dataset, args.crop_workers)
        except Exception:
            logger.exception('line crops failed')
            with lock:
                failed.extend(u.id for u in crops)
        with lock:
            finished.update(u.id for u in crops)

    crop_thread = None
    if crops:
        crop_thread = threading.Thread(target=crop_job)
        crop_thread.start()
    pending = order(todo, registry, dataset)
    todo_ids = {u.id for u in todo}
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    base = [sys.executable, os.path.abspath(__file__), '--exact', '--data', args.data, '--models', args.models,
            '--results', args.results, '--datasets', dataset.name, '--crop-workers', '0', '--decoder-jobs', '1'] + \
           (['--pages', str(args.pages)] if args.pages else []) + \
           (['--allow-dirty'] if args.allow_dirty else []) + (['--force-environment'] if args.force_environment else [])

    def done(uid):
        return os.path.exists(os.path.join(out, uid, 'unit.json'))

    def next_unit():
        """The first pending unit that can start, 'wait', or None when nothing is left."""
        for i, unit in enumerate(pending):
            if unit.stage != 'DECODER' and running['gpu'] >= args.jobs:
                continue
            deps = needs(unit, registry)
            if all(done(d) for d in deps):
                return pending.pop(i)
            lost = [d for d in deps if not done(d) and (d in failed or d not in todo_ids or d in finished)]
            if lost:
                pending.pop(i)
                failed.append(unit.id)
                finished.add(unit.id)
                logger.error(f'{dataset.name}: {unit.id} not run: {", ".join(lost)} failed')
                return 'wait'
        return 'wait' if pending else None

    def worker():
        while True:
            with lock:
                unit = next_unit()
                if unit is None:
                    return
                if unit != 'wait':
                    while mem_available_gb() < args.min_free_gb:
                        time.sleep(30)
                    name = unit.id.replace('/', '__')
                    logger.info(f'start {dataset.name}: {unit.id}')
                    log = open(os.path.join(out, 'logs', name + '.log'), 'w')
                    proc = subprocess.Popen(base + ['--units', unit.id], stdout=log, stderr=subprocess.STDOUT,
                                            preexec_fn=lambda: os.nice(10))
                    if unit.stage != 'DECODER':
                        running['gpu'] += 1
                    time.sleep(args.stagger)  # let it load its models before the memory check of the next one
            if unit == 'wait':
                time.sleep(10)
                continue
            code = proc.wait()
            log.close()
            with lock:
                if unit.stage != 'DECODER':
                    running['gpu'] -= 1
                finished.add(unit.id)
                if code != 0:
                    failed.append(unit.id)
                    logger.error(f'{dataset.name}: {unit.id} failed, see {dataset.name}/logs/{name}.log')
            write_report(inference, folder_name)

    threads = [threading.Thread(target=worker) for _ in range(max(args.jobs, args.decoder_jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if crop_thread:
        crop_thread.join()
    return [f'{dataset.name}: {u}' for u in failed]


def dataset_folders(data, wanted):
    """The datasets of the data folder: its subfolders with a registry.yaml."""
    found = sorted(d for d in os.listdir(data) if os.path.isfile(os.path.join(data, d, 'registry.yaml')))
    if wanted:
        missing = [d for d in wanted if d not in found]
        if missing:
            sys.exit(f'{data}: no dataset {", ".join(missing)} (a dataset is a folder with registry.yaml); '
                     f'found: {", ".join(found)}')
        found = [d for d in found if d in wanted]
    if not found:
        sys.exit(f'{data}: no dataset (a folder with registry.yaml, images/ and page_xml/)')
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=os.environ.get('PERO_TEST_DATA'))
    p.add_argument('--models', default=os.environ.get('PERO_TEST_MODELS'))
    p.add_argument('--results', default=os.environ.get('PERO_INFERENCE'))
    p.add_argument('--datasets', nargs='*', help='only these datasets of the data folder')
    p.add_argument('--units', nargs='*', default=[''], help='unit id prefixes to run, e.g. OCR/ DECODER/czech')
    p.add_argument('--exact', action='store_true', help='--units are exact unit ids; run them in this process (how the unit processes are started)')
    p.add_argument('--pages', type=int, help='only the first N pages of each dataset (a trial; needs --results of '
                                             'its own)')
    p.add_argument('--jobs', type=int, default=1, help='units running at the same time, each in its own process')
    p.add_argument('--decoder-jobs', type=int, default=10,
                   help='decoder units running at the same time (CPU-bound, one core each); --jobs limits the others')
    p.add_argument('--crop-workers', type=int, default=3)
    p.add_argument('--min-free-gb', type=float, default=5.0, help='start a unit process only with this much free RAM')
    p.add_argument('--stagger', type=float, default=45.0, help='seconds between starting two unit processes')
    p.add_argument('--list', action='store_true', help='show the units and whether they are done, run nothing')
    p.add_argument('--prune-logits', action='store_true',
                   help='delete the stored logits of the OCR runs whose decoders are all done, run nothing')
    p.add_argument('--allow-dirty', action='store_true')
    p.add_argument('--force-environment', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    for name in ('data', 'models', 'results'):
        if not getattr(args, name):
            sys.exit(f'--{name} (or its environment variable) is required')
    if args.pages and os.path.abspath(args.results) == os.path.abspath(os.environ.get('PERO_INFERENCE', '')):
        sys.exit('--pages makes incomplete records: give them a --results folder of their own')

    folder_name, commit = commit_folder_name(args.allow_dirty)
    inference = os.path.join(args.results, folder_name, 'inference')
    work = []
    for name in dataset_folders(args.data, args.datasets):
        dataset = Dataset(os.path.join(args.data, name))
        if args.pages:
            dataset.pages = dataset.pages[:args.pages]
        registry = Registry(os.path.join(args.data, name, 'registry.yaml'), args.models)
        out = os.path.join(inference, name)
        units = plan(registry, out)
        if args.exact:
            units = [u for u in units if u.id in set(args.units)]
        else:
            units = [u for u in units if any(u.id.startswith(prefix) for prefix in args.units)]
        work.append((dataset, registry, out, units))

    if args.prune_logits:
        for dataset, registry, out, _ in work:
            prune_logits(registry, out)
        return
    if args.list:
        for dataset, registry, out, units in work:
            print(f'{dataset.name} ({len(dataset.pages)} pages):')
            for u in units:
                note = ''
                if u.stage == 'OCR' and registry.base(u.name) != u.name:
                    note = f'  (derived from OCR/{registry.base(u.name)})'
                print('  done   ' if u.done() else '  missing', u.id + note)
        return
    if not any(not u.done() for *_, units in work for u in units):
        print('nothing to do')
        return
    os.makedirs(inference, exist_ok=True)
    check_record(os.path.join(inference, 'environment.json'), environment(), ENV_KEYS,
                 'in another environment', args.force_environment)
    failed = []
    for dataset, registry, out, units in work:
        todo = [u for u in units if not u.done()]
        if not todo:
            continue
        os.makedirs(out, exist_ok=True)
        if not args.exact:  # a unit process: the run that started it checked the dataset
            check_record(os.path.join(out, 'dataset.json'), dict(pages=len(dataset.all_pages),
                                                                 files=dataset.fingerprint()),
                         ('pages', 'files'), 'from other pages', args.force_environment)
        logger.info(f'{out}: {len(todo)} units to run')
        if args.exact or args.jobs <= 1:
            run_inline(todo, registry, dataset, out, args.crop_workers)
        else:
            failed += run_parallel(todo, registry, dataset, out, args, inference, folder_name)
        write_report(inference, folder_name)
    if failed:
        sys.exit(f'failed: {"; ".join(failed)}')
    logger.info('all units done')


if __name__ == '__main__':
    main()
