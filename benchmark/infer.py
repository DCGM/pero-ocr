#!/usr/bin/env python3
"""Run the engines of each dataset's registry on the dataset's pages and store what they produce, under the pero-ocr
commit.

    python benchmark/infer.py --jobs 6               # every dataset: everything not in the records yet
    python benchmark/infer.py --list                 # the units and which are done
    python benchmark/infer.py --datasets arabic      # only some datasets (rarely needed)
    python benchmark/infer.py --units OCR/           # only some units (rarely needed)
    python benchmark/infer.py --prune-logits         # delete the stored OCR logits no decoder still needs
    python benchmark/infer.py --out ~/bench/try1     # a free run: any code, environment, engines or data

Official runs (the default) record what a commit produces: the working tree must be clean and the records go to
<PERO_INFERENCE>/<commit>/. A free run (--out) writes the same structure into any folder without looking at the
commit, for experiments (another Python environment, uncommitted code, one's own engines or data); it records the
commit, the uncommitted changes and the command in <folder>/inference/run.json.

Inference is not a test: it only records what the code at this commit produces. benchmark/score.py turns the records
into the benchmark, and test/inference/ compares the records of two commits.

Inputs (environment variables or options):
    PERO_TEST_DATA     the data folder: one folder per dataset, <dataset>/registry.yaml (which engines run on it),
                       <dataset>/images/<page>.<ext>, <dataset>/page_xml/<page>.xml (the input layout; the croppers
                       crop its lines); a page is named by its file name. Other files (test lists) are not read.
    PERO_TEST_MODELS   the engines: <STAGE>/<name>/ with a config.ini holding the stage's section and the files it
                       refers to (STAGE: LAYOUT_PARSER, LINE_CROPPER, OCR, DECODER)
    PERO_INFERENCE     the records of official runs: <PERO_INFERENCE>/<commit>/inference/<dataset>/ (a free run:
                       <--out>/inference/<dataset>/)

The configs are used as they are; the code only sets the RUN_* switches of [PAGE_PARSER] to the engine's stage. Adding
an engine to a registry and running infer.py again computes only what is missing.

A run is on every page of the dataset, or on the pages of a page list the registry gives it (see Registry). A unit
records the pages it holds; when the registry gives it pages it does not hold (a longer page list, or none: every
page), a run computes only those and adds them to the unit (with the engine it was recorded with). A run adds its
pages in chunks of --chunk-pages pages, so an interrupted run loses at most one chunk: running the same command
again computes the pages the units do not hold yet. A run reads only
pages its input holds: the registry is refused at the start when it gives a run pages its input run does not get.

Units (one folder each in <commit>/inference/<dataset>/, complete when it holds unit.json):
    LAYOUT_PARSER/<layout>          the layout engine on every page image: page_xml/<page>.xml
    LINE_CROPPER/<cropper>          the line cropper on the lines of the dataset's PAGE XML: crops.lmdb, key
                                    "<page>/<line_id>", the crop as PNG (lossless)
    OCR/<ocr>@<cropper>             the OCR engine on the stored crops of the cropper: lines.tsv (text, confidence,
                                    kept), logprobs.npz (full log-probabilities of a fixed subset of lines, the same
                                    subset for every engine). An OCR run a decoder decodes also keeps its full output
                                    of every page, as the ScribbleSense worker stores it for the LM stage:
                                    page_xml/<page>.xml and logits/<page>.logits (PageLayout.save_logits_bytes), of
                                    the pages its decoders decode; --prune-logits deletes logits/ once all its
                                    decoders are done.
                                    An OCR engine that differs from another only in a higher [PAGE_PARSER]
                                    FILTER_CONFIDENT_LINES_THRESHOLD (same files, same config otherwise, same cropper)
                                    is not run: its unit is the other's lines without the lines of confidence <= the
                                    threshold, as PageParser removes them.
    DECODER/<lm>@<ocr>@<cropper>    the decoder on the stored output of the OCR run (where that output is not stored,
                                    the OCR engine runs again and must give the recorded lines): lines.tsv

Every unit.json has "pages", the pages the unit holds (records made before page lists: their number, the first N
pages). Every line of a page is inferred. Every page is processed by a fresh numpy random state seeded from its name, so
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
# A unit adds its computed pages to its folder every CHUNK_PAGES pages (--chunk-pages). 25 pages are under 10 minutes
# for the slowest units of the eb446a3 run (decoders, ~22 s a page) and half a minute for a median unit (~0.7 s a page).
CHUNK_PAGES = 25
KEEP_TORCHSCRIPT_FUSER = False  # --keep-torchscript-fuser
BIG_MODEL_MB = 300  # a transformer OCR engine with more model than this is big (--big-jobs)


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
    """A registry value: one name, a list of names, or names with their page lists ({name: page list or None}, also
    as list items "- name: page list"). Returns {name: page list or None}; None is every page."""
    def page_list(v, name):
        if v is None or isinstance(v, str):
            return v
        sys.exit(f'registry: {where}: {name}: expected a page list file, got {v!r}')
    items = value if isinstance(value, list) else [value]
    out = {}
    for item in items:
        if isinstance(item, str):
            pairs = [(item, None)]
        elif isinstance(item, dict) and item and all(isinstance(k, str) for k in item):
            pairs = [(k, page_list(v, k)) for k, v in item.items()]
        else:
            sys.exit(f'registry: {where}: expected a name, a list of names or names with page lists, got {value!r}')
        for name, pages in pairs:
            if name in out:
                sys.exit(f'registry: {where}: {name} is named twice')
            out[name] = pages
    if not out:
        sys.exit(f'registry: {where}: no name given')
    return out


class Engine:
    def __init__(self, stage, name, folder, inputs, pages=None):
        self.stage, self.name, self.folder = stage, name, folder
        self.inputs = inputs  # {name: page list or None} of the INPUT stage (OCR: croppers; DECODER: OCR runs)
        self.pages = pages  # LAYOUT_PARSER, LINE_CROPPER: the page list it runs on (None: every page)

    def config(self):
        """The engine's config.ini as it is, with the RUN_* switches of [PAGE_PARSER] set to its stage and
        TORCHSCRIPT_FUSER = no: with TorchScript's fuser on, a model's output on the GPU depends on the pages processed
        before it in the process, so a page would not get the same output in a full run, a resumed run or a top-up
        (--keep-torchscript-fuser leaves the config without it, to add to records made before)."""
        config = configparser.ConfigParser()
        config.read(os.path.join(self.folder, 'config.ini'))
        if not config.has_section('PAGE_PARSER'):
            config.add_section('PAGE_PARSER')
        for stage, switch in RUN_SWITCH.items():
            config['PAGE_PARSER'][switch] = 'yes' if stage == self.stage else 'no'
        if not KEEP_TORCHSCRIPT_FUSER:
            config['PAGE_PARSER']['TORCHSCRIPT_FUSER'] = 'no'
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
    <names> is one name or a list. Any subset of the stages is valid.

    Page lists: a run is on every page of the dataset, or on the pages of a page list given with its name: a layout
    engine or cropper as "<name>: <page list>", an OCR or decoder run as "<input>: <page list>" in its LINE_CROPPER or
    OCR. A page list is a file relative to the dataset folder, one page name per line (blank lines and # comments
    ignored). A run's pages must be among the pages of the run it reads (OCR: its cropper's; DECODER: its OCR run's;
    an OCR run derived by a higher threshold: its base run's)."""

    def __init__(self, path, engines_root, dataset):
        self.path, self.engines_root, self.dataset = path, engines_root, dataset
        self.page_lists = {}
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
            if stage not in INPUT:
                entries = names(entries, stage)  # {name: page list or None}
            elif isinstance(entries, (str, list)):
                entries = {e: None for e in names(entries, stage)}
            if not isinstance(entries, dict):
                sys.exit(f'{path}: {stage}: expected a list of engines or engines with what they run on')
            for name, settings in entries.items():
                self.engines[stage][name] = self._engine(stage, name, settings)
        self._base = None
        self._check()
        self._check_pages()

    def _engine(self, stage, name, settings):
        where = f'{stage}: {name}'
        if '@' in name:
            sys.exit(f'registry: {where}: "@" separates the parts of a run name; an engine name must not contain it')
        folder = os.path.join(self.engines_root, stage, name)
        if not os.path.isfile(os.path.join(folder, 'config.ini')):
            sys.exit(f'registry: {where}: no {os.path.join(folder, "config.ini")}')
        if stage not in INPUT:
            return Engine(stage, name, folder, {}, settings)  # settings: its page list or None
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {INPUT[stage]}:
            sys.exit(f'registry: {where}: a {stage} engine takes {INPUT[stage]} (what it runs on)')
        if INPUT[stage] not in settings:
            sys.exit(f'registry: {where}: {INPUT[stage]} (what it runs on) is required')
        return Engine(stage, name, folder, names(settings[INPUT[stage]], f'{where}: {INPUT[stage]}'))

    def page_list(self, name, where):
        """The pages of a page list file (relative to the dataset folder)."""
        if name is None:
            return frozenset(p['name'] for p in self.dataset.all_pages)
        if name not in self.page_lists:
            path = os.path.join(self.dataset.folder, name)
            if not os.path.isfile(path):
                sys.exit(f'registry: {where}: no page list {path}')
            with open(path, encoding='utf-8') as f:
                pages = {line.split('#', 1)[0].strip() for line in f} - {''}
            if not pages:
                sys.exit(f'registry: {where}: page list {name} is empty')
            unknown = sorted(pages - {p['name'] for p in self.dataset.all_pages})
            if unknown:
                sys.exit(f'registry: {where}: page list {name}: {len(unknown)} pages are not in the dataset: '
                         f'{", ".join(unknown[:5])}')
            self.page_lists[name] = frozenset(pages)
        return self.page_lists[name]

    def _check_pages(self):
        """The pages of every run, and that a run reads only pages its input run has."""
        self.unit_pages = {}  # unit id: (pages, page list name or None)
        for stage in ('LAYOUT_PARSER', 'LINE_CROPPER'):
            for e in self.engines[stage].values():
                uid = f'{stage}/{e.name}'
                self.unit_pages[uid] = (self.page_list(e.pages, uid), e.pages)
        for e in self.engines['OCR'].values():
            for cropper, pages in e.inputs.items():
                uid = f'OCR/{e.name}@{cropper}'
                self.unit_pages[uid] = (self.page_list(pages, uid), pages)
        for e, run in self.decoder_runs():
            uid = f'DECODER/{e.name}@{run}'
            self.unit_pages[uid] = (self.page_list(e.inputs[run], uid), e.inputs[run])

        def describe(uid):
            pages, name = self.unit_pages[uid]
            return 'every page' if name is None else f'the {len(pages)} pages of {name}'

        def subset(uid, input_uid, how):
            extra = self.unit_pages[uid][0] - self.unit_pages[input_uid][0]
            if extra:
                sys.exit(f'registry: {uid} runs on {describe(uid)}, {input_uid} ({how}) on {describe(input_uid)}: '
                         f'{len(extra)} pages it needs are not there ({", ".join(sorted(extra)[:5])}); give the '
                         f'input run these pages too, or the run fewer')
        for run in self.ocr_runs():
            subset(f'OCR/{run}', f"LINE_CROPPER/{run.split('@', 1)[1]}", 'its cropper')
            if self.base(run) != run:
                subset(f'OCR/{run}', f'OCR/{self.base(run)}', 'it is derived from it by a higher threshold')
        for e, run in self.decoder_runs():
            subset(f'DECODER/{e.name}@{run}', f'OCR/{run}', 'the OCR run it decodes')

    def pages(self, uid):
        """The pages the registry gives a unit."""
        return self.unit_pages[uid][0]

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
            resolved = {}
            for ref, pages in e.inputs.items():
                if ref not in runs:
                    ocr = self.engines['OCR'].get(ref)
                    if ocr is None:
                        sys.exit(f'registry: DECODER: {e.name}: OCR {ref} is not in the registry')
                    if len(ocr.inputs) > 1:
                        sys.exit(f'registry: DECODER: {e.name}: OCR {ref} runs with several croppers; name one as '
                                 f'{ref}@<cropper>')
                    ref = f'{ref}@{next(iter(ocr.inputs))}'
                if ref in resolved:
                    sys.exit(f'registry: DECODER: {e.name}: OCR run {ref} is named twice')
                resolved[ref] = pages
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

    def kept_pages(self, base):
        """The pages of an OCR run whose full output is kept: the pages its decoders decode."""
        return frozenset().union(*(self.pages(f'DECODER/{e.name}@{run}') for e, run in self.decoder_runs()
                                   if self.base(run) == base))


# ----------------------------------------------------------------------------------------------------------------------
# Commit and environment


def git(*args):
    return subprocess.run(['git', '-C', REPO] + list(args), check=True, capture_output=True, text=True).stdout.strip()


def official_commit():
    """The commit of an official run; the working tree must be clean."""
    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty:
        sys.exit('The working tree has uncommitted changes; the records would not belong to the commit. Commit them, '
                 'or make a free run into a folder of your own (--out).\n' + dirty)
    return commit


def record_free_run(inference):
    """Append to <inference>/run.json what a free run ran with: the commit, uncommitted changes, the command."""
    run = dict(started_utc=utc_now(), argv=sys.argv, python=sys.executable, cwd=os.getcwd())
    try:
        run['commit'] = git('rev-parse', 'HEAD')
        run['dirty'] = git('status', '--porcelain', '--untracked-files=no').splitlines()
        run['diff'] = git('diff', 'HEAD')
    except (subprocess.CalledProcessError, OSError):
        run['commit'] = None  # pero-ocr not in a git checkout
    path = os.path.join(inference, 'run.json')
    runs = []
    if os.path.exists(path):
        with open(path) as f:
            runs = json.load(f)
    with open(path, 'w') as f:
        json.dump(runs + [run], f, indent=1, ensure_ascii=False)


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


def recorded_pages(folder, dataset):
    """The pages a finished unit holds (set; empty when it is not finished). Records made before page lists hold the
    first <pages> pages of the dataset (every page, or the first N of a trial)."""
    path = os.path.join(folder, 'unit.json')
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        pages = json.load(f)['pages']
    if isinstance(pages, int):
        return {p['name'] for p in dataset.all_pages[:pages]}
    return set(pages)


class Unit:
    """A unit's folder holds the pages it was run on (unit.json "pages"). A run computes the pages the registry gives
    it that the unit does not hold yet into <unit>.partial/ and then adds them to the unit (a top-up), or, for a new
    unit, renames the folder."""

    def __init__(self, stage, name, out, dataset, wanted):
        self.stage, self.name, self.dataset = stage, name, dataset
        self.wanted = wanted  # the pages the registry gives it (of the pages processed, --pages)
        self.final = os.path.join(out, stage, name)
        self.partial = self.final + '.partial'

    @property
    def id(self):
        return f'{self.stage}/{self.name}'

    def info(self):
        path = os.path.join(self.final, 'unit.json')
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def recorded(self):
        return recorded_pages(self.final, self.dataset)

    def missing_pages(self):
        """The pages to compute, in dataset order."""
        have = self.recorded()
        return [p for p in self.dataset.all_pages if p['name'] in self.wanted and p['name'] not in have]

    def done(self):
        return not self.missing_pages()

    def check_engine(self, files, config):
        """A top-up only with the engine the unit was recorded with."""
        info = self.info()
        if info is not None and (info.get('files') != files or info.get('config') != config):
            sys.exit(f'{self.id}: the engine (its files or config) is not the one the unit was recorded with; adding '
                     f'pages would mix two engines: delete {self.final} to run it again')

    def start(self, pages, replace=False):
        """Compute <pages> (page dicts); replace: compute the unit anew instead of adding to it."""
        self.pages = pages
        self.topup = not replace and self.info() is not None
        shutil.rmtree(self.partial, ignore_errors=True)
        os.makedirs(self.partial)
        self.started = time.time()
        self.started_utc = utc_now()
        if self.topup:
            logger.info(f'{self.id}: adding {len(pages)} pages to the {len(self.recorded())} it holds')

    def finish(self, info):
        timing = dict(started_utc=self.started_utc, finished_utc=utc_now(),
                      seconds=round(time.time() - self.started, 1))
        new = [p['name'] for p in self.pages]
        if not self.topup:
            info = dict(unit=self.id, **timing, **info, pages=new)
            with open(os.path.join(self.partial, 'unit.json'), 'w') as f:
                json.dump(info, f, indent=1, ensure_ascii=False)
            shutil.rmtree(self.final, ignore_errors=True)
            os.rename(self.partial, self.final)
        else:
            old = self.info()
            order = {p['name']: i for i, p in enumerate(self.dataset.all_pages)}
            counts = merge_unit(self.partial, self.final, set(new), order)
            merged = dict(old, **{k: v for k, v in info.items() if k not in ('lines', 'logprobs_lines')}, **counts)
            merged['page_output'] = old.get('page_output', False) or info.get('page_output', False)
            if 'lines' not in counts:  # layout, crops: the new pages' lines with the old
                merged['lines'] = old.get('lines', 0) + info.get('lines', 0)
            merged['pages'] = sorted(self.recorded() | set(new), key=order.get)
            merged['topups'] = old.get('topups', []) + [dict(timing, pages=len(new), lines=info.get('lines'))]
            if 'page_output' not in old and 'page_output' not in info:
                merged.pop('page_output')
            tmp = os.path.join(self.final, '.unit.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(merged, f, indent=1, ensure_ascii=False)
            os.replace(tmp, os.path.join(self.final, 'unit.json'))  # the unit holds the new pages from here on
            shutil.rmtree(self.partial)
        held = len(self.recorded() & set(self.wanted))
        logger.info(f'{self.id}: {len(new)} pages saved in {timing["seconds"]:.0f} s, holds {held} of its '
                    f'{len(self.wanted)} pages' + (' (done)' if held == len(self.wanted) else ''))


def merge_unit(partial, final, new_pages, order):
    """Add the pages computed into <partial> to the unit in <final>. Safe to repeat after an interruption: the rows of
    the new pages already in <final> are replaced. Returns the counts of the merged files."""
    counts = {}
    for sub in ('page_xml', 'logits'):
        src = os.path.join(partial, sub)
        if os.path.isdir(src):
            os.makedirs(os.path.join(final, sub), exist_ok=True)
            for f in os.listdir(src):
                os.replace(os.path.join(src, f), os.path.join(final, sub, f))
    if os.path.exists(os.path.join(partial, 'lines.tsv')):
        old, new = read_lines(os.path.join(final, 'lines.tsv')), read_lines(os.path.join(partial, 'lines.tsv'))
        rows = [r for r in old[1:] if r[0] not in new_pages] + new[1:]
        rows.sort(key=lambda r: order[r[0]])  # stable: a page's lines stay in their order
        tmp = os.path.join(final, '.lines.tsv.tmp')
        with open(tmp, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
            w.writerows([old[0]] + rows)
        os.replace(tmp, os.path.join(final, 'lines.tsv'))
        counts['lines'] = len(rows)
    if os.path.exists(os.path.join(partial, 'logprobs.npz')):
        with np.load(os.path.join(final, 'logprobs.npz')) as f:
            logits = {k: f[k] for k in f.files if k.split('/', 1)[0] not in new_pages}
        with np.load(os.path.join(partial, 'logprobs.npz')) as f:
            logits.update({k: f[k] for k in f.files})
        logits = dict(sorted(logits.items(), key=lambda kv: order[kv[0].split('/', 1)[0]]))
        tmp = os.path.join(final, '.logprobs.tmp.npz')
        np.savez(tmp, **logits)
        os.replace(tmp, os.path.join(final, 'logprobs.npz'))
        counts['logprobs_lines'] = len(logits)
    if os.path.isdir(os.path.join(partial, 'crops.lmdb')):
        import lmdb
        src = lmdb.open(os.path.join(partial, 'crops.lmdb'), subdir=True, readonly=True, lock=False)
        dst = lmdb.open(os.path.join(final, 'crops.lmdb'), map_size=LMDB_MAP_SIZE, subdir=True)
        with src.begin() as r, dst.begin(write=True) as w:
            for key, value in r.cursor():
                w.put(key, value)
        src.close()
        dst.close()
    return counts


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def chunks(items):
    """<items> in chunks of CHUNK_PAGES (the caller adds each chunk to its unit before taking the next, so an
    interruption loses at most one chunk)."""
    for i in range(0, len(items), CHUNK_PAGES):
        yield items[i:i + CHUNK_PAGES]


def plan(registry, out, dataset):
    """All units, in run order, each with the pages the registry gives it (of the pages processed)."""
    processed = {p['name'] for p in dataset.pages}
    ids = [('LAYOUT_PARSER', n) for n in registry.engines['LAYOUT_PARSER']]
    ids += [('LINE_CROPPER', n) for n in registry.engines['LINE_CROPPER']]
    ids += [('OCR', r) for r in registry.ocr_runs()]
    ids += [('DECODER', f'{e.name}@{run}') for e, run in registry.decoder_runs()]
    return [Unit(stage, name, out, dataset, registry.pages(f'{stage}/{name}') & processed) for stage, name in ids]


def need_pages(uid, out, dataset, pages):
    """Stop unless the finished unit <uid> holds <pages> (page dicts): a unit reads them from it."""
    have = recorded_pages(os.path.join(out, uid), dataset)
    lacking = [p['name'] for p in pages if p['name'] not in have]
    if lacking:
        sys.exit(f'{uid} does not hold {len(lacking)} of the pages needed from it ({", ".join(lacking[:5])}); run it '
                 f'first')


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
    unit.check_engine(engine.files(), config_text(config))
    parser = PageParser(config, config_path=engine.folder)
    for pages in chunks(unit.missing_pages()):
        unit.start(pages)
        os.makedirs(os.path.join(unit.partial, 'page_xml'))
        n_lines = 0
        for page, image, _ in read_pages(pages, with_layout=False):
            np.random.seed(page_seed(page['name']))
            layout = PageLayout(id=page['name'], page_size=(image.shape[0], image.shape[1]))
            layout = parser.process_page(image, layout)
            layout.to_pagexml(os.path.join(unit.partial, 'page_xml', page['name'] + '.xml'))
            n_lines += sum(1 for _ in layout.lines_iterator())
        unit.finish(dict(engine=unit.name, files=engine.files(), config=config_text(config), lines=n_lines))
    del parser


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
    need = {}
    for u in units:
        u.check_engine(engines[u.name].files(), configs[u.name][0])
        need[u.name] = {p['name'] for p in u.missing_pages()}
    jobs = [(page, {n: c for n, c in configs.items() if page['name'] in need[n]}) for page in dataset.all_pages
            if any(page['name'] in pages for pages in need.values())]
    with multiprocessing.get_context('spawn').Pool(workers, initializer=os.nice, initargs=(10,)) as pool:
        for chunk in chunks(jobs):
            active = [u for u in units if any(u.name in job[1] for job in chunk)]
            envs, counts = {}, {u.name: 0 for u in active}
            for u in active:
                u.start([page for page, cropper_configs in chunk if u.name in cropper_configs])
                envs[u.name] = lmdb.open(os.path.join(u.partial, 'crops.lmdb'), map_size=LMDB_MAP_SIZE, subdir=True)
            for result in pool.imap(_crop_page, chunk, chunksize=2):
                for name, items in result.items():
                    with envs[name].begin(write=True) as txn:
                        for key, png in items:
                            txn.put(key.encode(), png)
                    counts[name] += len(items)
            for u in active:
                envs[u.name].close()
                u.finish(dict(engine=u.name, files=engines[u.name].files(), config=configs[u.name][0],
                              lines=counts[u.name], format='png'))


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
    engine = registry.ocr_engine(unit.name)
    unit.check_engine(engine.files(), config_text(engine.config()))
    pages = unit.missing_pages()
    need_pages(f"LINE_CROPPER/{unit.name.split('@', 1)[1]}", out, dataset, pages)
    parser, config = ocr_parser(registry, unit.name)
    env = crops_env(out, unit.name.split('@', 1)[1])
    kept = registry.kept_pages(unit.name) if unit.name in registry.kept_output() else frozenset()
    for chunk in chunks(pages):
        unit.start(chunk)
        if kept:
            os.makedirs(os.path.join(unit.partial, 'page_xml'))
            os.makedirs(os.path.join(unit.partial, 'logits'))
        writer = LineWriter(os.path.join(unit.partial, 'lines.tsv'))
        logits = {}
        for page, image, layout in read_pages(chunk, crops=env):
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
            if name in kept:
                layout.to_pagexml(os.path.join(unit.partial, 'page_xml', name + '.xml'))
                with open(os.path.join(unit.partial, 'logits', name + '.logits'), 'wb') as f:
                    f.write(layout.save_logits_bytes(missing_line_logits_ok=True))
        writer.close()
        np.savez(os.path.join(unit.partial, 'logprobs.npz'), **logits)
        unit.finish(dict(engine=engine.name, files=engine.files(), config=config_text(config), lines=writer.n,
                         logprobs_lines=len(logits),
                         logprobs_subset=f'1/{LOGITS_SUBSET} by md5 of "<page>/<line_id>"', page_output=bool(kept)))
    del parser
    env.close()


def run_derived(unit, registry, dataset, out):
    """An OCR run that differs from its base only in a higher FILTER_CONFIDENT_LINES_THRESHOLD: the base's lines,
    with the lines the filter removes (confidence <= threshold, as PageParser filters) marked kept=0. Cheap, so it is
    derived anew on all its pages (those it holds and those it gets) whenever pages are added."""
    base = registry.base(unit.name)
    base_folder = os.path.join(out, 'OCR', base)
    threshold = registry.threshold(unit.name)
    pages = unit.recorded() | set(unit.wanted)
    page_dicts = [p for p in dataset.all_pages if p['name'] in pages]
    need_pages(f'OCR/{base}', out, dataset, page_dicts)
    unit.start(page_dicts, replace=True)
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
                     lines=n, removed_lines=len(removed), logprobs_lines=len(logits)))


def ocr_output(run, pages, registry, dataset, out):
    """The folder with the stored output (page_xml/, logits/) of an OCR run, for <pages>. Pages whose output is not
    stored (pruned, or no decoder decoded the page when it ran) are run again; their lines must equal the unit's
    lines.tsv (same code, same seeds), otherwise the records do not belong together."""
    folder = os.path.join(out, 'OCR', run)
    need_pages(f'OCR/{run}', out, dataset, pages)
    missing = [p for p in pages if not os.path.exists(os.path.join(folder, 'logits', p['name'] + '.logits'))]
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
    config = engine.config()
    unit.check_engine(engine.files(), config_text(config))
    pages = unit.missing_pages()
    need_pages(f'OCR/{run}', out, dataset, pages)
    base = registry.base(run)
    ocr_folder = ocr_output(base, pages, registry, dataset, out)
    removed = set()
    if base != run:
        removed = {(r[0], r[1]) for r in read_lines(os.path.join(ocr_unit, 'lines.tsv'))[1:] if r[4] == '0'}
    decoder = PageParser(config, config_path=engine.folder)
    with open(os.path.join(ocr_unit, 'unit.json')) as f:
        ocr_info = json.load(f)
    for chunk in chunks(pages):
        unit.start(chunk)
        writer = LineWriter(os.path.join(unit.partial, 'lines.tsv'))
        for page in chunk:
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
        unit.finish(dict(engine=lm, files=engine.files(), config=config_text(config), ocr_run=run,
                         ocr_unit_finished_utc=ocr_info['finished_utc'], lines=writer.n))


def prune_logits(registry, out, units):
    """Delete OCR/<run>/logits/ of the OCR runs whose decoders are all done (a decoder added later runs the OCR
    engine again)."""
    for base in sorted(registry.kept_output()):
        folder = os.path.join(out, 'OCR', base, 'logits')
        decoders = [f'DECODER/{e.name}@{run}' for e, run in registry.decoder_runs() if registry.base(run) == base]
        if os.path.isdir(folder) and all(units[d].done() for d in decoders):
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
                    pages = info.get('pages', '')
                    seconds = info['seconds'] + sum(t['seconds'] for t in info.get('topups', []))
                    finished = info['topups'][-1]['finished_utc'] if info.get('topups') else info['finished_utc']
                    rows.append((dataset, stage, name, len(pages) if isinstance(pages, list) else pages,
                                 info.get('lines', ''), seconds, finished))
    with open(os.path.join(inference, 'environment.json')) as f:
        env = json.load(f)
    lines = [f'# Inference {commit}', '',
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


def order(todo, registry):
    """The units that run in processes of their own: layout and OCR first, decoders last (their results come after
    every OCR result); within that, most expensive first, a unit counting with the units that wait for it."""
    own = {}
    for u in todo:
        n = len(u.missing_pages())
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


def run_parallel(todo, registry, dataset, out, args, inference, folder_name, units):
    """Crops in this process (a CPU pool), every other unit in a process of its own, args.jobs at a time, at low CPU
    priority; a unit starts when the units it needs are done and enough memory is free."""
    crops = [u for u in todo if u.stage == 'LINE_CROPPER']
    lock, failed, finished = threading.Lock(), [], set()
    running = {'gpu': 0, 'big': 0}

    def big(unit):
        """An OCR unit that runs a big transformer (more than BIG_MODEL_MB of model files): the heaviest GPU load. A
        derived run computes nothing."""
        if unit.stage != 'OCR' or registry.base(unit.name) != unit.name:
            return False
        engine = registry.ocr_engine(unit.name)
        if engine.config()['OCR'].get('METHOD') != 'pytorch_ocr-transformer':
            return False
        size = sum(os.path.getsize(os.path.join(engine.folder, f)) for f in os.listdir(engine.folder)
                   if os.path.isfile(os.path.join(engine.folder, f)))
        return size > BIG_MODEL_MB * 2 ** 20

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
    pending = order(todo, registry)
    todo_ids = {u.id for u in todo}
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    base = [sys.executable, os.path.abspath(__file__), '--exact', '--data', args.data, '--models', args.models,
            '--datasets', dataset.name, '--crop-workers', '0', '--decoder-jobs', '1'] + \
           (['--out', args.out] if args.out else ['--results', args.results]) + \
           (['--pages', str(args.pages)] if args.pages else []) + \
           (['--force-environment'] if args.force_environment else []) + ['--chunk-pages', str(args.chunk_pages)] + \
           (['--keep-torchscript-fuser'] if args.keep_torchscript_fuser else [])

    def done(uid):
        """The unit holds the pages the registry gives it."""
        return units[uid].done()

    def next_unit():
        """The first pending unit that can start, 'wait', or None when nothing is left."""
        for i, unit in enumerate(pending):
            if unit.stage != 'DECODER' and running['gpu'] >= args.jobs:
                continue
            if big(unit) and running['big'] >= args.big_jobs:
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
                    running['big'] += big(unit)
                    time.sleep(args.stagger)  # let it load its models before the memory check of the next one
            if unit == 'wait':
                time.sleep(10)
                continue
            code = proc.wait()
            log.close()
            with lock:
                if unit.stage != 'DECODER':
                    running['gpu'] -= 1
                running['big'] -= big(unit)
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
    global CHUNK_PAGES, KEEP_TORCHSCRIPT_FUSER
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=os.environ.get('PERO_TEST_DATA'))
    p.add_argument('--models', default=os.environ.get('PERO_TEST_MODELS'))
    p.add_argument('--results', default=os.environ.get('PERO_INFERENCE'), help='the records of official runs')
    p.add_argument('--out', help='a free run into this folder: not bound to the commit (experiments)')
    p.add_argument('--datasets', nargs='*', help='only these datasets of the data folder')
    p.add_argument('--units', nargs='*', default=[''], help='unit id prefixes to run, e.g. OCR/ DECODER/czech')
    p.add_argument('--exact', action='store_true', help='--units are exact unit ids; run them in this process (how the unit processes are started)')
    p.add_argument('--pages', type=int, help='only the first N pages of each dataset (a trial; only with --out)')
    p.add_argument('--jobs', type=int, default=1, help='units running at the same time, each in its own process')
    p.add_argument('--decoder-jobs', type=int, default=10,
                   help='decoder units running at the same time (CPU-bound, one core each); --jobs limits the others')
    p.add_argument('--crop-workers', type=int, default=3)
    p.add_argument('--chunk-pages', type=int, default=CHUNK_PAGES,
                   help='a unit adds its computed pages to its folder every this many pages')
    p.add_argument('--big-jobs', type=int, default=1,
                   help=f'at most this many big transformer OCR units (over {BIG_MODEL_MB} MB of model) at a time, of '
                        f'--jobs: several at once overload the GPU')
    p.add_argument('--keep-torchscript-fuser', action='store_true',
                   help="leave TorchScript's fuser on (pero-ocr's default): only to add to records made with it on")
    p.add_argument('--min-free-gb', type=float, default=5.0, help='start a unit process only with this much free RAM')
    p.add_argument('--stagger', type=float, default=45.0, help='seconds between starting two unit processes')
    p.add_argument('--list', action='store_true', help='show the units and whether they are done, run nothing')
    p.add_argument('--prune-logits', action='store_true',
                   help='delete the stored logits of the OCR runs whose decoders are all done, run nothing')
    p.add_argument('--force-environment', action='store_true')
    args = p.parse_args()
    if args.chunk_pages < 1:
        sys.exit('--chunk-pages must be at least 1')
    CHUNK_PAGES = args.chunk_pages
    KEEP_TORCHSCRIPT_FUSER = args.keep_torchscript_fuser
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    for name in ('data', 'models') + (() if args.out else ('results',)):
        if not getattr(args, name):
            sys.exit(f'--{name} (or its environment variable) is required')
    if args.pages and not args.out:
        sys.exit('--pages makes incomplete records: only in a free run (--out)')
    if args.keep_torchscript_fuser and not args.out:
        sys.exit('--keep-torchscript-fuser: only in a free run (--out); official runs are made with the fuser off')

    if args.out:
        folder_name = os.path.basename(os.path.normpath(args.out))
        inference = os.path.join(args.out, 'inference')
    else:
        folder_name = official_commit()
        inference = os.path.join(args.results, folder_name, 'inference')
    work = []
    for name in dataset_folders(args.data, args.datasets):
        dataset = Dataset(os.path.join(args.data, name))
        if args.pages:
            dataset.pages = dataset.pages[:args.pages]
        registry = Registry(os.path.join(args.data, name, 'registry.yaml'), args.models, dataset)
        out = os.path.join(inference, name)
        all_units = {u.id: u for u in plan(registry, out, dataset)}
        if args.exact:
            units = [u for u in all_units.values() if u.id in set(args.units)]
        else:
            units = [u for u in all_units.values() if any(u.id.startswith(prefix) for prefix in args.units)]
        work.append((dataset, registry, out, units, all_units))

    if args.prune_logits:
        for dataset, registry, out, _, all_units in work:
            prune_logits(registry, out, all_units)
        return
    if args.list:
        for dataset, registry, out, units, _ in work:
            print(f'{dataset.name} ({len(dataset.pages)} pages):')
            for u in units:
                notes = []
                if registry.unit_pages[u.id][1] is not None:
                    notes.append(f'pages of {registry.unit_pages[u.id][1]}')
                if u.stage == 'OCR' and registry.base(u.name) != u.name:
                    notes.append(f'derived from OCR/{registry.base(u.name)}')
                have, missing = len(u.recorded() & u.wanted), len(u.missing_pages())
                state = 'done   ' if not missing else 'missing' if not have else 'partial'
                count = f'{have}/{len(u.wanted)} pages'
                print(f'  {state} {u.id}  ({count}{"".join("; " + n for n in notes)})')
        return
    if not any(not u.done() for _, _, _, units, _ in work for u in units):
        print('nothing to do')
        return
    os.makedirs(inference, exist_ok=True)
    if args.out and not args.exact:
        record_free_run(inference)
    check_record(os.path.join(inference, 'environment.json'), environment(), ENV_KEYS,
                 'in another environment', args.force_environment)
    failed = []
    for dataset, registry, out, units, all_units in work:
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
            failed += run_parallel(todo, registry, dataset, out, args, inference, folder_name, all_units)
        write_report(inference, folder_name)
    if failed:
        sys.exit(f'failed: {"; ".join(failed)}')
    logger.info('all units done')


if __name__ == '__main__':
    main()
