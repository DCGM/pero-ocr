#!/usr/bin/env python3
"""Run the engines a registry names on the pages of its benchmark and store what they produce, under the pero-ocr
commit.

    python benchmark/infer.py --registry R.yaml --jobs 6   # everything of the registry not in the records yet
    python benchmark/infer.py --registry R.yaml S.yaml     # several registries (one per benchmark) in one run
    python benchmark/infer.py --registry R.yaml --list     # the units and which are done
    python benchmark/infer.py --registry R.yaml --units OCR/          # only some units (rarely needed)
    python benchmark/infer.py --registry R.yaml --prune-logits        # delete the OCR logits no decoder still needs
    python benchmark/infer.py --registry R.yaml --out ~/bench/try1    # a free run: any code, environment, engines

A registry (a .yaml file anywhere, see Registry) names the benchmark it runs on (BENCHMARK, a folder of the data
folder) and the engines to run on it. Official runs (the default) record what a commit produces: the working tree
must be clean and the records go to <PERO_INFERENCE>/<commit>/<benchmark>/. A free run (--out) writes the same
structure into any folder without looking at the commit, for experiments (another Python environment, uncommitted
code, one's own engines or data); it records the commit, the uncommitted changes and the command in
<folder>/<benchmark>/run.json.

The records of a benchmark are self-contained: <commit>/<benchmark>/ holds the units, environment.json (the
environment they were made in; a benchmark's units are all made in one), benchmark.json (its pages and their
checksums), adopted.json (units adopted from another commit), report.md, logs/ (one log per unit, every attempt
appended between a start and an end line) and, from score.py, score/. Nothing is
stored at the level of the commit, so different benchmarks of one commit can be made by different people on different
machines and put together by copying their folders.

Inference is not a test: it only records what the code at this commit produces. benchmark/score.py turns the records
into the benchmark, and test/inference/ compares the records of two commits.

Inputs (environment variables or options):
    --registry         the registries to run, one per benchmark
    PERO_TEST_DATA     the data folder: one folder per benchmark, <benchmark>/images/<page>.<ext>,
                       <benchmark>/page_xml/<page>.xml (the input layout; the croppers crop its lines), and the page
                       lists the registries name; a page is named by its file name. The test lists are read by score.py.
    PERO_TEST_MODELS   the engines: <STAGE>/<name>/ with a config.ini holding the stage's section and the files it
                       refers to (STAGE: LAYOUT_PARSER, LINE_CROPPER, OCR, DECODER)
    PERO_INFERENCE     the records of official runs: <PERO_INFERENCE>/<commit>/<benchmark>/ (a free run:
                       <--out>/<benchmark>/)

Guards: adding to existing records (resuming a run, adding units or pages) is refused when it would mix them, unless
the guard's flag forces it; every forced addition is noted in <benchmark>/forced.json (what differed, when, where):
    environment   the same machine (hostname) and environment (python, packages, CUDA, cuDNN, GPU, GPU driver) as
                  environment.json, and the same pages as benchmark.json                     --force-environment
    commit        official records: the folder is the commit (a clean checkout), so other code never adds to them;
                  a free run: the commit and uncommitted changes of its first run (run.json)  --force-commit
    model         a unit: the model files and config it was recorded with (engine.json)      --force-model
A run holds
<benchmark>/.lock while it adds to the records: a second run on the same records stops with who holds it (released
however a run ends; the file is deleted at its normal end).

The configs are used as they are; the code only sets the RUN_* switches of [PAGE_PARSER] to the engine's stage. Adding
an engine to a registry and running infer.py again computes only what is missing.

A run is on every page of the benchmark, or on the pages of a page list the registry gives it (see Registry). A unit
records the pages it holds; when the registry gives it pages it does not hold (a longer page list, or none: every
page), a run computes only those and adds them to the unit (with the engine it was recorded with). A run adds its
pages in chunks of --chunk-pages pages, so an interrupted run loses at most one chunk: running the same command
again computes the pages the units do not hold yet. A run reads only
pages its input holds: the registry is refused at the start when it gives a run pages its input run does not get.

Units (one folder each in <commit>/<benchmark>/, complete when it holds engine.json):
    LAYOUT_PARSER/<layout>          the layout engine on every page image: page_xml/<page>.xml
    LINE_CROPPER/<cropper>          the line cropper on the lines of the benchmark's PAGE XML: crops.png.lmdb, key
                                    "<page>/<line_id>", the crop as PNG (lossless); its pages: pages.tsv
    OCR/<ocr>@<cropper>             the OCR engine on the stored crops of the cropper, per page (records.py):
                                    lines/<page>.txt (the lines in the format of the test lists), lines/<page>.tsv
                                    (line id, confidence, kept), logprobs.1in64/<page>.npz (full log-probabilities of the
                                    page's lines in a fixed subset of lines, the same subset for every engine). An OCR
                                    run a decoder decodes also keeps its full output
                                    of every page, as a production worker stores it for the LM stage:
                                    page_xml/<page>.xml and logits/<page>.logits (PageLayout.save_logits_bytes), of
                                    the pages its decoders decode; --prune-logits deletes logits/ once all its
                                    decoders are done.
                                    An OCR engine that differs from another only in a higher [PAGE_PARSER]
                                    FILTER_CONFIDENT_LINES_THRESHOLD (same files, same config otherwise, same cropper)
                                    is not run: its unit is the other's lines without the lines of confidence <= the
                                    threshold, as PageParser removes them.
    DECODER/<lm>@<ocr>@<cropper>    the decoder on the stored output of the OCR run (where that output is not stored,
                                    the OCR engine runs again and must give the recorded lines): lines/<page>.txt,
                                    lines/<page>.tsv

A unit holds the pages whose files it has (a cropper: the pages of its pages.tsv); a page is written completely or not at
all, so an interrupted run loses at most the chunk it was computing (--chunk-pages), and deleting a page's files makes
the next run compute it again. engine.json holds the engine (unit, engine, files, config); unit.json exists only for
an adopted unit (adopted_from) or one extended with --force-model (engine_changes); pages.tsv the processing of every
page in the columns of its stage (page, seconds: wall time of the engine's call, gpu_peak_mb; layout: lines detected;
cropper: crops; OCR: lines read, chars written, width_px of the crops; decoder: lines_decoded (the lines over its
confidence threshold keep the OCR text), chars written, frames of those lines; with --jobs over 1 units share a GPU,
so the times include waiting for each other). Pages are added only
with the engine the unit was recorded with; --force-model adds them anyway and records which pages got the new engine
(unit.json "engine_changes"). Every line of a page is inferred. Every page is processed with numpy's and Python's random generators seeded
from its name, so a rerun gives the same result (the layout engine breaks ties with np.random and orders the lines of a
region with random.uniform, issue #100).

Adopting units (--adopt-from <commit or folder> --adopt <unit id prefixes>): a run whose code changes only some stages
copies the complete units of the other stages from the records of another commit instead of computing them again, e.g.
a change of the layout engine adopts LINE_CROPPER/ OCR/ DECODER/ and computes only LAYOUT_PARSER/. The source must have
been inferred on the same machine, in the same environment and on the same pages (all checked, per benchmark; or
--force-environment); that the code change
does not touch the adopted stages is the caller's claim. <commit>/<benchmark>/adopted.json records the source, the
files the code changed between the two commits and the adopted units; each adopted unit.json gets "adopted_from";
report.md says so.
"""
import argparse
import fcntl
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
import random
import re
import shlex
import socket
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import cv2
import numpy as np

import records

logger = logging.getLogger('infer')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ('LAYOUT_PARSER', 'LINE_CROPPER', 'OCR', 'DECODER')
INPUT = {'OCR': 'LINE_CROPPER', 'DECODER': 'OCR'}  # the stage whose results an engine of this stage runs on
RUN_SWITCH = {'LAYOUT_PARSER': 'RUN_LAYOUT_PARSER', 'LINE_CROPPER': 'RUN_LINE_CROPPER', 'OCR': 'RUN_OCR',
              'DECODER': 'RUN_DECODER'}
THRESHOLD = 'FILTER_CONFIDENT_LINES_THRESHOLD'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp', '.jp2')
LMDB_MAP_SIZE = 1 << 40
# A unit adds its computed pages to its folder every CHUNK_PAGES pages (--chunk-pages). 25 pages are under 10 minutes
# for the slowest units of the eb446a3 run (decoders, ~22 s a page) and half a minute for a median unit (~0.7 s a page).
CHUNK_PAGES = 25
KEEP_TORCHSCRIPT_FUSER = False  # --keep-torchscript-fuser
FORCE_ENGINE = False  # --force-model
RUN = {}  # where this process runs (host, commit): noted with a forced addition (forced.json)
BIG_MODEL_PARAMETERS = 150e6  # an OCR model with more parameters is big (--big-jobs): the mega transformer (261 M);
#                               the other transformers have ~110 M, the CTC models ~17 M


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


def seed_page(name):
    """Seed every random generator pero-ocr uses from the page name (issue #100)."""
    np.random.seed(page_seed(name))
    random.seed(page_seed(name))


def in_logits_subset(key):
    """One line in records.LOGPROBS_SUBSET keeps its log-probabilities, chosen by a hash of its key (the same lines in
    every unit and every run with the same subset)."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % records.LOGPROBS_SUBSET == 0


class Dataset:
    """<folder>/images/<page>.<ext> and <folder>/page_xml/<page>.xml; a page is {'name', 'image', 'page_xml'}."""

    def __init__(self, folder):
        self.folder = folder
        self.name = os.path.basename(os.path.normpath(folder))
        images, xmls = os.path.join(folder, 'images'), os.path.join(folder, 'page_xml')
        if not os.path.isdir(images) or not os.path.isdir(xmls):
            sys.exit(f'{folder}: a benchmark needs images/ and page_xml/')
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


def registry_benchmark(path):
    """The benchmark a registry runs on: its BENCHMARK, the name of a folder of --data."""
    try:
        import yaml
    except ImportError:
        sys.exit('reading the registry needs PyYAML (pip install pyyaml)')
    if not os.path.isfile(path):
        sys.exit(f'no registry {path}')
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    name = data.get('BENCHMARK') if isinstance(data, dict) else None
    if not isinstance(name, str) or not name or os.sep in name or name.startswith('.'):
        sys.exit(f'{path}: BENCHMARK must name the benchmark it runs on (a folder of --data), got {name!r}')
    return name


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
    """A registry (any .yaml file, given with --registry): BENCHMARK, the benchmark it runs on (a folder of --data),
    and for every stage (LAYOUT_PARSER, LINE_CROPPER, OCR, DECODER) its engines, by name; an
    engine's folder is <STAGE>/<name>/ in the engines folder. An engine is a list item, or a key with what it runs on:
        LINE_CROPPER: <names>    OCR engines: the croppers whose line crops it reads (required)
        OCR: <names>             DECODER engines: the OCR runs whose output it decodes (required); <ocr>, or
                                 <ocr>@<cropper> for an OCR engine listed with several croppers
    <names> is one name or a list. Any subset of the stages is valid.

    Page lists: a run is on every page of the benchmark, or on the pages of a page list given with its name: a layout
    engine or cropper as "<name>: <page list>", an OCR or decoder run as "<input>: <page list>" in its LINE_CROPPER or
    OCR. A page list is a file relative to the benchmark folder, one page name per line (blank lines and # comments
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
        data.pop('BENCHMARK', None)  # read by registry_benchmark()
        unknown = set(data) - set(STAGES)
        if unknown:
            sys.exit(f'{path}: unknown stages {sorted(unknown)}; the stages are {", ".join(STAGES)} (and BENCHMARK)')
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
        """The pages of a page list file (relative to the benchmark folder)."""
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
                sys.exit(f'registry: {where}: page list {name}: {len(unknown)} pages are not in the benchmark: '
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
            for run in e.inputs:
                ocr = runs[run]
                if ocr.config()['OCR'].get('METHOD') == 'pytorch_ocr-transformer':
                    sys.exit(f'registry: DECODER: {e.name}: OCR {ocr.name} is a transformer; a decoder needs CTC '
                             f'logits')

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


def current_commit():
    """The commit of the checkout, '+' when it has uncommitted changes (None outside a git checkout)."""
    try:
        return git('rev-parse', 'HEAD') + ('+' if git('status', '--porcelain', '--untracked-files=no') else '')
    except (subprocess.CalledProcessError, OSError):
        return None


LOCKS = []  # (path, file) of the lock files held by this process (closing one releases its lock)


def lock_records(out):
    """Hold <benchmark records>/.lock for the rest of this process: two runs adding to the same records at once
    would write the same units. A second run stops with who holds it. The lock is the operating system's lock on the
    open file, released when the process ends in any way (a crash or kill leaves no stale lock); the file is deleted
    when the run ends normally (release_locks), a file left by a crash only says who ran last. After taking the lock,
    the file locked must still be the file at the path (another run may have deleted it meanwhile); otherwise take it
    again."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, '.lock')
    while True:
        f = open(path, 'a+')
        try:
            fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.seek(0)
            holder = f.read().strip() or 'unknown'
            f.close()
            sys.exit(f'{out}: another run is adding to these records ({holder}); wait for it or stop it')
        try:
            same = os.stat(path).st_ino == os.fstat(f.fileno()).st_ino
        except FileNotFoundError:
            same = False
        if same:
            break
        f.close()  # locked a file that was deleted meanwhile: take the lock of the current one
    f.seek(0)
    f.truncate()
    f.write(f'pid {os.getpid()} on {socket.gethostname()} since {utc_now()}: {shlex.join(sys.argv)}\n')
    f.flush()
    LOCKS.append((path, f))


def release_locks():
    """Delete the lock files and release the locks (at the normal end of a run)."""
    while LOCKS:
        path, f = LOCKS.pop()
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        f.close()


def record_free_run(out, force):
    """Append to <benchmark records>/run.json what a free run ran with: the commit, uncommitted changes, the command.
    Adding to a free run's records needs the code of its first run (the commit and the uncommitted changes), unless
    --force-commit (noted in forced.json)."""
    run = dict(started_utc=utc_now(), argv=sys.argv, python=sys.executable, cwd=os.getcwd(), host=RUN.get('hostname'))
    try:
        run['commit'] = git('rev-parse', 'HEAD')
        run['dirty'] = git('status', '--porcelain', '--untracked-files=no').splitlines()
        run['diff'] = git('diff', 'HEAD')
    except (subprocess.CalledProcessError, OSError):
        run['commit'] = None  # pero-ocr not in a git checkout
    path = os.path.join(out, 'run.json')
    runs = []
    if os.path.exists(path):
        with open(path) as f:
            runs = json.load(f)
    if runs:
        first = runs[0]
        def show(diff):  # uncommitted changes: their fingerprint and size, not the text
            return 'none' if not diff else f'md5 {hashlib.md5(diff.encode()).hexdigest()[:12]} ({diff.count(chr(10)) + 1} lines)'
        details = ([f"commit: {first.get('commit')} -> {run.get('commit')}"] if first.get('commit') != run.get('commit')
                   else []) + ([f"uncommitted changes: {show(first.get('diff'))} -> {show(run.get('diff'))}"]
                               if first.get('diff') != run.get('diff') else [])
        if details and not force:
            sys.exit(f'{out} was inferred with other code; adding to it would mix them (--force-commit to do it '
                     f'anyway):\n' + '\n'.join('  ' + d for d in details))
        if details:
            logger.warning(f'{out}: adding with other code (--force-commit): ' + '; '.join(details))
            note_forced(out, 'with other code', '--force-commit', details)
    with open(path, 'w') as f:
        json.dump(runs + [run], f, indent=1, ensure_ascii=False)


def environment():
    import importlib.metadata as metadata
    import torch
    env = {
        'hostname': socket.gethostname(),
        'gpu_driver': gpu_driver(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {d.metadata['Name'].lower(): d.version for d in metadata.distributions() if d.metadata['Name']},
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    env['packages'].pop('pero-ocr', None)  # the commit identifies pero-ocr
    return env


def gpu_driver():
    """The NVIDIA driver version (None without nvidia-smi)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], check=True,
                             capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0] if out else None


# The environment of a benchmark's records: the machine (hostname) and the software and GPU of its runs. Adding to the
# records needs the same, unless --force-environment. hostname and gpu_driver are compared only with records that have
# them (records made before they were recorded do not).
ENV_KEYS = ('hostname', 'gpu_driver', 'python', 'packages', 'torch_cuda', 'cudnn', 'gpu')
ENV_KEYS_ADDED_LATER = ('hostname', 'gpu_driver')


def record_differences(old, record, keys):
    differ = [k for k in keys if old.get(k) != record.get(k)
              and not (k in ENV_KEYS_ADDED_LATER and k not in old)]
    details = []
    for k in differ:
        a, b = old.get(k), record.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            details += [f'{k}: {p}: {a.get(p)} -> {b.get(p)}' for p in sorted(set(a) | set(b)) if a.get(p) != b.get(p)]
        else:
            details.append(f'{k}: {a} -> {b}')
    return details


def note_forced(out, what, flag, details):
    """<benchmark records>/forced.json: every guard a run broke through with its --force flag, and what differed."""
    path = os.path.join(out, 'forced.json')
    entries = []
    if os.path.exists(path):
        with open(path) as f:
            entries = json.load(f)
    entries.append(dict(utc=utc_now(), what=what, flag=flag, host=RUN.get('hostname'), commit=RUN.get('commit'),
                        argv=sys.argv, differences=details))
    with open(path, 'w') as f:
        json.dump(entries, f, indent=1, ensure_ascii=False)


def check_record(path, record, keys, what, flag, force):
    """Write the record, or require that an existing one has the same keys; with the force flag, add anyway and note
    it in forced.json (the record keeps its first values)."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(record, f, indent=1, sort_keys=True)
        return
    with open(path) as f:
        old = json.load(f)
    details = record_differences(old, record, keys)
    if not details:
        return
    if not force:
        sys.exit(f'{os.path.dirname(path)} was inferred {what}; adding to it would mix them ({flag} to do it anyway):\n'
                 + '\n'.join('  ' + d for d in details[:20]))
    logger.warning(f'{os.path.dirname(path)}: adding {what} ({flag}): ' + '; '.join(details[:5]))
    note_forced(os.path.dirname(path), what, flag, details)


# ----------------------------------------------------------------------------------------------------------------------
# Units


ENGINE_FIELDS = ('unit', 'engine', 'files', 'config')  # engine.json: what every unit records, the engine it ran


def read_unit(folder):
    """A unit's record: engine.json (unit, engine, files, config; a unit is complete when it has it) and unit.json (only
    adopted_from, engine_changes when they apply), as one dict; None for a unit that does not exist."""
    path = os.path.join(folder, 'engine.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        info = json.load(f)
    extra = os.path.join(folder, 'unit.json')
    if os.path.exists(extra):
        with open(extra) as f:
            info.update(json.load(f))
    return info


def write_unit(folder, info):
    """Write a unit's record (read_unit): unit.json first, engine.json last (its presence marks a complete unit)."""
    for name, part in (('unit.json', {k: v for k, v in info.items() if k not in ENGINE_FIELDS}),
                       ('engine.json', {k: info[k] for k in ENGINE_FIELDS})):
        path = os.path.join(folder, name)
        if name == 'unit.json' and not part:
            if os.path.exists(path):
                os.remove(path)
            continue
        tmp = os.path.join(folder, f'.{name}.tmp')
        with open(tmp, 'w') as f:
            json.dump(part, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)


def recorded_pages(folder, dataset):
    """The pages a unit holds (empty when it was never run). OCR and DECODER units: the pages whose line files are
    there (records.py); layout units: the pages whose page_xml/<page>.xml is there; line croppers: the pages of its
    pages.tsv (its crops are one LMDB, so the page's row says the page is in it)."""
    info = read_unit(folder)
    if info is None:
        return set()
    stage = os.path.basename(os.path.dirname(os.path.normpath(folder)))
    if stage in ('OCR', 'DECODER'):
        return records.held_pages(folder)
    if stage == 'LAYOUT_PARSER':
        xml = os.path.join(folder, 'page_xml')
        return {f[:-4] for f in os.listdir(xml) if f.endswith('.xml') and not f.startswith('.')} if os.path.isdir(
            xml) else set()
    return set(read_page_stats(folder))


PAGE_STATS = 'pages.tsv'  # per unit: the processing of every page, the columns of its stage (PAGE_COLUMNS)
PAGE_COLUMNS = {
    'LAYOUT_PARSER': ('page', 'seconds', 'gpu_peak_mb', 'lines'),             # lines: detected
    'LINE_CROPPER': ('page', 'seconds', 'crops'),                            # on the CPU
    'OCR': ('page', 'seconds', 'gpu_peak_mb', 'lines', 'chars', 'width_px'),  # lines read, chars written, crop width
    'DECODER': ('page', 'seconds', 'gpu_peak_mb', 'lines_decoded', 'chars', 'frames'),  # frames: of the decoded lines
}
CROPS = 'crops.png.lmdb'  # a cropper's line crops, PNG images


def measured(fn, *args):
    """Run fn(*args) and return (its result, wall seconds, peak GPU memory in MB or '' without a GPU): the processing
    of one page by an engine, without reading its input or writing its output."""
    import torch
    gpu = torch.cuda.is_available() and torch.cuda.is_initialized()
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = fn(*args)
    if gpu or (torch.cuda.is_available() and torch.cuda.is_initialized()):
        torch.cuda.synchronize()
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    else:
        peak = ''
    return result, round(time.perf_counter() - t0, 3), peak


def read_page_stats(folder):
    path = os.path.join(folder, PAGE_STATS)
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        return {r['page']: r for r in csv.DictReader(f, delimiter='\t')}


def write_page_stats(folder, rows, order, stage):
    """<folder>/pages.tsv: {page: row}, in the benchmark's page order, the columns of the stage."""
    tmp = os.path.join(folder, '.' + PAGE_STATS + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(PAGE_COLUMNS[stage]), delimiter='\t',
                           lineterminator='\n')
        w.writeheader()
        for page in sorted(rows, key=lambda p: order.get(p, len(order))):
            w.writerow({k: rows[page].get(k, '') for k in w.fieldnames})
    os.replace(tmp, os.path.join(folder, PAGE_STATS))


class Unit:
    """A unit's folder holds the pages it was run on: one set of files per page (OCR, DECODER: records.py; layout:
    page_xml/<page>.xml), the line crops of a cropper in one LMDB (its pages: pages.tsv). A run computes the pages the
    registry gives it that the unit does not hold, a chunk at a time, into <unit>.partial/ and then moves them into
    the unit (a top-up), or, for a new unit, renames the folder. Deleting a page's files makes the next run compute
    the page again."""

    def __init__(self, stage, name, out, dataset, wanted):
        self.stage, self.name, self.dataset = stage, name, dataset
        self.wanted = wanted  # the pages the registry gives it (of the pages processed, --pages)
        self.final = os.path.join(out, stage, name)
        self.partial = self.final + '.partial'

    @property
    def id(self):
        return f'{self.stage}/{self.name}'

    def info(self):
        return read_unit(self.final)

    def recorded(self):
        return recorded_pages(self.final, self.dataset)

    def missing_pages(self):
        """The pages to compute, in the benchmark's order."""
        have = self.recorded()
        return [p for p in self.dataset.all_pages if p['name'] in self.wanted and p['name'] not in have]

    def done(self):
        return not self.missing_pages()

    def check_engine(self, files, config):
        """A top-up only with the engine the unit was recorded with, unless --force-model: then the change is recorded
        in unit.json ("engine_changes") when the pages are added."""
        info = self.info()
        self.engine_change = None
        if info is None or (info.get('files') == files and info.get('config') == config):
            return
        if not FORCE_ENGINE:
            sys.exit(f'{self.id}: the engine (its files or config) is not the one the unit was recorded with; adding '
                     f'pages would mix two engines: delete {self.final} to run it again (--force-model to add them '
                     f'anyway)')
        old_files, old_config = info.get('files') or {}, info.get('config')
        change = {'files_before': {f: old_files.get(f) for f in sorted(set(old_files) | set(files))
                                   if old_files.get(f) != files.get(f)}}
        if old_config != config:
            change['config_before'] = old_config
        logger.warning(f'{self.id}: adding pages with another engine than the unit was recorded with (--force-model): '
                       + ', '.join(change['files_before']) + (' config' if 'config_before' in change else ''))
        change['started_utc'] = utc_now()
        self.engine_change = change

    def start(self, pages, replace=False):
        """Compute <pages> (page dicts); replace: compute the unit anew instead of adding to it."""
        self.pages = pages
        self.topup = not replace and self.info() is not None
        shutil.rmtree(self.partial, ignore_errors=True)
        os.makedirs(self.partial)
        self.started_utc = utc_now()  # identifies the run in engine_changes
        self.stats = {}
        if self.topup:
            logger.info(f'{self.id}: adding {len(pages)} pages to the {len(self.recorded())} it holds')

    def page_done(self, name, seconds, **counts):
        """The processing of one page: the wall seconds of the engine's call and what its stage counts
        (PAGE_COLUMNS): peak GPU memory in MB, lines detected (layout), crops (cropper), lines read, characters
        written (as the engine wrote them, before any NFC of the output) and the total width of the crops (OCR), lines
        decoded (the lines over the decoder's confidence threshold keep the OCR text), characters written and logit
        frames of those lines (decoder)."""
        row = dict(page=name, seconds=seconds, **counts)
        if set(row) != set(PAGE_COLUMNS[self.stage]):
            raise ValueError(f'{self.id}: page columns {sorted(row)}, expected {PAGE_COLUMNS[self.stage]}')
        self.stats[name] = row

    def finish(self, info):
        """Add the chunk computed into <unit>.partial/ to the unit. engine.json holds the engine, unit.json (when it
        applies) adopted_from and engine_changes; the pages are the files
        (a cropper's: the rows of pages.tsv), their processing (seconds, lines, GPU memory) is pages.tsv."""
        new = [p['name'] for p in self.pages]
        order = {p['name']: i for i, p in enumerate(self.dataset.all_pages)}
        if not self.topup:
            if self.stats:
                write_page_stats(self.partial, self.stats, order, self.stage)
            write_unit(self.partial, dict(unit=self.id, **info))
            shutil.rmtree(self.final, ignore_errors=True)
            os.rename(self.partial, self.final)
        else:
            old = self.info()
            merge_unit(self.partial, self.final)
            if self.stats:  # before the record: a page's row may stay if interrupted here, the next run replaces it
                write_page_stats(self.final, dict(read_page_stats(self.final), **self.stats), order, self.stage)
            merged = dict(old, **info)
            change = getattr(self, 'engine_change', None)
            if change:  # --force-model: the pages computed with the new engine, one entry per run
                changes = old.get('engine_changes', [])
                if changes and changes[-1].get('started_utc') == change['started_utc']:
                    changes[-1]['pages'] = changes[-1]['pages'] + new
                else:
                    changes = changes + [dict(change, pages=new)]
                merged['engine_changes'] = changes
            write_unit(self.final, merged)  # the unit holds the new pages from here on
            shutil.rmtree(self.partial)
        held = len(self.recorded() & set(self.wanted))
        seconds = sum(r['seconds'] for r in self.stats.values())
        logger.info(f'{self.id}: {len(new)} pages saved ({seconds:.0f} s processing), holds {held} of its '
                    f'{len(self.wanted)} pages' + (' (done)' if held == len(self.wanted) else ''))


def merge_unit(partial, final):
    """Move the pages computed into <partial> into the unit in <final> (a page's files replace the ones there). Safe to
    repeat after an interruption."""
    for sub in ('page_xml', 'logits', records.LINES, records.LOGPROBS):
        src = os.path.join(partial, sub)
        if os.path.isdir(src):
            os.makedirs(os.path.join(final, sub), exist_ok=True)
            for f in os.listdir(src):
                os.replace(os.path.join(src, f), os.path.join(final, sub, f))
    if os.path.isdir(os.path.join(partial, CROPS)):
        import lmdb
        src = lmdb.open(os.path.join(partial, CROPS), subdir=True, readonly=True, lock=False)
        dst = lmdb.open(os.path.join(final, CROPS), map_size=LMDB_MAP_SIZE, subdir=True)
        with src.begin() as r, dst.begin(write=True) as w:
            for key, value in r.cursor():
                w.put(key, value)
        src.close()
        dst.close()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


STORAGE_BYTES = {'FloatStorage': 4, 'BFloat16Storage': 2, 'HalfStorage': 2, 'DoubleStorage': 8}


def model_parameters(folder):
    """The number of parameters of the torch models (zip archives) in an engine folder, from the sizes of their
    tensors and their storage type, without loading them."""
    n = 0
    for f in sorted(os.listdir(folder)):
        path = os.path.join(folder, f)
        if not os.path.isfile(path) or not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as z:
            pickles = [i for i in z.namelist() if i.endswith('/data.pkl') and i.count('/') == 1]
            if not pickles:
                continue
            types = set(re.findall(rb'(\w+Storage)', z.read(pickles[0]))) - {b'LongStorage'}
            nbytes = sum(i.file_size for i in z.infolist() if '/data/' in i.filename)
        width = max((STORAGE_BYTES.get(t.decode(), 4) for t in types), default=4)
        n += nbytes // width
    return n


def chunks(items):
    """<items> in chunks of CHUNK_PAGES (the caller adds each chunk to its unit before taking the next, so an
    interruption loses at most one chunk)."""
    for i in range(0, len(items), CHUNK_PAGES):
        yield items[i:i + CHUNK_PAGES]


def unit_engine(registry, unit):
    """The engine whose files a unit records: the stage's engine, for OCR the run's engine, for a decoder the LM."""
    if unit.stage == 'OCR':
        return registry.ocr_engine(unit.name)
    return registry.engines[unit.stage][unit.name.split('@', 1)[0]]


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
        for page, image, _ in read_pages(pages, with_layout=False):
            seed_page(page['name'])
            layout = PageLayout(id=page['name'], page_size=(image.shape[0], image.shape[1]))
            layout, seconds, gpu = measured(parser.process_page, image, layout)
            layout.to_pagexml(os.path.join(unit.partial, 'page_xml', page['name'] + '.xml'))
            unit.page_done(page['name'], seconds, gpu_peak_mb=gpu, lines=sum(1 for _ in layout.lines_iterator()))
        unit.finish(dict(engine=unit.name, files=engine.files(), config=config_text(config)))
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
        seed_page(page['name'])
        layout = PageLayout(file=page['page_xml'])
        t0 = time.perf_counter()
        layout = _croppers[name].process_page(image, layout)
        seconds = round(time.perf_counter() - t0, 3)
        items = [(f"{page['name']}/{line.id}", cv2.imencode('.png', line.crop)[1].tobytes())
                 for line in layout.lines_iterator() if line.crop is not None]
        result[name] = (items, seconds)
    return page['name'], result


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
            envs = {}
            for u in active:
                u.start([page for page, cropper_configs in chunk if u.name in cropper_configs])
                envs[u.name] = lmdb.open(os.path.join(u.partial, CROPS), map_size=LMDB_MAP_SIZE, subdir=True)
            by_name = {u.name: u for u in active}
            for page_name, result in pool.imap(_crop_page, chunk, chunksize=2):
                for name, (items, seconds) in result.items():
                    with envs[name].begin(write=True) as txn:
                        for key, png in items:
                            txn.put(key.encode(), png)
                    by_name[name].page_done(page_name, seconds, crops=len(items))
            for u in active:
                envs[u.name].close()
                u.finish(dict(engine=u.name, files=engines[u.name].files(), config=configs[u.name][0]))


# ----------------------------------------------------------------------------------------------------------------------
# OCR and decoders


def page_rows(line_ids, layout):
    """(line_id, text, confidence, kept) of every line of the input (records.write_page); kept '0' for the lines the
    PAGE_PARSER filter removed (no text, no confidence)."""
    lines = {line.id: line for line in layout.lines_iterator()}
    rows = []
    for line_id in line_ids:
        line = lines.get(line_id)
        if line is None:
            rows.append((line_id, '', '', '0'))
        else:
            text = (line.transcription or '').replace('\t', ' ').replace('\n', ' ')
            conf = '' if line.transcription_confidence is None else repr(float(line.transcription_confidence))
            rows.append((line_id, text, conf, '1'))
    return rows


def crops_env(out, cropper):
    import lmdb
    folder = os.path.join(out, 'LINE_CROPPER', cropper)
    if not os.path.exists(os.path.join(folder, 'engine.json')):
        sys.exit(f'LINE_CROPPER/{cropper} is not done: the OCR runs on it read their line crops from it')
    return lmdb.open(os.path.join(folder, CROPS), subdir=True, readonly=True, lock=False)


def ocr_parser(registry, run):
    from pero_ocr.document_ocr.page_parser import PageParser
    engine = registry.ocr_engine(run)
    config = engine.config()
    return PageParser(config, config_path=engine.folder), config


def check_logprobs_subset(folder, what):
    """A unit keeps the log-probabilities of one subset: refuse to add to one recorded with another
    (--logprobs-subset)."""
    other = sorted(f for f in os.listdir(folder) if f.startswith('logprobs.') and f != records.LOGPROBS) \
        if os.path.isdir(folder) else []
    if other:
        sys.exit(f'{what} keeps {other[0]}/, this run {records.LOGPROBS}/ (--logprobs-subset); run it with the same '
                 f'subset, or into another folder')


def run_ocr(unit, registry, dataset, out):
    """The OCR engine on the stored line crops of its cropper."""
    from pero_ocr.document_ocr.page_parser import prepare_dense_logits
    engine = registry.ocr_engine(unit.name)
    unit.check_engine(engine.files(), config_text(engine.config()))
    check_logprobs_subset(unit.final, unit.id)
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
        for page, image, layout in read_pages(chunk, crops=env):
            name = page['name']
            seed_page(name)
            line_ids = [line.id for line in layout.lines_iterator()]
            for line in layout.lines_iterator():
                line.transcription = None
            layout, seconds, gpu = measured(parser.process_page, image, layout)
            read = [line for line in layout.lines_iterator() if line.crop is not None]
            unit.page_done(name, seconds, gpu_peak_mb=gpu, lines=len(read),
                           chars=sum(len(line.transcription or '') for line in read),
                           width_px=sum(line.crop.shape[1] for line in read))
            rows = page_rows(line_ids, layout)
            logprobs = {line.id: prepare_dense_logits(line).astype(np.float32) for line in layout.lines_iterator()
                        if line.logits is not None and in_logits_subset(f'{name}/{line.id}')}
            if name in kept:
                layout.to_pagexml(os.path.join(unit.partial, 'page_xml', name + '.xml'))
                with open(os.path.join(unit.partial, 'logits', name + '.logits'), 'wb') as f:
                    f.write(layout.save_logits_bytes(missing_line_logits_ok=True))
            records.write_logprobs(unit.partial, name, logprobs)
            records.write_page(unit.partial, name, rows)
        unit.finish(dict(engine=engine.name, files=engine.files(), config=config_text(config)))
    del parser
    env.close()


def run_derived(unit, registry, dataset, out):
    """An OCR run that differs from its base only in a higher FILTER_CONFIDENT_LINES_THRESHOLD: the base's lines,
    with the lines the filter removes (confidence <= threshold, as PageParser filters) marked kept=0. Cheap, so it is
    derived anew on all its pages (those it holds and those it gets) whenever it lacks pages: after pages of the base
    were computed again, delete them here too."""
    base = registry.base(unit.name)
    base_folder = os.path.join(out, 'OCR', base)
    threshold = registry.threshold(unit.name)
    pages = unit.recorded() | set(unit.wanted)
    page_dicts = [p for p in dataset.all_pages if p['name'] in pages]
    need_pages(f'OCR/{base}', out, dataset, page_dicts)
    check_logprobs_subset(base_folder, f'OCR/{base}')
    unit.start(page_dicts, replace=True)
    for page in page_dicts:
        name = page['name']
        rows, removed = [], set()
        for line_id, text, conf, kept in records.read_page(base_folder, name):
            if kept == '1' and conf != '' and float(conf) <= threshold:
                text, conf, kept = '', '', '0'
                removed.add(line_id)
            rows.append((line_id, text, conf, kept))
        logprobs = {k: v for k, v in records.read_logprobs(base_folder, name).items() if k not in removed}
        records.write_logprobs(unit.partial, name, logprobs)
        records.write_page(unit.partial, name, rows)
    engine = registry.ocr_engine(unit.name)
    unit.finish(dict(engine=engine.name, files=engine.files(), config=config_text(engine.config())))  # no pages.tsv


def ocr_output(run, pages, registry, dataset, out):
    """The folder with the stored output (page_xml/, logits/) of an OCR run, for <pages>. Pages whose output is not
    stored (pruned, or no decoder decoded the page when it ran) are run again; their lines must equal the unit's
    recorded lines (same code, same seeds), otherwise the records do not belong together."""
    folder = os.path.join(out, 'OCR', run)
    need_pages(f'OCR/{run}', out, dataset, pages)
    missing = [p for p in pages if not os.path.exists(os.path.join(folder, 'logits', p['name'] + '.logits'))]
    if not missing:
        return folder
    logger.info(f'OCR/{run}: output of {len(missing)} pages is not stored, running the OCR engine again')
    parser, _ = ocr_parser(registry, run)
    env = crops_env(out, run.split('@', 1)[1])
    for sub in ('page_xml', 'logits'):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    for page, image, layout in read_pages(missing, crops=env):
        name = page['name']
        seed_page(name)
        line_ids = [line.id for line in layout.lines_iterator()]
        for line in layout.lines_iterator():
            line.transcription = None
        layout = parser.process_page(image, layout)
        if page_rows(line_ids, layout) != records.read_page(folder, name):
            sys.exit(f'OCR/{run}: page {name} run again differs from its recorded lines; the OCR unit was recorded '
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
    """The decoder on the stored output of the OCR run (PAGE XML and logits, as a production worker stores them
    for the LM stage); for a run derived by a higher threshold, without the lines that threshold removes."""
    from pero_ocr.core.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import PageParser
    lm, run = unit.name.split('@', 1)
    engine = registry.engines['DECODER'][lm]
    ocr_unit = os.path.join(out, 'OCR', run)
    config = engine.config()
    unit.check_engine(engine.files(), config_text(config))
    charset = decoder_charset(registry.engines['OCR'][run.split('@', 1)[0]])  # the OCR engine's json
    pages = unit.missing_pages()
    need_pages(f'OCR/{run}', out, dataset, pages)
    base = registry.base(run)
    ocr_folder = ocr_output(base, pages, registry, dataset, out)
    run_config = engine.config()
    if not run_config.has_section('OCR'):
        run_config.add_section('OCR')
    run_config['OCR']['OCR_JSON'] = charset
    decoder = PageParser(run_config, config_path=engine.folder)
    for chunk in chunks(pages):
        unit.start(chunk)
        for page in chunk:
            name = page['name']
            seed_page(name)
            line_ids = [line.id for line in PageLayout(file=page['page_xml']).lines_iterator()]
            layout = PageLayout(file=os.path.join(ocr_folder, 'page_xml', name + '.xml'))
            if base != run:  # the lines the run's higher threshold removes
                removed = {line_id for line_id, _, _, kept in records.read_page(ocr_unit, name) if kept == '0'}
                for region in layout.regions:
                    region.lines = [line for line in region.lines if line.id not in removed]
            layout.load_logits(os.path.join(ocr_folder, 'logits', name + '.logits'))
            to_decode = lines_to_decode(decoder.decoder, layout)
            decoded = decoder.decoder.lines_decoded
            layout, seconds, gpu = measured(decoder.process_page, None, layout)
            n = decoder.decoder.lines_decoded - decoded
            if n != len(to_decode):
                logger.warning(f'{unit.id}: page {name}: {n} lines decoded, {len(to_decode)} expected')
            unit.page_done(name, seconds, gpu_peak_mb=gpu, lines_decoded=n,
                           chars=sum(len(line.transcription or '') for line in layout.lines_iterator()
                                     if line.id in to_decode), frames=sum(to_decode.values()))
            rows = page_rows(line_ids, layout)
            records.write_page(unit.partial, name, rows)
        unit.finish(dict(engine=lm, files=engine.files(), config=config_text(config)))


def lines_to_decode(page_decoder, layout):
    """{line id: logit frames} of the lines a PageDecoder will decode: its categories, with logits, and not confident
    enough to keep the OCR text (pero-ocr's own test, line_confident_enough, with its threshold); the frames are those
    it decodes (prepare_dense_logits: every frame, the padding included)."""
    from pero_ocr.document_ocr.page_parser import line_confident_enough, prepare_dense_logits
    threshold = page_decoder.line_confidence_threshold
    out = {}
    for line in layout.lines_iterator(page_decoder.categories):
        if line.logits is None:
            continue
        logits = prepare_dense_logits(line)
        if threshold is None or not line_confident_enough(logits, threshold):
            out[line.id] = logits.shape[0]
    return out


def decoder_charset(ocr_engine):
    """The characters a decoder reads the logits of an OCR engine with: the OCR engine's own (its [OCR] OCR_JSON), as
    for a decoder in the same config as the OCR. The decoder's own [OCR] OCR_JSON is not used: a decoder can decode
    OCR engines with different char sets, and its config keeps its own section for the record."""
    return os.path.abspath(os.path.join(ocr_engine.folder, ocr_engine.config()['OCR']['OCR_JSON']))


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


def write_report(out, commit):
    """<benchmark records>/report.md: the units of the benchmark."""
    rows = []
    for stage in STAGES:
        folder = os.path.join(out, stage)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            info = read_unit(os.path.join(folder, name))
            if info is not None:
                pages = sorted(recorded_pages(os.path.join(folder, name), None))
                stats = list(read_page_stats(os.path.join(folder, name)).values())
                seconds = sum(float(r['seconds']) for r in stats if r.get('seconds')) if stats else ''
                gpu = max((int(r['gpu_peak_mb']) for r in stats if r.get('gpu_peak_mb')), default='')
                counted = [c for c in PAGE_COLUMNS[stage] if c not in ('page', 'seconds', 'gpu_peak_mb')]
                processed = ', '.join(f"{sum(int(r[c]) for r in stats if r.get(c))} {c.replace('_', ' ')}" for c in counted
                                      if any(r.get(c) for r in stats))
                if info.get('adopted_from'):
                    name += f" (adopted from {info['adopted_from'][:7]})"
                rows.append((stage, name, len(pages) if isinstance(pages, list) else pages, seconds, gpu, processed))
    with open(os.path.join(out, 'environment.json')) as f:
        env = json.load(f)
    lines = [f'# Inference {commit}: {os.path.basename(os.path.normpath(out))}', '',
             'Processing per unit from its pages.tsv (lines a decoder decoded; empty: not recorded, e.g. derived or '
             'adopted from older records).', '',
             f"Python {env['python']}, torch {env['packages'].get('torch')} (CUDA {env['torch_cuda']}), "
             f"numpy {env['packages'].get('numpy')}, opencv {env['packages'].get('opencv-python')}, GPU {env['gpu']}.",
             '', '| stage | unit | pages | seconds | peak GPU MB | processed |', '|---|---|---|---|---|---|']
    lines += [f"| {k} | {n} | {p} | {f'{s:.0f}' if s != '' else ''} | {g} | {pr} |" for k, n, p, s, g, pr in rows]
    if any('(adopted from' in r[1] for r in rows):
        lines += ['', 'Adopted units were computed by the commit they name and copied here, not run by this commit '
                      '(adopted.json: the source and the files the code changed in between).']
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------------------------------------------------
# Adopting units of another commit


def adopt_source(value, results):
    """The records to adopt from: a free run's folder (holding <benchmark>/ folders), or a commit with records in
    --results. Returns (folder, commit or None); the commit of a free run is the one its run.json names, per
    benchmark (adopt_units)."""
    if os.path.isdir(value):
        return value, None
    try:
        commit = git('rev-parse', '--verify', value + '^{commit}')
    except subprocess.CalledProcessError:
        sys.exit(f'--adopt-from {value}: neither a folder of records nor a commit')
    source = os.path.join(results or '', commit)
    if not os.path.isdir(source):
        sys.exit(f'--adopt-from {value}: no records of {commit} in {results}')
    return source, commit


def adopt_units(source, source_commit, work, prefixes, force):
    """Copy the complete units matching <prefixes> from <source>/<benchmark>/ into the records being made, where they
    are not there yet. The source must have been inferred in this benchmark's environment and on the same pages.
    Copies, not links: a later top-up of an adopted unit must not change the source."""
    for dataset, registry, out, _, all_units in work:
        src = os.path.join(source, dataset.name)
        if not os.path.isdir(src):
            logger.info(f'{dataset.name}: no records in {source}; nothing adopted')
            continue
        with open(os.path.join(src, 'environment.json')) as f:
            source_env = json.load(f)
        with open(os.path.join(out, 'environment.json')) as f:
            env = json.load(f)
        details = record_differences(source_env, env, ENV_KEYS)
        if details and not force:
            sys.exit(f'--adopt-from {src}: inferred on another machine or in another environment; its units would not '
                     f'be what this one produces (--force-environment to adopt them anyway):\n'
                     + '\n'.join('  ' + d for d in details[:20]))
        if details:
            note_forced(out, f'units adopted from {src}, made on another machine or in another environment',
                        '--force-environment', details)
        record = dict(pages=len(dataset.all_pages), files=dataset.fingerprint())
        with open(os.path.join(src, 'benchmark.json')) as f:
            source_record = json.load(f)
        if any(source_record.get(k) != v for k, v in record.items()):
            sys.exit(f'--adopt-from {src}: {dataset.name} was inferred from other pages')
        commit = source_commit
        if commit is None and os.path.exists(os.path.join(src, 'run.json')):
            with open(os.path.join(src, 'run.json')) as f:
                commit = json.load(f)[-1].get('commit')
        adopted = []
        for unit in all_units.values():
            if not any(unit.id.startswith(prefix) for prefix in prefixes) or unit.info() is not None:
                continue
            folder = os.path.join(src, unit.stage, unit.name)
            if read_unit(folder) is None:
                logger.info(f'{dataset.name}: {unit.id} is not in {source}; it will be computed')
                continue
            tmp = unit.final + '.adopting'
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(folder, tmp)
            info = read_unit(tmp)
            info['adopted_from'] = commit or src
            write_unit(tmp, info)
            os.rename(tmp, unit.final)
            adopted.append(unit.id)
            logger.info(f'{dataset.name}: adopted {unit.id}')
        if not adopted:
            continue
        entry = dict(adopted_utc=utc_now(), source=src, source_commit=commit, units=adopted)
        if commit:
            try:
                entry['changed_files'] = git('diff', '--name-only', commit, 'HEAD').splitlines()
            except subprocess.CalledProcessError:
                pass
        path = os.path.join(out, 'adopted.json')
        entries = []
        if os.path.exists(path):
            with open(path) as f:
                entries = json.load(f)
        with open(path, 'w') as f:
            json.dump(entries + [entry], f, indent=1, ensure_ascii=False)


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


def run_parallel(todo, registry, dataset, out, args, folder_name, units):
    """Crops in this process (a CPU pool), every other unit in a process of its own at low CPU priority: args.jobs
    layout and OCR units on each GPU of args.gpus, args.decoder_jobs decoder units on args.decoder_gpus; a unit starts
    when the units it needs are done and enough memory is free, on the GPU of its kind running the fewest units."""
    crops = [u for u in todo if u.stage == 'LINE_CROPPER']
    lock, failed, finished = threading.Lock(), [], set()
    gpus, decoder_gpus = args.gpus or [None], args.decoder_gpus or args.gpus or [None]
    running = {g: 0 for g in gpus}  # layout and OCR units on each GPU
    running_big = {g: 0 for g in gpus}
    running_decoders = {g: 0 for g in decoder_gpus}

    def big(unit):
        """An OCR unit whose model has more than BIG_MODEL_PARAMETERS parameters: the heaviest GPU load. A derived
        run computes nothing."""
        if unit.stage != 'OCR' or registry.base(unit.name) != unit.name:
            return False
        return model_parameters(registry.ocr_engine(unit.name).folder) > BIG_MODEL_PARAMETERS

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
            '--registry', registry.path, '--crop-workers', '0', '--decoder-jobs', '1'] + \
           (['--out', args.out] if args.out else ['--results', args.results]) + \
           (['--pages', str(args.pages)] if args.pages else []) + \
           (['--force-environment'] if args.force_environment else []) + \
           (['--force-commit'] if args.force_commit else []) + \
           (['--force-model'] if args.force_model else []) + ['--chunk-pages', str(args.chunk_pages)] + \
           ['--logprobs-subset', str(args.logprobs_subset)] + \
           (['--keep-torchscript-fuser'] if args.keep_torchscript_fuser else [])

    def done(uid):
        """The unit holds the pages the registry gives it."""
        return units[uid].done()

    def gpu_for(unit):
        """The GPU the unit can start on now (None: the GPU the process sees by default), or False when there is
        none."""
        if unit.stage == 'DECODER':
            if sum(running_decoders.values()) >= args.decoder_jobs:
                return False
            return min(decoder_gpus, key=lambda g: running_decoders[g])
        free = [g for g in gpus if running[g] < args.jobs and not (big(unit) and running_big[g] >= args.big_jobs)]
        return min(free, key=lambda g: running[g]) if free else False

    def next_unit():
        """The first pending unit that can start and its GPU, 'wait', or None when nothing is left."""
        for i, unit in enumerate(pending):
            gpu = gpu_for(unit)
            if gpu is False:
                continue
            deps = needs(unit, registry)
            if all(done(d) for d in deps):
                return pending.pop(i), gpu
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
                    unit, gpu = unit
                    while mem_available_gb() < args.min_free_gb:
                        time.sleep(30)
                    name = unit.id.replace('/', '__')
                    logger.info(f'start {dataset.name}: {unit.id}' + ('' if gpu is None else f' on GPU {gpu}'))
                    # one log per unit, every attempt appended (a resumed unit keeps the log of the attempt that
                    # crashed or was stopped), each between a header and a footer line
                    cmd = base + ['--units', unit.id]
                    log = open(os.path.join(out, 'logs', name + '.log'), 'a')
                    log.write(f'===== {utc_now()} start: holds {len(unit.recorded() & unit.wanted)} of its '
                              f'{len(unit.wanted)} pages; GPU {gpu}; {shlex.join(cmd)}\n')
                    log.flush()
                    env = None if gpu is None else dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            env=env, preexec_fn=lambda: os.nice(10))
                    if unit.stage == 'DECODER':
                        running_decoders[gpu] += 1
                    else:
                        running[gpu] += 1
                        running_big[gpu] += big(unit)
                    time.sleep(args.stagger)  # let it load its models before the memory check of the next one
            if unit == 'wait':
                time.sleep(10)
                continue
            code = proc.wait()
            log.write(f'===== {utc_now()} end: exit code {code}\n')
            log.close()
            with lock:
                if unit.stage == 'DECODER':
                    running_decoders[gpu] -= 1
                else:
                    running[gpu] -= 1
                    running_big[gpu] -= big(unit)
                finished.add(unit.id)
                if code != 0:
                    failed.append(unit.id)
                    logger.error(f'{dataset.name}: {unit.id} failed, see {dataset.name}/logs/{name}.log')
            write_report(out, folder_name)

    threads = [threading.Thread(target=worker) for _ in range(args.jobs * len(gpus) + args.decoder_jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if crop_thread:
        crop_thread.join()
    return [f'{dataset.name}: {u}' for u in failed]


def main():
    global CHUNK_PAGES, KEEP_TORCHSCRIPT_FUSER, FORCE_ENGINE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--registry', nargs='+', metavar='REGISTRY',
                   help='the registries to run (.yaml files; each names its benchmark, BENCHMARK)')
    p.add_argument('--data', default=os.environ.get('PERO_TEST_DATA'), help='the benchmarks: one folder each')
    p.add_argument('--models', default=os.environ.get('PERO_TEST_MODELS'))
    p.add_argument('--results', default=os.environ.get('PERO_INFERENCE'), help='the records of official runs')
    p.add_argument('--out', help='a free run into this folder: not bound to the commit (experiments)')
    p.add_argument('--units', nargs='*', default=[''], help='unit id prefixes to run, e.g. OCR/ DECODER/czech')
    p.add_argument('--exact', action='store_true', help='--units are exact unit ids; run them in this process (how the unit processes are started)')
    p.add_argument('--pages', type=int, help='only the first N pages of each benchmark (a trial; only with --out)')
    p.add_argument('--jobs', type=int, default=1,
                   help='layout and OCR units running at the same time on each GPU, each in its own process')
    p.add_argument('--decoder-jobs', type=int, default=10,
                   help='decoder units running at the same time, in all (about one CPU core each, and their LM keeps '
                        'a GPU busy: one GPU serves about two at full speed)')
    p.add_argument('--gpus', type=lambda s: s.split(','),
                   help='GPUs to spread the layout and OCR units over, e.g. 0,1,2 (CUDA_VISIBLE_DEVICES of each unit '
                        'process); by default every unit uses the GPU this process sees')
    p.add_argument('--decoder-gpus', type=lambda s: s.split(','),
                   help='GPUs to spread the decoder units over (default: --gpus)')
    p.add_argument('--crop-workers', type=int, default=3)
    p.add_argument('--logprobs-subset', type=int, default=records.LOGPROBS_SUBSET, metavar='N',
                   help='OCR units keep the full log-probabilities of one line in N (default 64, logprobs.1in64/); '
                        'another N only in a free run (--out), e.g. 1 for every line when debugging a model')
    p.add_argument('--chunk-pages', type=int, default=CHUNK_PAGES,
                   help='a unit adds its computed pages to its folder every this many pages')
    p.add_argument('--big-jobs', type=int, default=1,
                   help=f'at most this many big OCR units (models over {BIG_MODEL_PARAMETERS / 1e6:.0f} M parameters) at '
                        f'a time on each GPU, of --jobs: several at once overload the GPU')
    p.add_argument('--keep-torchscript-fuser', action='store_true',
                   help="leave TorchScript's fuser on (pero-ocr's default): only to add to records made with it on")
    p.add_argument('--min-free-gb', type=float, default=5.0, help='start a unit process only with this much free RAM')
    p.add_argument('--stagger', type=float, default=45.0, help='seconds between starting two unit processes')
    p.add_argument('--list', action='store_true', help='show the units and whether they are done, run nothing')
    p.add_argument('--prune-logits', action='store_true',
                   help='delete the stored logits of the OCR runs whose decoders are all done, run nothing')
    p.add_argument('--force-environment', action='store_true',
                   help='add to records made on another machine or in another environment (python, packages, CUDA, '
                        'GPU, driver) or from other pages; noted in forced.json')
    p.add_argument('--force-commit', action='store_true',
                   help='add to a free run\'s records with other code than its first run (commit or uncommitted '
                        'changes); noted in forced.json. Official records are always per commit')
    p.add_argument('--force-model', '--force-engine', dest='force_model', action='store_true',
                   help='add pages to a unit although its model (files or config) changed since it was recorded; '
                        'unit.json records the change ("engine_changes"). For development: such a unit mixes models')
    p.add_argument('--adopt-from', metavar='COMMIT_OR_FOLDER',
                   help='take the --adopt units over from the records of this commit (in --results) or folder instead '
                        'of computing them; only when the code change does not touch their stages')
    p.add_argument('--adopt', nargs='*', default=[], metavar='PREFIX',
                   help='unit id prefixes to adopt (with --adopt-from), e.g. LINE_CROPPER/ OCR/ DECODER/')
    args = p.parse_args()
    if bool(args.adopt_from) != bool(args.adopt):
        sys.exit('--adopt-from and --adopt go together')
    if args.adopt and args.exact:
        sys.exit('--adopt: not with --exact')
    if args.chunk_pages < 1:
        sys.exit('--chunk-pages must be at least 1')
    if args.logprobs_subset < 1:
        sys.exit('--logprobs-subset must be at least 1')
    if args.logprobs_subset != records.LOGPROBS_SUBSET and not args.out:
        sys.exit(f'--logprobs-subset: only in a free run (--out); official runs keep 1 line in '
                 f'{records.LOGPROBS_SUBSET}, so that the records of all commits compare')
    records.set_logprobs_subset(args.logprobs_subset)
    CHUNK_PAGES = args.chunk_pages
    KEEP_TORCHSCRIPT_FUSER = args.keep_torchscript_fuser
    FORCE_ENGINE = args.force_model
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    if not args.registry:
        sys.exit('--registry is required: the registries to run')
    for name in ('data', 'models') + (() if args.out else ('results',)):
        if not getattr(args, name):
            sys.exit(f'--{name} (or its environment variable) is required')
    if args.pages and not args.out:
        sys.exit('--pages makes incomplete records: only in a free run (--out)')
    if args.keep_torchscript_fuser and not args.out:
        sys.exit('--keep-torchscript-fuser: only in a free run (--out); official runs are made with the fuser off')

    if args.out:
        folder_name = os.path.basename(os.path.normpath(args.out))
        root = args.out
    else:
        folder_name = official_commit()
        root = os.path.join(args.results, folder_name)
    work = []
    for path in args.registry:
        name = registry_benchmark(path)
        if any(d.name == name for d, *_ in work):
            sys.exit(f'--registry: two registries for the benchmark {name}; one run takes one registry per benchmark')
        if not os.path.isdir(os.path.join(args.data, name)):
            sys.exit(f'{path}: no benchmark {name} in {args.data}')
        dataset = Dataset(os.path.join(args.data, name))
        if args.pages:
            dataset.pages = dataset.pages[:args.pages]
        registry = Registry(os.path.abspath(path), args.models, dataset)
        out = os.path.join(root, name)
        all_units = {u.id: u for u in plan(registry, out, dataset)}
        if args.exact:
            units = [u for u in all_units.values() if u.id in set(args.units)]
        else:
            units = [u for u in all_units.values() if any(u.id.startswith(prefix) for prefix in args.units)]
        work.append((dataset, registry, out, units, all_units))

    RUN.update(hostname=socket.gethostname(), commit=folder_name if not args.out else current_commit())
    if not args.list and not args.exact:  # unit processes run under the lock of the run that started them
        for _, _, out, _, _ in work:
            lock_records(out)
    if args.prune_logits:
        for dataset, registry, out, _, all_units in work:
            prune_logits(registry, out, all_units)
        return
    if args.list:
        for dataset, registry, out, units, all_units in work:
            print(f'{dataset.name} ({len(dataset.pages)} pages):')
            for u in units:
                notes = []
                info = u.info()
                if info is not None and info.get('files') != unit_engine(registry, u).files():
                    notes.append('ENGINE CHANGED since recorded: delete the unit to run it again, or its pages and '
                                 'run them with --force-model')
                if registry.unit_pages[u.id][1] is not None:
                    notes.append(f'pages of {registry.unit_pages[u.id][1]}')
                if u.stage == 'OCR' and registry.base(u.name) != u.name:
                    notes.append(f'derived from OCR/{registry.base(u.name)}')
                have, missing = len(u.recorded() & u.wanted), len(u.missing_pages())
                state = 'done   ' if not missing else 'missing' if not have else 'partial'
                count = f'{have}/{len(u.wanted)} pages'
                print(f'  {state} {u.id}  ({count}{"".join("; " + n for n in notes)})')
            for stage in STAGES:
                folder = os.path.join(out, stage)
                for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
                    uid = f'{stage}/{name}'
                    if (os.path.isdir(os.path.join(folder, name)) and not name.endswith('.partial')
                            and uid not in all_units and any(uid.startswith(prefix) for prefix in args.units)):
                        print(f'  orphan  {uid}  (not in the registry)')
        return
    if args.adopt:
        source, source_commit = adopt_source(args.adopt_from, args.results)
        for dataset, registry, out, _, _ in work:
            os.makedirs(out, exist_ok=True)
            check_record(os.path.join(out, 'environment.json'), environment(), ENV_KEYS,
                         'on another machine or in another environment', '--force-environment', args.force_environment)
            check_record(os.path.join(out, 'benchmark.json'), dict(pages=len(dataset.all_pages),
                                                                   files=dataset.fingerprint()),
                         ('pages', 'files'), 'from other pages', '--force-environment', args.force_environment)
        adopt_units(source, source_commit, work, args.adopt, args.force_environment)
        for _, _, out, _, _ in work:
            write_report(out, folder_name)
    if not any(not u.done() for _, _, _, units, _ in work for u in units):
        print('nothing to do')
        return
    failed = []
    for dataset, registry, out, units, all_units in work:
        todo = [u for u in units if not u.done()]
        if not todo:
            continue
        os.makedirs(out, exist_ok=True)
        check_record(os.path.join(out, 'environment.json'), environment(), ENV_KEYS,
                     'on another machine or in another environment', '--force-environment', args.force_environment)
        if not args.exact:  # a unit process: the run that started it checked the benchmark and wrote run.json
            if args.out:
                record_free_run(out, args.force_commit)
            check_record(os.path.join(out, 'benchmark.json'), dict(pages=len(dataset.all_pages),
                                                                   files=dataset.fingerprint()),
                         ('pages', 'files'), 'from other pages', '--force-environment', args.force_environment)
        logger.info(f'{out}: {len(todo)} units to run')
        if args.exact or (args.jobs <= 1 and len(args.gpus or []) <= 1 and len(args.decoder_gpus or []) <= 1):
            run_inline(todo, registry, dataset, out, args.crop_workers)
        else:
            failed += run_parallel(todo, registry, dataset, out, args, folder_name, all_units)
        write_report(out, folder_name)
    if failed:
        sys.exit(f'failed: {"; ".join(failed)}')
    logger.info('all units done')


if __name__ == '__main__':
    try:
        main()
    finally:
        release_locks()  # also after sys.exit (a failed unit, a refused run): the locks were this run's
