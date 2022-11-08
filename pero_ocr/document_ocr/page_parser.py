import numpy as np

import logging
from multiprocessing import Pool
import math
import time

import torch.cuda
from pero_ocr.utils import compose_path
from .layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import crop_engine as cropper
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
from pero_ocr.layout_engines.simple_region_engine import SimpleThresholdRegion
from pero_ocr.layout_engines.simple_baseline_engine import EngineLineDetectorSimple
from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine, LineFilterEngine
from pero_ocr.layout_engines.line_postprocessing_engine import PostprocessingEngine
from pero_ocr.layout_engines.naive_sorter import NaiveRegionSorter
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter
from pero_ocr.layout_engines.line_in_region_detector import detect_lines_in_region
from pero_ocr.layout_engines.baseline_refiner import refine_baseline
from pero_ocr.layout_engines import layout_helpers as helpers


logger = logging.getLogger(__name__)


def layout_parser_factory(config, device, config_path='', order=1):
    config = config['LAYOUT_PARSER_{}'.format(order)]
    if config['METHOD'] == 'REGION_WHOLE_PAGE':
        layout_parser = WholePageRegion(config, config_path=config_path)
    elif config['METHOD'] == 'REGION_SIMPLE_THRESHOLD':
        layout_parser = SimpleThresholdRegion(config, config_path=config_path)
    elif config['METHOD'] == 'LAYOUT_CNN':
        layout_parser = LayoutExtractor(config, device, config_path=config_path)
    elif config['METHOD'] == 'LINES_SIMPLE_THRESHOLD':
        layout_parser = TextlineExtractorSimple(config, config_path=config_path)
    elif config['METHOD'] == 'LINE_FILTER':
        layout_parser = LineFilter(config, device, config_path=config_path)
    elif config['METHOD'] == 'LINE_POSTPROCESSING':
        layout_parser = LinePostprocessor(config, config_path=config_path)
    elif config['METHOD'] == 'LAYOUT_POSTPROCESSING':
        layout_parser = LayoutPostprocessor(config, config_path=config_path)
    elif config['METHOD'] == 'REGION_SORTER_NAIVE':
        layout_parser = NaiveRegionSorter(config, config_path=config_path)
    elif config['METHOD'] == 'REGION_SORTER_SMART':
        layout_parser = SmartRegionSorter(config, config_path=config_path)
    else:
        raise ValueError('Unknown layout parser method: {}'.format(config['METHOD']))
    return layout_parser


def line_cropper_factory(config, config_path=''):
    config = config['LINE_CROPPER']
    return LineCropper(config, config_path=config_path)


def ocr_factory(config, device, config_path=''):
    config = config['OCR']
    return PageOCR(config, device, config_path=config_path)


def page_decoder_factory(config, device, config_path=''):
    from pero_ocr.decoding import decoding_itf
    ocr_chars = decoding_itf.get_ocr_charset(compose_path(config['OCR']['OCR_JSON'], config_path))

    use_cpu = config['DECODER'].getboolean('USE_CPU')
    device = device if not use_cpu else torch.device("cpu")

    decoder = decoding_itf.decoder_factory(config['DECODER'], ocr_chars, device, allow_no_decoder=False, config_path=config_path)
    confidence_threshold = config['DECODER'].getfloat('CONFIDENCE_THRESHOLD', fallback=math.inf)
    carry_h_over = config['DECODER'].getboolean('CARRY_H_OVER')
    return PageDecoder(decoder, line_confidence_threshold=confidence_threshold, carry_h_over=carry_h_over)


class MissingLogits(Exception):
    pass


def line_confident_enough(logits, confidence_threshold):
    log_probs = logits - np.logaddexp.reduce(logits, axis=1)[:, np.newaxis]
    best_probs = np.max(log_probs, axis=-1)
    worst_best_prob = np.exp(np.min(best_probs))

    return worst_best_prob > confidence_threshold


def prepare_dense_logits(line):
    if line.logits is None:
        raise MissingLogits(f"Line {line.id} has {line.logits} in place of logits")

    return line.get_full_logprobs()


class PageDecoder:
    def __init__(self, decoder, line_confidence_threshold=None, carry_h_over=False):
        self.decoder = decoder
        self.line_confidence_threshold = line_confidence_threshold
        self.lines_examined = 0
        self.lines_decoded = 0
        self.seconds_decoding = 0.0
        self.continue_lines = carry_h_over

        self.last_h = None
        self.last_line = None

    def process_page(self, page_layout: PageLayout):
        self.last_h = None
        for line in page_layout.lines_iterator():
            try:
                line.transcription = self.decode_line(line)
            except Exception:
                logger.error(f'Failed to process line {line.id} of page {page_layout.id}. The page has been processed no further.', exc_info=True)

        return page_layout

    def decode_line(self, line):
        self.lines_examined += 1

        logits = prepare_dense_logits(line)
        if self.line_confidence_threshold is not None:
            if line_confident_enough(logits, self.line_confidence_threshold):
                self.last_h = None
                self.last_line = line.transcription
                return line.transcription

        t0 = time.time()
        if self.continue_lines:
            if not self.last_h and self.last_line:
                self.last_h = self.decoder._lm.initial_h_from_line(self.last_line)

            hypotheses, last_h = self.decoder(logits, return_h=True, init_h=self.last_h)
            last_h = self.decoder._lm.add_line_end(last_h)
            self.last_h = last_h
        else:
            hypotheses = self.decoder(logits)

        self.seconds_decoding += time.time() - t0
        self.lines_decoded += 1

        transcription = hypotheses.best_hyp()
        self.last_line = transcription

        return transcription

    def decoding_summary(self):
        if self.lines_examined == 0:
            return 'This PageDecoder has not processed a single line yet'

        if self.lines_decoded == 0:
            return f'Processed {self.lines_examined} lines, but none required actual decoding'

        decoded_pct = 100.0 * self.lines_decoded / self.lines_examined
        ms_per_line_decoded = 1000.0 * self.seconds_decoding / self.lines_decoded
        return f'Ran on {self.lines_examined}, decoded {self.lines_decoded} lines ({decoded_pct:.1f} %) in {self.seconds_decoding:.2f}s ({ms_per_line_decoded:.1f}ms per line)'


class WholePageRegion(object):
    def __init__(self, config, config_path=''):
        pass

    def process_page(self, img, page_layout: PageLayout):
        corners = np.asarray([
            [0, 0],
            [page_layout.page_size[1], 0],
            [page_layout.page_size[1], page_layout.page_size[0]],
            [0, page_layout.page_size[0]]
        ])
        page_layout.regions = [RegionLayout('r1', corners)]
        return page_layout


class TextlineExtractorSimple(object):
    def __init__(self, config, config_path=''):
        adaptive_threshold = config.getint('ADAPTIVE_THRESHOLD')
        block_size = config.getint('BLOCK_SIZE')
        minimum_length = config.getint('MINIMUM_LENGTH')
        ignored_border_pixels = config.getint('IGNORED_BORDER_PIXELS')
        self.engine = EngineLineDetectorSimple(
            adaptive_threshold=adaptive_threshold,
            block_size=block_size,
            minimum_length=minimum_length,
            ignored_border_pixels=ignored_border_pixels
        )

    def process_page(self, img, page_layout: PageLayout):
        for region in page_layout.regions:
            b_list, h_list, t_list = self.engine.detect_lines(
                img, region.polygon)
            for line_num, (baseline, heights, textline) in enumerate(zip(b_list, h_list, t_list)):
                new_textline = TextLine(
                    id='{}-l{:03d}'.format(region.id, line_num+1),
                    baseline=baseline,
                    polygon=textline,
                    heights=heights
                )
                region.lines.append(new_textline)
        return page_layout


class LayoutExtractor(object):
    def __init__(self, config, device, config_path=''):
        self.detect_regions = config.getboolean('DETECT_REGIONS')
        self.detect_lines = config.getboolean('DETECT_LINES')
        self.detect_straight_lines_in_regions = config.getboolean('DETECT_STRAIGHT_LINES_IN_REGIONS')
        self.merge_lines = config.getboolean('MERGE_LINES')
        self.adjust_heights = config.getboolean('ADJUST_HEIGHTS')
        self.multi_orientation = config.getboolean('MULTI_ORIENTATION')
        self.adjust_baselines = config.getboolean('ADJUST_BASELINES')

        use_cpu = config.getboolean('USE_CPU')
        self.device = device if not use_cpu else torch.device("cpu")

        self.engine = LayoutEngine(
            model_path=compose_path(config['MODEL_PATH'], config_path),
            device=self.device,
            downsample=config.getint('DOWNSAMPLE'),
            adaptive_downsample=config.getboolean('ADAPTIVE_DOWNSAMPLE', fallback=True),
            detection_threshold=config.getfloat('DETECTION_THRESHOLD'),
            max_mp=config.getfloat('MAX_MEGAPIXELS'),
            line_end_weight=config.getfloat('LINE_END_WEIGHT', fallback=1.0),
            vertical_line_connection_range=config.getint('VERTICAL_LINE_CONNECTION_RANGE', fallback=5),
            smooth_line_predictions=config.getboolean('SMOOTH_LINE_PREDICTIONS', fallback=True),
            paragraph_line_threshold=config.getfloat('PARAGRAPH_LINE_THRESHOLD', fallback=0.3),
        )
        self.pool = Pool(1)

    def process_page(self, img, page_layout: PageLayout):
        if self.detect_regions or self.detect_lines:
            if self.detect_regions:
                page_layout.regions = []
            if self.detect_lines:
                for region in page_layout.regions:
                    region.lines = []

            if self.multi_orientation:
                orientations = [0, 1, 3]
            else:
                orientations = [0]

            for rot in orientations:
                regions = []
                p_list, b_list, h_list, t_list = self.engine.detect(img, rot=rot)
                if self.detect_regions:
                    for id, polygon in enumerate(p_list):
                        if rot > 0:
                            id = 'r{:03d}_{}'.format(id, rot)
                        else:
                            id = 'r{:03d}'.format(id)
                        region = RegionLayout(id, polygon)
                        regions.append(region)
                if self.detect_lines:
                    if not self.detect_regions:
                        regions = page_layout.regions
                    regions = helpers.assign_lines_to_regions(
                        b_list, h_list, t_list, regions)
                if self.detect_regions:
                    page_layout.regions += regions

        if self.merge_lines:
            for region in page_layout.regions:
                while True:
                    original_line_count = len(region.lines)
                    r_b_list, r_h_list = helpers.merge_lines(
                        [line.baseline for line in region.lines],
                        [line.heights for line in region.lines]
                    )
                    r_t_list = [helpers.baseline_to_textline(b, h) for b, h in zip(r_b_list, r_h_list)]
                    region.lines = []
                    region = helpers.assign_lines_to_regions(
                        r_b_list, r_h_list, r_t_list, [region])[0]
                    if len(region.lines) == original_line_count:
                        break

        if self.detect_straight_lines_in_regions or self.adjust_heights or self.adjust_baselines:
            maps, ds = self.engine.parsenet.get_maps_with_optimal_resolution(img)

        if self.detect_straight_lines_in_regions:
            for region in page_layout.regions:
                pb_list, ph_list, pt_list = detect_lines_in_region(region.polygon, maps, ds)
                region.lines = []
                region = helpers.assign_lines_to_regions(pb_list, ph_list, pt_list, [region])[0]

        if self.adjust_heights:
            for line in page_layout.lines_iterator():
                sample_points = helpers.resample_baselines(
                    [line.baseline], num_points=40)[0]
                line.heights = self.engine.get_heights(maps, ds, sample_points)
                line.polygon = helpers.baseline_to_textline(
                    line.baseline, line.heights)

        if self.adjust_baselines:
            crop_engine = cropper.EngineLineCropper(
                line_height=32, poly=0, scale=1)
            for line in page_layout.lines_iterator():
                line.baseline = refine_baseline(line.baseline, line.heights, maps, ds, crop_engine)
                line.polygon = helpers.baseline_to_textline(line.baseline, line.heights)
        return page_layout


class LineFilter(object):
    def __init__(self, config, device, config_path):
        self.filter_directions = config.getboolean('FILTER_DIRECTIONS')
        self.filter_incomplete_pages = config.getboolean('FILTER_INCOMPLETE_PAGES')
        self.filter_pages_with_short_lines = config.getboolean('FILTER_PAGES_WITH_SHORT_LINES')
        self.length_threshold = config.getint('LENGTH_THRESHOLD')

        use_cpu = config.getboolean('USE_CPU')
        self.device = device if not use_cpu else torch.device("cpu")

        if self.filter_directions:
            self.engine = LineFilterEngine(
                model_path=compose_path(config['MODEL_PATH'], config_path),
                device=self.device
            )

    def process_page(self, img, page_layout: PageLayout):
        if self.filter_directions:
            self.engine.predict_directions(img)
            for region in page_layout.regions:
                region.lines = [line for line in region.lines if self.engine.check_line_rotation(line.polygon, line.baseline)]

        if self.filter_incomplete_pages:
            for region in page_layout.regions:
                region.lines = [line for line in region.lines if helpers.check_line_position(line.baseline, page_layout.page_size)]

        if self.filter_pages_with_short_lines:
            b_list = [line.baseline for line in page_layout.lines_iterator()]
            if helpers.get_max_line_length(b_list) < self.length_threshold:
                page_layout.regions = []

        page_layout.regions = [region for region in page_layout.regions if region.lines]

        return page_layout


class LinePostprocessor(object):
    def __init__(self, config, config_path=''):
        stretch_lines = config['STRETCH_LINES']
        if stretch_lines != 'max':
            stretch_lines = int(stretch_lines)
        self.engine = PostprocessingEngine(
            stretch_lines=stretch_lines,
            resample_lines=config.getboolean('RESAMPLE_LINES'),
            heights_from_regions=config.getboolean('HEIGHTS_FROM_REGIONS')
        )

    def process_page(self, img, page_layout: PageLayout):
        if not page_layout.regions:
            print(f"Warning: Skipping line post processing for page {page_layout.id}. No text region present.")
            return page_layout

        for region in page_layout.regions:
            region = self.engine.postprocess(region)

        return page_layout


class LayoutPostprocessor(object):
    def __init__(self, config, config_path=''):
        self.retrace_regions = config.getboolean('RETRACE_REGIONS')

    def process_page(self, img, page_layout: PageLayout):
        if not page_layout.regions:
            print(f"Warning: Skipping layout post processing for page {page_layout.id}. No text region present.")
            return page_layout

        if self.retrace_regions:
            for region in page_layout.regions:
                helpers.retrace_region(region)

        return page_layout


class LineCropper(object):
    def __init__(self, config, config_path=''):
        poly = config.getint('INTERP')
        line_scale = config.getfloat('LINE_SCALE')
        line_height = config.getint('LINE_HEIGHT')
        self.crop_engine = cropper.EngineLineCropper(
            line_height=line_height, poly=poly, scale=line_scale)

    def process_page(self, img, page_layout: PageLayout):
        for line in page_layout.lines_iterator():
            try:
                line.crop = self.crop_engine.crop(
                    img, line.baseline, line.heights)
            except ValueError:
                line.crop = np.zeros(
                    (self.crop_engine.line_height, self.crop_engine.line_height, 3))
                print(f"WARNING: Failed to crop line {line.id} in page {page_layout.id}. Probably contain vertical line. Contanct Olda Kodym to fix this bug!")
        return page_layout

    def crop_lines(self, img, lines: list):
        for line in lines:
            try:
                line.crop = self.crop_engine.crop(
                    img, line.baseline, line.heights)
            except ValueError:
                line.crop = np.zeros(
                    (self.crop_engine.line_height, self.crop_engine.line_height, 3))
                print(f"WARNING: Failed to crop line {line.id}. Probably contain vertical line. Contanct Olda Kodym to fix this bug!")


class PageOCR(object):
    def __init__(self, config, device, config_path=''):
        json_file = compose_path(config['OCR_JSON'], config_path)
        use_cpu = config.getboolean('USE_CPU')

        self.device = device if not use_cpu else torch.device("cpu")
        self.ocr_engine = PytorchEngineLineOCR(json_file, self.device)

    def process_page(self, img, page_layout: PageLayout):
        for line in page_layout.lines_iterator():
            if line.crop is None:
                raise Exception(f'Missing crop in line {line.id}.')

        transcriptions, logits, logit_coords = self.ocr_engine.process_lines([line.crop for line in page_layout.lines_iterator()])

        for line, line_transcription, line_logits, line_logit_coords in zip(page_layout.lines_iterator(), transcriptions, logits, logit_coords):
            line.transcription = line_transcription
            line.logits = line_logits
            line.characters = self.ocr_engine.characters
            line.logit_coords = line_logit_coords
        return page_layout


def get_prob(best_ids, best_probs):
    last_id = -1
    last_prob = 1
    worst_prob = 1
    for id, prob in zip(best_ids, best_probs):
        if id != last_id:
            worst_prob = min(worst_prob, last_prob)
            last_prob = prob
            last_id = id
        else:
            last_prob = max(prob, last_prob)

    worst_prob = min(worst_prob, last_prob)
    return worst_prob


class PageParser(object):
    def __init__(self, config, device, config_path='', ):
        self.run_layout_parser = config['PAGE_PARSER'].getboolean('RUN_LAYOUT_PARSER', fallback=False)
        self.run_line_cropper = config['PAGE_PARSER'].getboolean('RUN_LINE_CROPPER', fallback=False)
        self.run_ocr = config['PAGE_PARSER'].getboolean('RUN_OCR', fallback=False)
        self.run_decoder = config['PAGE_PARSER'].getboolean('RUN_DECODER', fallback=False)
        self.filter_confident_lines_threshold = config['PAGE_PARSER'].getfloat('FILTER_CONFIDENT_LINES_THRESHOLD',
                                                                               fallback=-1)

        self.layout_parser = None
        self.line_cropper = None
        self.ocr = None
        self.decoder = None

        self.device = device

        if self.run_layout_parser:
            self.layout_parsers = []
            for i in range(1, 10):
                if config.has_section('LAYOUT_PARSER_{}'.format(i)):
                    self.layout_parsers.append(layout_parser_factory(config, device, config_path=config_path, order=i))
        if self.run_line_cropper:
            self.line_cropper = line_cropper_factory(config, config_path=config_path)
        if self.run_ocr:
            self.ocr = ocr_factory(config, device, config_path=config_path)
        if self.run_decoder:
            self.decoder = page_decoder_factory(config, device, config_path=config_path)

    @staticmethod
    def compute_line_confidence(line, threshold=None):
        logits = line.get_dense_logits()
        log_probs = logits - np.logaddexp.reduce(logits, axis=1)[:, np.newaxis]
        best_ids = np.argmax(log_probs, axis=-1)
        best_probs = np.exp(np.max(log_probs, axis=-1))
        worst_best_prob = get_prob(best_ids, best_probs)
        # print(worst_best_prob, np.sum(np.exp(best_probs) < threshold), best_probs.shape, np.nonzero(np.exp(best_probs) < threshold))
        # for i in np.nonzero(np.exp(best_probs) < threshold)[0]:
        #     print(best_probs[i-1:i+2], best_ids[i-1:i+2])

        return worst_best_prob

    def update_confidences(self, page_layout):
        for line in page_layout.lines_iterator():
            if line.logits is not None:
                line.transcription_confidence = self.compute_line_confidence(line)

    def filter_confident_lines(self, page_layout):
        for region in page_layout.regions:
            region.lines = [line for line in region.lines if line.transcription_confidence > self.filter_confident_lines_threshold]
        return page_layout

    def process_page(self, image, page_layout):
        if self.run_layout_parser:
            for layout_parser in self.layout_parsers:
                page_layout = layout_parser.process_page(image, page_layout)
        if self.run_line_cropper:
            page_layout = self.line_cropper.process_page(image, page_layout)
        if self.run_ocr:
            page_layout = self.ocr.process_page(image, page_layout)
        if self.run_decoder:
            page_layout = self.decoder.process_page(page_layout)

        self.update_confidences(page_layout)

        if self.filter_confident_lines_threshold > 0:
            page_layout = self.filter_confident_lines(page_layout)

        return page_layout
