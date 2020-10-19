import numpy as np

from multiprocessing import Pool
import math
import time

from pero_ocr.utils import compose_path
from .layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import crop_engine as cropper
from pero_ocr.ocr_engine import line_ocr_engine
from pero_ocr.layout_engines.simple_region_engine import SimpleThresholdRegion
from pero_ocr.layout_engines.simple_baseline_engine import EngineLineDetectorSimple
from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine, LineFilterEngine
from pero_ocr.layout_engines.line_postprocessing_engine import PostprocessingEngine
from pero_ocr.layout_engines.naive_sorter import NaiveRegionSorter
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter
from pero_ocr.layout_engines import layout_helpers as helpers


def layout_parser_factory(config, config_path='', order=1):
    config = config['LAYOUT_PARSER_{}'.format(order)]
    if config['METHOD'] == 'REGION_WHOLE_PAGE':
        layout_parser = WholePageRegion(config, config_path=config_path)
    elif config['METHOD'] == 'REGION_SIMPLE_THRESHOLD':
        layout_parser = SimpleThresholdRegion(config, config_path=config_path)
    elif config['METHOD'] == 'LAYOUT_CNN':
        layout_parser = LayoutExtractor(config, config_path=config_path)
    elif config['METHOD'] == 'LINES_SIMPLE_THRESHOLD':
        layout_parser = TextlineExtractorSimple(config, config_path=config_path)
    elif config['METHOD'] == 'LINE_FILTER':
        layout_parser = LineFilter(config, config_path=config_path)
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


def ocr_factory(config, config_path=''):
    config = config['OCR']
    return PageOCR(config, config_path=config_path)


def page_decoder_factory(config, config_path=''):
    from pero_ocr.decoding import decoding_itf
    ocr_chars = decoding_itf.get_ocr_charset(compose_path(config['OCR']['OCR_JSON'], config_path))
    decoder = decoding_itf.decoder_factory(config['DECODER'], ocr_chars, allow_no_decoder=False, use_gpu=True,
                                           config_path=config_path)
    confidence_threshold = config['DECODER'].getfloat('CONFIDENCE_THRESHOLD', fallback=math.inf)
    return PageDecoder(decoder, line_confidence_threshold=confidence_threshold)


class MissingLogits(Exception):
    pass


class PageDecoder:
    def __init__(self, decoder, line_confidence_threshold=None):
        self.decoder = decoder
        self.line_confidence_threshold = line_confidence_threshold
        self.lines_examined = 0
        self.lines_decoded = 0
        self.seconds_decoding = 0.0

    def process_page(self, page_layout: PageLayout):
        for line in page_layout.lines_iterator():
            self.lines_examined += 1
            logits = self.prepare_dense_logits(line)
            if self.line_confidence_threshold is not None:
                if self.line_confident_enough(logits):
                    continue

            t0 = time.time()
            hypotheses = self.decoder(logits)
            self.seconds_decoding += time.time() - t0
            self.lines_decoded += 1
            if hypotheses is not None:
                line.transcription = hypotheses.best_hyp()

        return page_layout

    def prepare_dense_logits(self, line):
        if line.logits is None:
            raise MissingLogits(f"Line {line.id} has {line.logits} in place of logits")

        return line.get_dense_logits()

    def line_confident_enough(self, logits):
        log_probs = logits - np.logaddexp.reduce(logits, axis=1)[:, np.newaxis]
        best_probs = np.max(log_probs, axis=-1)
        worst_best_prob = np.exp(np.min(best_probs))

        return worst_best_prob > self.line_confidence_threshold


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
    def __init__(self, config, config_path=''):
        self.detect_regions = config.getboolean('DETECT_REGIONS')
        self.detect_lines = config.getboolean('DETECT_LINES')
        self.merge_lines = config.getboolean('MERGE_LINES')
        self.adjust_lines = config.getboolean('REFINE_LINES')
        self.multi_orientation = config.getboolean('MULTI_ORIENTATION')
        self.engine = LayoutEngine(
            model_path=compose_path(config['MODEL_PATH'], config_path),
            downsample=config.getint('DOWNSAMPLE'),
            pad=config.getint('PAD'),
            use_cpu=config.getboolean('USE_CPU'),
            detection_threshold=config.getfloat('DETECTION_THRESHOLD'),
            max_mp=config.getfloat('MAX_MEGAPIXELS'),
            gpu_fraction=config.getfloat('GPU_FRACTION')
        )
        self.pool = Pool()

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
                    # if len(regions) > 4:
                    #     regions = list(self.pool.map(partial(helpers.assign_lines_to_region, b_list, h_list, t_list),
                    #                      regions))
                    # else:
                    for region in regions:
                        region = helpers.assign_lines_to_region(
                            b_list, h_list, t_list, region)

                if self.detect_regions:
                    page_layout.regions += regions

        if self.merge_lines:
            for region in page_layout.regions:
                r_b_list, r_h_list = helpers.merge_lines(
                    [line.baseline for line in region.lines],
                    [line.heights for line in region.lines]
                )
                r_t_list = [helpers.baseline_to_textline(b, h) for b, h in zip(r_b_list, r_h_list)]
                region.lines = []
                region = helpers.assign_lines_to_region(
                    r_b_list, r_h_list, r_t_list, region)

        if self.adjust_lines:
            heights_map, ds = self.engine.get_maps(img)[:, :, :2]
            for line in page_layout.lines_iterator():
                sample_points = helpers.resample_baselines(
                    [line.baseline], num_points=40)[0]
                line.heights = self.engine.get_heights(heights_map, ds, sample_points)
                line.polygon = helpers.baseline_to_textline(
                    line.baseline, line.heights)

        return page_layout


class LineFilter(object):
    def __init__(self, config, config_path):
        self.filter_directions = config.getboolean('FILTER_DIRECTIONS')
        self.filter_incomplete_pages = config.getboolean('FILTER_INCOMPLETE_PAGES')

        if self.filter_directions:
            self.engine = LineFilterEngine(
                model_path=compose_path(config['MODEL_PATH'], config_path),
                gpu_fraction=config.getfloat('GPU_FRACTION')
            )

    def process_page(self, img, page_layout: PageLayout):
        if self.filter_directions:
            self.engine.predict_directions(img)
            for region in page_layout.regions:
                region.lines = [line for line in region.lines if self.engine.check_line_rotation(line.polygon, line.baseline)]

        if self.filter_incomplete_pages:
            for region in page_layout.regions:
                region.lines = [line for line in region.lines if helpers.check_line_position(line.baseline, page_layout.page_size)]

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


class PageOCR(object):
    def __init__(self, config, config_path=''):
        json_file = compose_path(config['OCR_JSON'], config_path)
        if 'METHOD' in config and config['METHOD'] == 'pytorch_ocr':
            from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
            self.ocr_engine = PytorchEngineLineOCR(json_file, gpu_id=0)
        else:
            self.ocr_engine = line_ocr_engine.EngineLineOCR(json_file, gpu_id=0)

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
    def __init__(self, config, config_path=''):
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

        if self.run_layout_parser:
            self.layout_parsers = []
            for i in range(1, 10):
                if config.has_section('LAYOUT_PARSER_{}'.format(i)):
                    self.layout_parsers.append(layout_parser_factory(config, config_path=config_path, order=i))
        if self.run_line_cropper:
            self.line_cropper = line_cropper_factory(config, config_path=config_path)
        if self.run_ocr:
            self.ocr = ocr_factory(config, config_path=config_path)
        if self.run_decoder:
            self.decoder = page_decoder_factory(config, config_path=config_path)

    @staticmethod
    def compute_line_confidence(line, threshold):
        logits = line.get_dense_logits()
        log_probs = logits - np.logaddexp.reduce(logits, axis=1)[:, np.newaxis]
        best_ids = np.argmax(log_probs, axis=-1)
        best_probs = np.exp(np.max(log_probs, axis=-1))
        worst_best_prob = get_prob(best_ids, best_probs)
        print(worst_best_prob, np.sum(np.exp(best_probs) < threshold), best_probs.shape, np.nonzero(np.exp(best_probs) < threshold))
        # for i in np.nonzero(np.exp(best_probs) < threshold)[0]:
        #     print(best_probs[i-1:i+2], best_ids[i-1:i+2])

        return worst_best_prob

    def filter_confident_lines(self, page_layout):
        for region in page_layout.regions:
            region.lines = [line for line in region.lines
                            if PageParser.compute_line_confidence(line, self.filter_confident_lines_threshold) > self.filter_confident_lines_threshold]
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
        if self.filter_confident_lines_threshold > 0:
            page_layout = self.filter_confident_lines(page_layout)

        return page_layout
