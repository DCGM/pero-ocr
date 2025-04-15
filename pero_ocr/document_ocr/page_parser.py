import numpy as np

import logging
from multiprocessing import Pool
import math
import time
import re
from typing import Union, Tuple, List

import torch.cuda

from pero_ocr.utils import compose_path, config_get_list
from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
import pero_ocr.core.crop_engine as cropper
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
from pero_ocr.ocr_engine.transformer_ocr_engine import TransformerEngineLineOCR
from pero_ocr.layout_engines.simple_region_engine import SimpleThresholdRegion
from pero_ocr.layout_engines.simple_baseline_engine import EngineLineDetectorSimple
from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine, LineFilterEngine, LayoutEngineYolo
from pero_ocr.layout_engines.line_postprocessing_engine import PostprocessingEngine
from pero_ocr.layout_engines.naive_sorter import NaiveRegionSorter
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter
from pero_ocr.layout_engines.line_in_region_detector import detect_lines_in_region
from pero_ocr.layout_engines.baseline_refiner import refine_baseline
from pero_ocr.layout_engines import layout_helpers as helpers


logger = logging.getLogger(__name__)


def layout_parser_factory(config, device, config_path=''):
    if config['METHOD'] == 'REGION_WHOLE_PAGE':
        layout_parser = WholePageRegion(config, config_path=config_path)
    elif config['METHOD'] == 'REGION_SIMPLE_THRESHOLD':
        layout_parser = SimpleThresholdRegion(config, config_path=config_path)
    elif config['METHOD'] == 'LAYOUT_CNN':
        layout_parser = LayoutExtractor(config, device, config_path=config_path)
    elif config['METHOD'] == 'LAYOUT_YOLO':
        layout_parser = LayoutExtractorYolo(config, device, config_path=config_path)
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


def line_cropper_factory(config, config_path='', device=None):
    return LineCropper(config, config_path=config_path)


def ocr_factory(config, device, config_path=''):
    return PageOCR(config, device, config_path=config_path)


def page_decoder_factory(config, device, config_path=''):
    from pero_ocr.decoding import decoding_itf
    ocr_chars = decoding_itf.get_ocr_charset(compose_path(config['OCR']['OCR_JSON'], config_path))

    use_cpu = config['DECODER'].getboolean('USE_CPU')
    device = device if not use_cpu else torch.device("cpu")

    decoder = decoding_itf.decoder_factory(config['DECODER'], ocr_chars, device, allow_no_decoder=False, config_path=config_path)
    confidence_threshold = config['DECODER'].getfloat('CONFIDENCE_THRESHOLD', fallback=math.inf)
    carry_h_over = config['DECODER'].getboolean('CARRY_H_OVER')
    categories = config_get_list(config['DECODER'], key='CATEGORIES', fallback=[])
    return PageDecoder(decoder, line_confidence_threshold=confidence_threshold, carry_h_over=carry_h_over,
                       categories=categories)


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
    def __init__(self, decoder, line_confidence_threshold=None, carry_h_over=False, categories=None):
        self.decoder = decoder
        self.line_confidence_threshold = line_confidence_threshold
        self.lines_examined = 0
        self.lines_decoded = 0
        self.seconds_decoding = 0.0
        self.continue_lines = carry_h_over
        self.categories = categories if categories else ['text']

        self.last_h = None
        self.last_line = None

    def process_page(self, page_layout: PageLayout):
        self.last_h = None
        for line in page_layout.lines_iterator(self.categories):
            try:
                line.transcription = self.decode_line(line)
            except Exception:
                logger.error(f'Failed to process line {line.id} of page {page_layout.id}. '
                             f'The page has been processed no further.', exc_info=True)

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
                    heights=heights,
                    category='text'
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
        self.categories = config_get_list(config, key='CATEGORIES', fallback=[])

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
        page_layout, page_layout_no_text = helpers.split_page_layout(page_layout)

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
                        region = RegionLayout(id, polygon, category='text')
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
            for line in page_layout.lines_iterator(self.categories):
                sample_points = helpers.resample_baselines(
                    [line.baseline], num_points=40)[0]
                line.heights = self.engine.get_heights(maps, ds, sample_points)
                line.polygon = helpers.baseline_to_textline(
                    line.baseline, line.heights)

        if self.adjust_baselines:
            crop_engine = cropper.EngineLineCropper(
                line_height=32, poly=0, scale=1)
            for line in page_layout.lines_iterator(self.categories):
                line.baseline = refine_baseline(line.baseline, line.heights, maps, ds, crop_engine)
                line.polygon = helpers.baseline_to_textline(line.baseline, line.heights)
        page_layout = helpers.merge_page_layouts(page_layout, page_layout_no_text)
        return page_layout


class LayoutExtractorYolo(object):
    def __init__(self, config, device, config_path=''):
        try:
            import ultralytics  # check if ultralytics library is installed
            # (ultralytics need different numpy version than some specific version installed on pero-ocr machines)
        except ImportError:
            raise ImportError("To use LayoutExtractorYolo, you need to install ultralytics library. "
                              "You can do it by running 'pip install ultralytics'.")

        use_cpu = config.getboolean('USE_CPU')
        self.device = device if not use_cpu else torch.device("cpu")
        self.categories = config_get_list(config, key='CATEGORIES', fallback=[])
        self.line_categories = config_get_list(config, key='LINE_CATEGORIES', fallback=[])
        self.image_size = self.get_image_size(config)

        self.engine = LayoutEngineYolo(
            model_path=compose_path(config['MODEL_PATH'], config_path),
            device=self.device,
            detection_threshold=config.getfloat('DETECTION_THRESHOLD'),
            image_size=self.image_size
        )

    def process_page(self, img, page_layout: PageLayout):
        page_layout_text, page_layout = helpers.split_page_layout(page_layout)
        page_layout.regions = []

        result = self.engine.detect(img)
        start_id = self.get_start_id([region.id for region in page_layout_text.regions])

        boxes = result.boxes.data.cpu()
        for box_id, box in enumerate(boxes):
            id_str = 'r{:03d}'.format(start_id + box_id)

            x_min, y_min, x_max, y_max, conf, class_id = box.tolist()
            polygon = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]])
            baseline_y = y_min + (y_max - y_min) / 2
            baseline = np.array([[x_min, baseline_y], [x_max, baseline_y]])
            height = np.floor(np.array([baseline_y - y_min, y_max - baseline_y]))

            category = result.names[class_id]
            if self.categories and category not in self.categories:
                continue

            region = RegionLayout(id_str, polygon, category=category, detection_confidence=conf)

            if category in self.line_categories:
                line = TextLine(
                    id=f'{id_str}-l000',
                    index=0,
                    polygon=polygon,
                    baseline=baseline,
                    heights=height,
                    category=category
                )
                region.lines.append(line)
            page_layout.regions.append(region)

        page_layout = helpers.merge_page_layouts(page_layout_text, page_layout)
        return page_layout

    @staticmethod
    def get_image_size(config) -> Union[int, Tuple[int, int], None]:
        if 'IMAGE_SIZE' not in config:
            return None

        try:
            image_size = config.getint('IMAGE_SIZE')
        except ValueError:
            image_size = config_get_list(config, key='IMAGE_SIZE')
            if len(image_size) != 2:
                raise ValueError(f'Invalid image size. Expected int or list of two ints, but got: '
                                 f'{image_size} of type {type(image_size)}')
            image_size = image_size[0], image_size[1]
        return image_size

    @staticmethod
    def get_start_id(used_ids: list) -> int:
        """Get int from which to start id naming for new regions.

        Expected region id is in format rXXX, where XXX is number.
        """
        used_region_ids = sorted(used_ids)
        if not used_region_ids:
            return 0

        ids = []
        for id in used_region_ids:
            id = re.match(r'r(\d+)', id).group(1)
            try:
                ids.append(int(id))
            except ValueError:
                pass

        last_used_id = sorted(ids)[-1]
        return last_used_id + 1


class LineFilter(object):
    def __init__(self, config, device, config_path):
        self.filter_directions = config.getboolean('FILTER_DIRECTIONS')
        self.filter_incomplete_pages = config.getboolean('FILTER_INCOMPLETE_PAGES')
        self.filter_pages_with_short_lines = config.getboolean('FILTER_PAGES_WITH_SHORT_LINES')
        self.length_threshold = config.getint('LENGTH_THRESHOLD')
        self.categories = config_get_list(config, key='CATEGORIES', fallback=[])

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
            b_list = [line.baseline for line in page_layout.lines_iterator(self.categories)]
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
        self.categories = config_get_list(config, key='CATEGORIES', fallback=[])
        self.crop_engine = cropper.EngineLineCropper(
            line_height=line_height, poly=poly, scale=line_scale)

    def process_page(self, img, page_layout: PageLayout):
        for line in page_layout.lines_iterator(self.categories):
            try:
                line.crop = self.crop_engine.crop(
                    img, line.baseline, line.heights)
            except ValueError:
                line.crop = np.zeros(
                    (self.crop_engine.line_height, self.crop_engine.line_height, 3))
                print(f"WARNING: Failed to crop line {line.id} in page {page_layout.id}. "
                      f"Probably contain vertical line. Contanct Olda Kodym to fix this bug!")
        return page_layout

    def crop_lines(self, img, lines: list):
        for line in lines:
            try:
                line.crop = self.crop_engine.crop(
                    img, line.baseline, line.heights)
            except ValueError:
                line.crop = np.zeros(
                    (self.crop_engine.line_height, self.crop_engine.line_height, 3))
                print(f"WARNING: Failed to crop line {line.id}. Probably contain vertical line. "
                      f"Contanct Olda Kodym to fix this bug!")


class PageOCR:
    default_confidence = 0.0

    def __init__(self, config, device, config_path=''):
        json_file = compose_path(config['OCR_JSON'], config_path)
        use_cpu = config.getboolean('USE_CPU')

        self.device = device if not use_cpu else torch.device("cpu")
        self.categories = config_get_list(config, key='CATEGORIES', fallback=[])
        self.substitute_output = config.getboolean('SUBSTITUTE_OUTPUT', fallback=True)
        self.substitute_output_atomic = config.getboolean('SUBSTITUTE_OUTPUT_ATOMIC', fallback=True)
        self.update_transcription_by_confidence = config.getboolean(
            'UPDATE_TRANSCRIPTION_BY_CONFIDENCE', fallback=False)

        if 'METHOD' in config and config['METHOD'] == "pytorch_ocr-transformer":
            self.ocr_engine = TransformerEngineLineOCR(json_file, self.device,
                                                       substitute_output_atomic=self.substitute_output_atomic)
        else:
            self.ocr_engine = PytorchEngineLineOCR(json_file, self.device,
                                                   substitute_output_atomic=self.substitute_output_atomic)

    def process_page(self, img, page_layout: PageLayout):
        lines_to_process = []
        for line in page_layout.lines_iterator(self.categories):
            if line.crop is None:
                raise Exception(f'Missing crop in line {line.id}.')
            lines_to_process.append(line)

        transcriptions, logits, logit_coords = self.ocr_engine.process_lines([line.crop for line in lines_to_process])

        for line, line_transcription, line_logits, line_logit_coords in zip(lines_to_process, transcriptions,
                                                                            logits, logit_coords):
            new_line = TextLine(id=line.id,
                                transcription=line_transcription,
                                logits=line_logits,
                                characters=self.ocr_engine.characters,
                                logit_coords=line_logit_coords)
            new_line.calculate_confidences(default_transcription_confidence=self.default_confidence)

            if not self.update_transcription_by_confidence:
                self.update_line(line, new_line)
            else:
                if (line.transcription_confidence in [None, self.default_confidence] or
                        line.transcription_confidence < new_line.transcription_confidence):
                    self.update_line(line, new_line)

        if self.substitute_output and self.ocr_engine.output_substitution is not None:
            self.substitute_transcriptions(lines_to_process)

        return page_layout

    def substitute_transcriptions(self, lines_to_process: List[TextLine]):
        transcriptions_substituted = []

        for line in lines_to_process:
            transcriptions_substituted.append(self.ocr_engine.output_substitution(line.transcription))

            if transcriptions_substituted[-1] is None:
                if self.substitute_output_atomic:
                    return  # scratch everything if the last line couldn't be substituted atomically
                else:
                    transcriptions_substituted[-1] = line.transcription  # keep the original transcription

        for line, transcription_substituted in zip(lines_to_process, transcriptions_substituted):
            line.transcription = transcription_substituted

    @property
    def provides_ctc_logits(self):
        return isinstance(self.ocr_engine, PytorchEngineLineOCR) or isinstance(self.ocr_engine, TransformerEngineLineOCR)

    @staticmethod
    def update_line(line, new_line):
        line.transcription = new_line.transcription
        line.logits = new_line.logits
        line.characters = new_line.characters
        line.logit_coords = new_line.logit_coords
        line.character_confidences = new_line.character_confidences
        line.transcription_confidence = new_line.transcription_confidence


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


def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PageParser(object):
    def __init__(self, config, device=None, config_path='', ):
        if not config.sections():
            raise ValueError('Config file is empty or does not exist.')

        self.run_layout_parser = config['PAGE_PARSER'].getboolean('RUN_LAYOUT_PARSER', fallback=False)
        self.run_line_cropper = config['PAGE_PARSER'].getboolean('RUN_LINE_CROPPER', fallback=False)
        self.run_ocr = config['PAGE_PARSER'].getboolean('RUN_OCR', fallback=False)
        self.run_decoder = config['PAGE_PARSER'].getboolean('RUN_DECODER', fallback=False)
        self.filter_confident_lines_threshold = config['PAGE_PARSER'].getfloat('FILTER_CONFIDENT_LINES_THRESHOLD',
                                                                               fallback=-1)

        self.device = device if device is not None else get_default_device()

        self.layout_parsers = {}
        self.line_croppers = {}
        self.ocrs = {}
        self.decoder = None

        if self.run_layout_parser:
            self.layout_parsers = self.init_config_sections(config, config_path, 'LAYOUT_PARSER', layout_parser_factory)
        if self.run_line_cropper:
            self.line_croppers = self.init_config_sections(config, config_path, 'LINE_CROPPER', line_cropper_factory)
        if self.run_ocr:
            self.ocrs = self.init_config_sections(config, config_path, 'OCR', ocr_factory)
        if self.run_decoder:
            self.decoder = page_decoder_factory(config, self.device, config_path=config_path)

    @property
    def provides_ctc_logits(self):
        if not self.ocrs:
            return False

        return any(ocr.provides_ctc_logits for ocr in self.ocrs.values())

    def filter_confident_lines(self, page_layout):
        for region in page_layout.regions:
            region.lines = [line for line in region.lines if line.transcription_confidence is None or
                            line.transcription_confidence > self.filter_confident_lines_threshold]
        return page_layout

    def process_page(self, image, page_layout):
        if self.run_layout_parser:
            for _, layout_parser in sorted(self.layout_parsers.items()):
                page_layout = layout_parser.process_page(image, page_layout)

        merged_keys = set(self.line_croppers.keys()) | set(self.ocrs.keys())
        for key in sorted(merged_keys):
            if self.run_line_cropper and key in self.line_croppers:
                page_layout = self.line_croppers[key].process_page(image, page_layout)
            if self.run_ocr and key in self.ocrs:
                page_layout = self.ocrs[key].process_page(image, page_layout)

        if self.run_decoder:
            page_layout = self.decoder.process_page(page_layout)

        for line in page_layout.lines_iterator():
            line.calculate_confidences()

        if self.filter_confident_lines_threshold > 0:
            page_layout = self.filter_confident_lines(page_layout)

        page_layout.calculate_confidence()

        return page_layout

    def init_config_sections(self, config, config_path, section_name, section_factory) -> dict:
        """Return dict of sections.

        Naming convention: section_name_[0-9]+.
            Also accepts other names, but logges warning.
            e.g. for OCR section: OCR, OCR_0, OCR_42_asdf, OCR_99_last_one..."""
        sections = {}
        if section_name in config.sections():
            sections['-1'] = section_name

        section_names = [config_section for config_section in config.sections()
                         if re.match(rf'{section_name}_(\d+)', config_section)]
        section_names = sorted(section_names)

        for config_section in section_names:
            section_id = config_section.replace(section_name + '_', '')
            try:
                int(section_id)
            except ValueError:
                logger.warning(
                    f'Warning: section name {config_section} does not follow naming convention. '
                    f'Use only {section_name}_[0-9]+.')
            sections[section_id] = config_section

        if 0 in sections.keys() and -1 in sections.keys():
            logger.warning(f'Warning: sections {sections[0]} and {sections[-1]} are both present. '
                           f'Use only names following {section_name}_[0-9]+ convention.')

        for section_id, section_full_name in sections.items():
            sections[section_id] = section_factory(config[section_full_name],
                                                   config_path=config_path, device=self.device)

        return sections
