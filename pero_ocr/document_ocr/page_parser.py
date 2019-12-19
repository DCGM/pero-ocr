import numpy as np

from .layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import crop_engine as cropper
from pero_ocr.ocr_engine import line_ocr_engine
from pero_ocr.line_engine import baseline_engine
from pero_ocr.region_engine import region_engine
from pero_ocr.region_engine import SimpleThresholdRegion
import pero_ocr.line_engine.line_postprocessing as linepp



def layout_parser_factory(config):
    config = config['LAYOUT_PARSER']
    if config['METHOD'] == 'WHOLE_PAGE_REGION':
        region_parser = WholePageRegion(config)
    elif config['METHOD'] == 'cnn':
        region_parser = RegionExtractorCNN(config)
    elif config['METHOD'] == 'SIMPLE_THRESHOLD_REGION':
        region_parser = SimpleThresholdRegion(config)
    else:
        raise ValueError('Unknown layout parser method: {}'.format(config['METHOD']))
    return region_parser


def line_parser_factory(config):
    config = config['LINE_PARSER']
    if config['METHOD'] == 'cnn':
        line_parser = TextlineExtractorCNN(config)
    elif config['METHOD'] == 'simple':
        line_parser = TextlineExtractorSimple(config)
    else:
        raise ValueError('Unknown line parser method: {}'.format(config['METHOD']))
    return line_parser


def line_cropper_factory(config):
    config = config['LINE_CROPPER']
    return LineCropper(config)


def ocr_factory(config):
    config = config['OCR']
    return PageOCR(config)


def page_decoder_factory(config):
    from pero.decoding import decoding_itf
    ocr_chars = decoding_itf.get_ocr_charset(config['OCR']['OCR_JSON'])
    decoder = decoding_itf.decoder_factory(config['DECODER'], ocr_chars, allow_no_decoder=False)
    return PageDecoder(decoder)


class MissingLogits(Exception):
    pass


class PageDecoder:
    def __init__(self, decoder):
        self.decoder = decoder

    def process_page(self, page_layout: PageLayout):
        for line in page_layout.lines_iterator():
            if line.logits is None:
                raise MissingLogits(f"Line {line.id} has {line.logits} in place of logits")

            logits = decoding_itf.prepare_dense_logits(line.logits)
            line.transcription = self.decoder(logits).best_hyp()

        return page_layout


class WholePageRegion(object):
    def __init__(self, config):
        pass

    def process_page(self, img, page_layout: PageLayout):
        corners = np.asarray([
            [0, 0],
            [0, page_layout.page_size[1]],
            [page_layout.page_size[0], page_layout.page_size[1]],
            [page_layout.page_size[0], 0]
        ])
        page_layout.regions = [RegionLayout('r1', corners)]
        return page_layout


class RegionExtractorCNN(object):
    def __init__(self, config):
        model_path = config['MODEL_PATH']
        downsample = config.getint('DOWNSAMPLE')
        use_cpu = config.getboolean('USE_CPU')
        self.region_engine = region_engine.EngineRegionDetector(
            model_path=model_path,
            downsample=downsample,
            use_cpu=use_cpu
        )

    def process_page(self, img, page_layout: PageLayout):
        region_list = self.region_engine.detect(img)
        for r_num, region in enumerate(region_list):
            new_region = RegionLayout('r{:03d}'.format(r_num), np.asarray(region))
            page_layout.regions.append(new_region)
        return page_layout


class BaseTextlineExtractor(object):
    def __init__(self, config):
        self.merge_lines = config.getboolean('MERGE_LINES')

    def assign_lines_to_region(self, baseline_list, heights_list, textline_list, region):
        for line_num, (baseline, heights, textline) in enumerate(zip(baseline_list, heights_list, textline_list)):
            baseline_intersection, textline_intersection = linepp.mask_textline_by_region(baseline, textline, region.polygon)
            if baseline_intersection is not None and textline_intersection is not None:
                new_textline = TextLine(id='{}-l{:03d}'.format(region.id, line_num+1), baseline=baseline_intersection, polygon=textline_intersection, heights=heights)
                region.lines.append(new_textline)
        return region

    def process_page(self, img, page_layout: PageLayout):
        if not page_layout.regions:
            print(f"Warning: Skipping line detection for page {page_layout.id}. No text region present.")
            return page_layout

        baseline_list, heights_list, textline_list = self.line_engine.detect_lines(img)

        for region in page_layout.regions:
            region = self.assign_lines_to_region(baseline_list, heights_list, textline_list, region)
            if self.merge_lines:
                region_baseline_list = [line.baseline for line in region.lines]
                region_heights_list = [line.heights for line in region.lines]
                region_baseline_list, region_heights_list = linepp.merge_lines(region_baseline_list, region_heights_list)
                region_textline_list = [linepp.baseline_to_textline(baseline, heights) for baseline, heights in zip(region_baseline_list, region_heights_list)]
                region.lines = []
                region = self.assign_lines_to_region(region_baseline_list, region_heights_list, region_textline_list, region)
        return page_layout


class TextlineExtractorCNN(BaseTextlineExtractor):
    def __init__(self, config):
        super(TextlineExtractorCNN, self).__init__(config)
        model_path = config['MODEL_PATH']
        downsample = config.getint('DOWNSAMPLE')
        pad = config.getint('PAD')
        use_cpu = config.getboolean('USE_CPU')
        order_lines = config['ORDER_LINES']
        detection_threshold = config.getfloat('DETECTION_THRESHOLD')
        stretch_lines = config.getint('STRETCH_LINES')
        self.line_engine = baseline_engine.EngineLineDetectorCNN(
            model_path=model_path,
            downsample=downsample,
            pad=pad,
            use_cpu=use_cpu,
            order_lines=order_lines,
            detection_threshold=detection_threshold,
            stretch_lines=stretch_lines
        )


class TextlineExtractorSimple(object):
    def __init__(self, config):
        adaptive_threshold = config.getint('ADAPTIVE_THRESHOLD')
        block_size = config.getint('BLOCK_SIZE')
        minimum_length = config.getint('MINIMUM_LENGTH')
        ignored_border_pixels = config.getint('IGNORED_BORDER_PIXELS')
        self.line_engine = baseline_engine.EngineLineDetectorSimple(
            adaptive_threshold=adaptive_threshold,
            block_size=block_size,
            minimum_length=minimum_length,
            ignored_border_pixels=ignored_border_pixels
        )

    def process_page(self, img, page_layout: PageLayout):
        for region in page_layout.regions:
            baselines_list, heights_list, textlines_list = self.line_engine.detect_lines(img, region.polygon)
            for line_num, (baseline, heights, textline) in enumerate(zip(baselines_list, heights_list, textlines_list)):
                new_textline = TextLine(id='{}-l{:03d}'.format(region.id, line_num+1), baseline=baseline, polygon=textline, heights=heights)
                region.lines.append(new_textline)
        return page_layout


class LineCropper(object):
    def __init__(self, config):
        poly = config.getint('INTERP')
        line_scale = config.getfloat('LINE_SCALE')
        line_height = config.getint('LINE_HEIGHT')
        self.crop_engine = cropper.EngineLineCropper(line_height=line_height, poly=poly, scale=line_scale)

    def process_page(self, img, page_layout: PageLayout):
        for line in page_layout.lines_iterator():
            try:
                line.crop = self.crop_engine.crop(img, line.baseline, line.heights)
            except ValueError:
                line.crop = np.zeros((self.crop_engine.line_height, self.crop_engine.line_height, 3))
                print(f"WARNING: Failed to crop line {line.id} in page {page_layout.id}. Probably contain vertical line. Contanct Olda Kodym to fix this bug!")
        return page_layout


class PageOCR(object):
    def __init__(self, config):
        json_file = config['OCR_JSON']
        self.ocr_engine = line_ocr_engine.EngineLineOCR(json_file, gpu_id=0)

    def process_page(self, img, page_layout: PageLayout):
        for line in page_layout.lines_iterator():
            if line.crop is None:
                raise Exception(f'Missing crop in line {line.id}.')

        transcriptions, logits = self.ocr_engine.process_lines([line.crop for line in page_layout.lines_iterator()])

        for line, line_transcription, line_logits in zip(page_layout.lines_iterator(), transcriptions, logits):
            line.transcription = line_transcription
            line.logits = line_logits
        return page_layout


class PageParser (object):
    def __init__(self, config):
        self.run_layout_parser = config['PAGE_PARSER'].getboolean('RUN_LAYOUT_PARSER')
        self.run_line_parser = config['PAGE_PARSER'].getboolean('RUN_LINE_PARSER')
        self.run_line_cropper = config['PAGE_PARSER'].getboolean('RUN_LINE_CROPPER')
        self.run_ocr = config['PAGE_PARSER'].getboolean('RUN_OCR')
        self.run_decoder = config['PAGE_PARSER'].getboolean('RUN_DECODER')

        self.layout_parser = None
        self.line_parser = None
        self.line_cropper = None
        self.ocr = None
        self.decoder = None
        if self.run_layout_parser:
            self.layout_parser = layout_parser_factory(config)
        if self.run_line_parser:
            self.line_parser = line_parser_factory(config)
        if self.run_line_cropper:
            self.line_cropper = line_cropper_factory(config)
        if self.run_ocr:
            self.ocr = ocr_factory(config)
        if self.run_decoder:
            self.decoder = page_decoder_factory(config)

    def process_page(self, image, page_layout):
        if self.run_layout_parser:
            page_layout = self.layout_parser.process_page(image, page_layout)
        if self.run_line_parser:
            page_layout = self.line_parser.process_page(image, page_layout)
        if self.run_line_cropper:
            page_layout = self.line_cropper.process_page(image, page_layout)
        if self.run_ocr:
            page_layout = self.ocr.process_page(image, page_layout)
        if self.run_decoder:
            page_layout = self.decoder.process_page(page_layout)

        return page_layout
