import numpy as np

from multiprocessing import Pool
from functools import partial
from scipy import ndimage

from pero_ocr.utils import compose_path

from .layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import crop_engine as cropper
from pero_ocr.ocr_engine import line_ocr_engine
from pero_ocr.line_engine import baseline_engine
from pero_ocr.region_engine import region_engine
from pero_ocr.region_engine import region_engine_splic
from pero_ocr.region_engine import region_engine_simple
from pero_ocr.region_engine import SimpleThresholdRegion
import pero_ocr.line_engine.line_postprocessing as linepp


def layout_parser_factory(config, config_path=''):
    config = config['LAYOUT_PARSER']
    if config['METHOD'] == 'WHOLE_PAGE_REGION':
        region_parser = WholePageRegion(config, config_path=config_path)
    elif config['METHOD'] == 'cnn':
        region_parser = RegionExtractorCNN(config, config_path=config_path)
    elif config['METHOD'] == 'SIMPLE_THRESHOLD_REGION':
        region_parser = SimpleThresholdRegion(config, config_path=config_path)
    elif config['METHOD'] == 'SPLIC':
        region_parser = RegionExtractorSPLIC(config, config_path=config_path)
    elif config['METHOD'] == 'SIMPLE':
        region_parser = RegionExtractorSimple(config, config_path=config_path)
    else:
        raise ValueError('Unknown layout parser method: {}'.format(config['METHOD']))
    return region_parser


def line_parser_factory(config, config_path=''):
    config = config['LINE_PARSER']
    if config['METHOD'] == 'cnn':
        line_parser = TextlineExtractorCNN(config, config_path=config_path)
    elif config['METHOD'] == 'cnn_reg':
        line_parser = TextlineExtractorCNNReg(config, config_path=config_path)
    elif config['METHOD'] == 'simple':
        line_parser = TextlineExtractorSimple(config, config_path=config_path)
    elif config['METHOD'] == 'line_refiner':
        line_parser = LineRefiner(config, config_path=config_path)
    else:
        raise ValueError('Unknown line parser method: {}'.format(config['METHOD']))
    return line_parser


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
    return PageDecoder(decoder)


def region_sorter_factory(config, config_path=''):
    class_name = config['REGION_SORTER']['METHOD']
    imported = __import__('pero_ocr.region_sorter', fromlist=[class_name])

    try:
        object_class = getattr(imported, class_name)
    except AttributeError:
        raise ValueError(f"Region sorter method not found. Ensure \"{class_name}\" is valid region sorter"
                         " class name imported in \"__init__.py\".")

    return object_class(config['REGION_SORTER'], config_path)


class MissingLogits(Exception):
    pass


class PageDecoder:
    def __init__(self, decoder, line_confidence_threshold=None):
        self.decoder = decoder
        self.line_confidence_threshold = line_confidence_threshold

    def process_page(self, page_layout: PageLayout):
        for line in page_layout.lines_iterator():
            logits = self.prepare_dense_logits(line)
            if self.line_confidence_threshold is not None:
                if self.line_confident_enough(logits, self.line_confidence_threshold):
                    continue

            hypotheses = self.decoder(logits)
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


class RegionExtractorCNN(object):
    def __init__(self, config, config_path=''):
        model_path = compose_path(config['MODEL_PATH'], config_path)
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


class RegionExtractorSPLIC(object):
    def __init__(self, config, config_path=''):
        model_path = compose_path(config['MODEL_PATH'], config_path)
        downsample = config.getint('DOWNSAMPLE')
        use_cpu = config.getboolean('USE_CPU')
        min_size = config.getint('MIN_SIZE')
        self.keep_lines = config.getboolean('KEEP_LINES')
        self.region_engine = region_engine_splic.EngineRegionSPLIC(
            model_path=model_path,
            downsample=downsample,
            use_cpu=use_cpu,
            min_size=min_size
        )
        self.pool = Pool()

    def process_page(self, img, page_layout: PageLayout):
        polygons_list, baselines_list, heights_list, textlines_list = self.region_engine.detect(img)
        for id, polygon in enumerate(polygons_list):
            region = RegionLayout('r{:03d}'.format(id), polygon)
            page_layout.regions.append(region)

        if self.keep_lines:
            if len(page_layout.regions) > 4:
                page_layout.regions = list(self.pool.map(partial(assign_lines_to_region, baselines_list, heights_list, textlines_list),
                                 page_layout.regions))
            else:
                for region in page_layout.regions:
                    region = assign_lines_to_region(baselines_list, heights_list, textlines_list, region)

        return page_layout

class RegionExtractorSimple(object):
    def __init__(self, config, config_path=''):
        model_path = compose_path(config['MODEL_PATH'], config_path)
        downsample = config.getint('DOWNSAMPLE')
        use_cpu = config.getboolean('USE_CPU')
        max_mp = config.getfloat('MAX_MEGAPIXELS')
        gpu_fraction = config.getfloat('GPU_FRACTION')
        self.keep_lines = config.getboolean('KEEP_LINES')
        self.region_engine = region_engine_simple.EngineRegionSimple(
            model_path=model_path,
            downsample=downsample,
            use_cpu=use_cpu,
            max_mp=max_mp,
            gpu_fraction=gpu_fraction
        )
        self.pool = Pool()

    def process_page(self, img, page_layout: PageLayout):
        polygons_list, baselines_list, heights_list, textlines_list = self.region_engine.detect(img)
        for id, polygon in enumerate(polygons_list):
            region = RegionLayout('r{:03d}'.format(id), polygon)
            page_layout.regions.append(region)

        if self.keep_lines:
            if len(page_layout.regions) > 4:
                page_layout.regions = list(self.pool.map(partial(assign_lines_to_region, baselines_list, heights_list, textlines_list),
                                 page_layout.regions))
            else:
                for region in page_layout.regions:
                    region = assign_lines_to_region(baselines_list, heights_list, textlines_list, region)

        return page_layout


class LineRefiner(object):
    def __init__(self, config, config_path=''):
        self.model_path = compose_path(config['MODEL_PATH'], config_path)
        self.downsample = config.getint('DOWNSAMPLE')
        self.pad = config.getint('PAD')
        self.use_cpu = config.getboolean('USE_CPU')
        self.line_engine = baseline_engine.EngineLineDetectorCNNReg(
            model_path=self.model_path,
            downsample=self.downsample,
            pad=self.pad,
            use_cpu=self.use_cpu,
            detection_threshold=1
        )
        self.adjust_baselines = config.getboolean('ADJUST_BASELINES')
        self.adjust_heights = config.getboolean('ADJUST_HEIGHTS')

    def process_page(self, img, page_layout: PageLayout):
        if not list(page_layout.lines_iterator()):
            print(f"Warning: Skipping line reninement for page {page_layout.id}. No text lines present.")
            return page_layout

        out_map = self.line_engine.get_maps(img, self.downsample)
        baseline_map, heights_map = out_map[:,:,2], out_map[:,:,:2]

        if self.adjust_baselines:
            baselines = [line.baseline for line in page_layout.lines_iterator()]
            baselines = linepp.adjust_baselines_to_intensity(baselines, img)
            for line, baseline in zip(page_layout.lines_iterator(), baselines):
                line.baseline = baseline

        if self.adjust_heights:
            #heights_map = ndimage.morphology.grey_dilation(heights_map, size=(5, 1, 1))
            for line in page_layout.lines_iterator():
                baseline = line.baseline / self.downsample
                sample_points = linepp.resample_baselines([baseline], num_points=40)[0]
                y_inds = np.clip(np.round(sample_points[:,1]).astype(np.int), 0, heights_map.shape[0]-1)
                x_inds = np.clip(np.round(sample_points[:,0]).astype(np.int), 0, heights_map.shape[1]-1)
                heights_pred = self.line_engine.get_heights(
                    heights_map,
                    (y_inds, x_inds))
                line.heights = heights_pred * self.downsample

        height = np.median([l.heights[0] + l.heights[1] for l in page_layout.lines_iterator()])
        if height / self.downsample <= 6 or height / self.downsample > 18:
            temp_downsample = self.downsample
            print("ADAPT DOWNAMPLING", img.shape[0:2], self.downsample, height, height / self.downsample)
            self.downsample = max(1, int(height / 12 + 0.5))
            ds_threshold = (img.shape[0] * img.shape[1]) / (5 * 10e5)
            if self.downsample < ds_threshold:
                self.downsample = ds_threshold

            self.line_engine.downsample = self.downsample
            out_map = self.line_engine.get_maps(img, self.downsample)
            baseline_map, heights_map = out_map[:,:,2], out_map[:,:,:2]
            self.line_engine.downsample = temp_downsample
            if self.adjust_heights:
                heights_map = ndimage.morphology.grey_dilation(heights_map, size=(11, 1, 1))
                for line in page_layout.lines_iterator():
                    baseline = line.baseline / self.downsample
                    sample_points = linepp.resample_baselines([baseline], num_points=40)[0]
                    y_inds = np.clip(np.round(sample_points[:,1]).astype(np.int), 0, heights_map.shape[0]-1)
                    x_inds = np.clip(np.round(sample_points[:,0]).astype(np.int), 0, heights_map.shape[1]-1)
                    heights_pred = self.line_engine.get_heights(
                        heights_map,
                        (y_inds, x_inds))
                    line.heights = heights_pred * self.downsample


        for line in page_layout.lines_iterator():
            line.polygon = linepp.baseline_to_textline(line.baseline, line.heights)

        return page_layout

def assign_lines_to_region(baseline_list, heights_list, textline_list, region):
    for line_num, (baseline, heights, textline) in enumerate(zip(baseline_list, heights_list, textline_list)):
        baseline_intersection, textline_intersection = linepp.mask_textline_by_region(baseline, textline, region.polygon)
        if baseline_intersection is not None and textline_intersection is not None:
            new_textline = TextLine(id='{}-l{:03d}'.format(region.id, line_num+1), baseline=baseline_intersection, polygon=textline_intersection, heights=heights)
            region.lines.append(new_textline)
    return region

class BaseTextlineExtractor(object):
    def __init__(self, config):
        self.merge_lines = config.getboolean('MERGE_LINES')
        self.stretch_lines = config['STRETCH_LINES']
        if self.stretch_lines != 'max':
            self.stretch_lines = int(self.stretch_lines)
        self.resample_lines = config.getboolean('RESAMPLE_LINES')
        self.order_lines = config['ORDER_LINES']
        self.heights_from_regions = config.getboolean('HEIGHTS_FROM_REGIONS')
        self.pool = Pool()

    def postprocess_region_lines(self, region):
        if region.lines:
            region_baseline_list = [line.baseline for line in region.lines]
            region_textline_list = [line.polygon for line in region.lines]
            region_heights_list = [line.heights for line in region.lines]
            region.lines = []

            rotation = linepp.get_rotation(region_baseline_list)
            region_baseline_list = [linepp.rotate_coords(baseline, rotation, (0, 0)) for baseline in region_baseline_list]

            if self.merge_lines:
                region_baseline_list, region_heights_list = linepp.merge_lines(region_baseline_list, region_heights_list)

            if self.stretch_lines == 'max':
                region_baseline_list = linepp.stretch_baselines_to_region(region_baseline_list, linepp.rotate_coords(region.polygon.copy(), rotation, (0, 0)))
            elif self.stretch_lines > 0:
                region_baseline_list = linepp.stretch_baselines(region_baseline_list, self.stretch_lines)

            if self.resample_lines:
                region_baseline_list = linepp.resample_baselines(region_baseline_list)

            if self.heights_from_regions:
                scores = []
                region_heights_list = []
                for baseline in region_baseline_list:
                    baseline = linepp.rotate_coords(baseline, -rotation, (0, 0))
                    height_asc = int(round(np.amin(baseline[:,1]) - np.amin(region.polygon[:,1])))
                    height_des = int(round(np.amax(region.polygon[:,1]) - np.amax(baseline[:,1])))
                    region_heights_list.append((height_asc, height_des))
                    # the final line in the bounding box should be the longest and in case of ambiguity, also have the biggest ascender height
                    scores.append(np.amax(baseline[:,0]) - np.amin(baseline[:,0]) + height_asc)
                best_ind = np.argmax(np.asarray(scores))
                region_baseline_list = [region_baseline_list[best_ind]]
                region_heights_list = [region_heights_list[best_ind]]

            region_textline_list = []
            for baseline, height in zip(region_baseline_list, region_heights_list):
                region_textline_list.append(linepp.baseline_to_textline(baseline, height))

            if self.order_lines == 'vertical':
                region_baseline_list, region_heights_list, region_textline_list = linepp.order_lines_vertical(region_baseline_list, region_heights_list, region_textline_list)
            elif self.order_lines == 'reading_order':
                region_baseline_list, region_heights_list, region_textline_list = linepp.order_lines_general(region_baseline_list, region_heights_list, region_textline_list)
            else:
                raise ValueError("Argument order_lines must be either 'vertical' or 'reading_order'.")

            region_textline_list = [linepp.rotate_coords(textline, -rotation, (0, 0)) for textline in region_textline_list]
            region_baseline_list = [linepp.rotate_coords(baseline, -rotation, (0, 0)) for baseline in region_baseline_list]

            scores = []
            for line in region.lines:
                width = line.baseline[-1][0] - line.baseline[0][0]
                height = line.heights[0] + line.heights[1]
                scores.append((width - self.stretch_lines) / height)
            region.lines = [line for line, score in zip(region.lines, scores) if score > 0.5]
            region = assign_lines_to_region(region_baseline_list, region_heights_list, region_textline_list, region)

        return region

    def process_page(self, img, page_layout: PageLayout):
        if not page_layout.regions:
            print(f"Warning: Skipping line detection for page {page_layout.id}. No text region present.")
            return page_layout

        baseline_list, heights_list, textline_list = self.line_engine.detect_lines(img)

        if len(page_layout.regions) > 4:
            page_layout.regions = list(self.pool.map(partial(assign_lines_to_region, baseline_list, heights_list, textline_list),
                             page_layout.regions))
        else:
            for region in page_layout.regions:
                region = assign_lines_to_region(baseline_list, heights_list, textline_list, region)

        for region in page_layout.regions:
            region = self.postprocess_region_lines(region)

        return page_layout


class TextlineExtractorCNN(BaseTextlineExtractor):
    def __init__(self, config, config_path=''):
        super(TextlineExtractorCNN, self).__init__(config)
        model_path = compose_path(config['MODEL_PATH'], config_path)
        downsample = config.getint('DOWNSAMPLE')
        pad = config.getint('PAD')
        use_cpu = config.getboolean('USE_CPU')
        detection_threshold = config.getfloat('DETECTION_THRESHOLD')
        self.line_engine = baseline_engine.EngineLineDetectorCNN(
            model_path=model_path,
            downsample=downsample,
            pad=pad,
            use_cpu=use_cpu,
            detection_threshold=detection_threshold,
        )

class TextlineExtractorCNNReg(BaseTextlineExtractor):
    def __init__(self, config, config_path=''):
        super(TextlineExtractorCNNReg, self).__init__(config)
        model_path = compose_path(config['MODEL_PATH'], config_path)
        downsample = config.getint('DOWNSAMPLE')
        pad = config.getint('PAD')
        use_cpu = config.getboolean('USE_CPU')
        detection_threshold = config.getfloat('DETECTION_THRESHOLD')
        max_mp = config.getfloat('MAX_MEGAPIXELS')
        gpu_fraction = config.getfloat('GPU_FRACTION')
        self.line_engine = baseline_engine.EngineLineDetectorCNNReg(
            model_path=model_path,
            downsample=downsample,
            pad=pad,
            use_cpu=use_cpu,
            detection_threshold=detection_threshold,
            max_mp=max_mp,
            gpu_fraction=gpu_fraction
        )


class TextlineExtractorSimple(object):
    def __init__(self, config, config_path=''):
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
    def __init__(self, config, config_path=''):
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

        transcriptions, logits = self.ocr_engine.process_lines([line.crop for line in page_layout.lines_iterator()])

        for line, line_transcription, line_logits in zip(page_layout.lines_iterator(), transcriptions, logits):
            line.transcription = line_transcription
            line.logits = line_logits
            line.characters = self.ocr_engine.characters
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
        self.run_line_parser = config['PAGE_PARSER'].getboolean('RUN_LINE_PARSER', fallback=False)
        self.run_line_cropper = config['PAGE_PARSER'].getboolean('RUN_LINE_CROPPER', fallback=False)
        self.run_ocr = config['PAGE_PARSER'].getboolean('RUN_OCR', fallback=False)
        self.run_decoder = config['PAGE_PARSER'].getboolean('RUN_DECODER', fallback=False)
        self.run_region_sorter = config['PAGE_PARSER'].getboolean('RUN_REGION_SORTER', fallback=False)
        self.filter_confident_lines_threshold = config['PAGE_PARSER'].getfloat('FILTER_CONFIDENT_LINES_THRESHOLD',
                                                                                 fallback=-1)

        self.layout_parser = None
        self.line_parser = None
        self.line_cropper = None
        self.ocr = None
        self.decoder = None
        self.region_sorter = None

        if self.run_layout_parser:
            self.layout_parser = layout_parser_factory(config, config_path=config_path)
        if self.run_region_sorter:
            self.region_sorter = region_sorter_factory(config, config_path=config_path)

        if self.run_line_parser:
            self.line_parser = line_parser_factory(config, config_path=config_path)
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
        print(worst_best_prob, np.sum(np.exp(best_probs) < threshold), best_probs.shape,  np.nonzero(np.exp(best_probs) < threshold))
        #for i in np.nonzero(np.exp(best_probs) < threshold)[0]:
        #    print(best_probs[i-1:i+2], best_ids[i-1:i+2])

        return worst_best_prob

    def filter_confident_lines(self, page_layout):
        for region in page_layout.regions:
            region.lines = [line for line in region.lines
                            if PageParser.compute_line_confidence(line, self.filter_confident_lines_threshold) > self.filter_confident_lines_threshold]
        return page_layout

    def process_page(self, image, page_layout):
        if self.run_layout_parser:
            page_layout = self.layout_parser.process_page(image, page_layout)
        if self.run_region_sorter:
            page_layout = self.region_sorter.process_page(image, page_layout)
        if self.run_line_parser:
            page_layout = self.line_parser.process_page(image, page_layout)
        if self.run_line_cropper:
            page_layout = self.line_cropper.process_page(image, page_layout)
        if self.run_ocr:
            page_layout = self.ocr.process_page(image, page_layout)
        if self.run_decoder:
            page_layout = self.decoder.process_page(page_layout)
        if self.filter_confident_lines_threshold > 0:
            page_layout = self.filter_confident_lines(page_layout)

        return page_layout
