import logging
import re
import pickle
import json
from io import BytesIO
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union

import numpy as np
import lxml.etree as ET
import cv2
from shapely.geometry import LineString, Polygon
import scipy

from pero_ocr.core.crop_engine import EngineLineCropper
from pero_ocr.core.force_alignment import align_text
from pero_ocr.core.confidence_estimation import get_line_confidence
from pero_ocr.core.arabic_helper import ArabicHelper

Num = Union[int, float]


logger = logging.getLogger(__name__)


class PAGEVersion(Enum):
    PAGE_2019_07_15 = 1
    PAGE_2013_07_15 = 2


def log_softmax(x):
    a = np.logaddexp.reduce(x, axis=1)[:, np.newaxis]
    return x - a


def export_id(id, validate_change_id):
    return 'id_' + id if validate_change_id else id


class TextLine(object):
    def __init__(self, id: str = None,
                 baseline: Optional[np.ndarray] = None,
                 polygon: Optional[np.ndarray] = None,
                 heights: Optional[np.ndarray] = None,
                 transcription: Optional[str] = None,
                 logits: Optional[Union[scipy.sparse.csc_matrix, np.ndarray]] = None,
                 crop: Optional[np.ndarray] = None,
                 characters: Optional[list[str]] = None,
                 logit_coords: Optional[Union[list[int, int], list[None, None]]] = None,
                 transcription_confidence: Optional[Num] = None,
                 index: Optional[int] = None):
        self.id = id
        self.index = index
        self.baseline = baseline
        self.polygon = polygon
        self.heights = heights
        self.transcription = transcription
        self.logits = logits
        self.crop = crop
        self.characters = characters
        self.logit_coords = logit_coords
        self.transcription_confidence = transcription_confidence

    def get_dense_logits(self, zero_logit_value: int = -80):
        dense_logits = self.logits.toarray()
        dense_logits[dense_logits == 0] = zero_logit_value
        return dense_logits

    def get_full_logprobs(self, zero_logit_value: int = -80):
        dense_logits = self.get_dense_logits(zero_logit_value)
        return log_softmax(dense_logits)


class RegionLayout(object):
    def __init__(self, id: str,
                 polygon: np.ndarray,
                 region_type=None):
        self.id = id  # ID string
        self.polygon = polygon  # bounding polygon
        self.region_type = region_type
        self.lines: list[TextLine] = []
        self.transcription = None

    def to_page_xml(self, page_element: ET.SubElement, validate_id: bool = False):
        region_element = ET.SubElement(page_element, "TextRegion")
        coords = ET.SubElement(region_element, "Coords")
        region_element.set("id", export_id(self.id, validate_id))

        if self.region_type is not None:
            region_element.set("type", self.region_type)

        points = ["{},{}".format(int(np.round(coord[0])), int(np.round(coord[1]))) for coord in self.polygon]
        points = " ".join(points)
        coords.set("points", points)
        if self.transcription is not None:
            text_element = ET.SubElement(region_element, "TextEquiv")
            text_element = ET.SubElement(text_element, "Unicode")
            text_element.text = self.transcription
        return region_element


def get_coords_form_page_xml(coords_element, schema):
    if 'points' in coords_element.attrib:
        coords = points_string_to_array(coords_element.attrib['points'])
    else:
        coords = []
        for point in coords_element.findall(schema + 'Point'):
            x, y = point.attrib['x'], point.attrib['y']
            coords.append([float(x), float(y)])
        coords = np.asarray(coords)
    return coords


def get_region_from_page_xml(region_element, schema):
    coords_element = region_element.find(schema + 'Coords')
    region_coords = get_coords_form_page_xml(coords_element, schema)

    region_type = None
    if "type" in region_element.attrib:
        region_type = region_element.attrib["type"]

    layout_region = RegionLayout(region_element.attrib['id'], region_coords, region_type)

    transcription = region_element.find(schema + 'TextEquiv')
    if transcription is not None:
        layout_region.transcription = transcription.find(schema + 'Unicode').text
        if layout_region.transcription is None:
            layout_region.transcription = ''
    return layout_region


def guess_line_heights_from_polygon(text_line: TextLine, use_center: bool = False, n: int = 10, interpolate=False):
    '''
    Guess line heights for line if missing (e.g. import from Transkribus).
    Heights are computed from polygon intersection with baseline normal in the middle of baseline.
    '''
    try:
        heights_up = []
        heights_down = []
        points = []

        if use_center:
            if text_line.baseline.shape[0] % 2 == 0:
                center = (text_line.baseline[text_line.baseline.shape[0]//2 - 1] + text_line.baseline[text_line.baseline.shape[0]//2]) / 2
            else:
                center = text_line.baseline[text_line.baseline.shape[0]//2]

            points = [center]
            n -= 1

        replace = len(text_line.baseline) < n

        if interpolate:
            points_per_segment = int(n / len(text_line.baseline))

            for start_point, end_point in zip(text_line.baseline[:-1], text_line.baseline[1:]):
                points.append(np.linspace(start_point, end_point, points_per_segment, endpoint=False))

            points.append(text_line.baseline[-1])

        else:
            points += text_line.baseline[np.random.choice(text_line.baseline.shape[0], n, replace=replace), :].tolist()

        for point in points:
            heights = guess_height_at_point(text_line, point)
            if heights is None:
                continue

            up, down = heights
            heights_up.append(up)
            heights_down.append(down)

        if len(heights_up) > 0:
            height_up = np.mean(heights_up)
            height_down = np.mean(heights_down)

        else:
            height_up, height_down = guess_height_simple(text_line)

    except:
        height_up, height_down = guess_height_simple(text_line)

    text_line.heights = [height_up, height_down]


def guess_height_simple(text_line: TextLine):
    height = text_line.polygon[:, 1].max() - text_line.polygon[:, 1].min()
    return [height * 0.8, height * 0.2]


def guess_height_at_point(text_line: TextLine, point):
    direction = text_line.baseline[0] - text_line.baseline[-1]
    direction = direction[::-1]
    direction[0] = -direction[0]
    cross_line = np.stack([point - direction * 10, point + direction * 10])

    cross_line = LineString(cross_line)
    polygon = Polygon(text_line.polygon)
    intersection = polygon.intersection(cross_line)

    if type(intersection) == LineString:
        intersection = np.asarray(intersection.coords.xy).T
    else:
        return None

    if len(intersection) == 0:
        return None

    if intersection[0][1] < intersection[1][1]:
        intersection_above = intersection[0]
        intersection_below = intersection[1]
    else:
        intersection_above = intersection[1]
        intersection_below = intersection[0]

    heights = [((point - intersection_above) ** 2).sum() ** 0.5, ((point - intersection_below) ** 2).sum() ** 0.5]
    return heights


def get_reading_order(page_element, schema):
    reading_order = {}

    for reading_order_element in page_element.iter(schema + "ReadingOrder"):
        for ordered_group_element in reading_order_element.iter(schema + "OrderedGroup"):
            for indexed_region_element in ordered_group_element.iter(schema + "RegionRefIndexed"):
                region_index = int(indexed_region_element.attrib["index"])
                region_id = indexed_region_element.attrib["regionRef"]
                reading_order[region_id] = region_index

    return reading_order


class PageLayout(object):
    def __init__(self, id: str = None, page_size: list[int, int] = (0, 0), file: str = None):
        self.id = id
        self.page_size = page_size  # (height, width)
        self.regions: list[RegionLayout] = []
        self.reading_order = None

        if file is not None:
            self.from_pagexml(file)

        if self.reading_order is not None and len(self.regions) > 0:
            self.sort_regions_by_reading_order()

    def from_pagexml_string(self, pagexml_string: str):
        self.from_pagexml(BytesIO(pagexml_string.encode('utf-8')))

    def from_pagexml(self, file: Union[str, BytesIO]):
        page_tree = ET.parse(file)
        schema = element_schema(page_tree.getroot())

        page = page_tree.findall(schema + 'Page')[0]
        self.id = page.attrib['imageFilename']
        self.page_size = (int(page.attrib['imageHeight']), int(page.attrib['imageWidth']))

        self.reading_order = get_reading_order(page, schema)

        for region in page_tree.iter(schema + 'TextRegion'):
            region_layout = get_region_from_page_xml(region, schema)

            for line_i, line in enumerate(region.iter(schema + 'TextLine')):
                new_textline = TextLine(id=line.attrib['id'])
                if 'custom' in line.attrib:
                    custom_str = line.attrib['custom']
                    if 'heights_v2' in custom_str:
                        for word in custom_str.split():
                            if 'heights_v2' in word:
                                new_textline.heights = json.loads(word.split(":")[1])
                    else:
                        if re.findall("heights", line.attrib['custom']):
                            heights = re.findall("\d+", line.attrib['custom'])
                            heights_array = np.asarray([float(x) for x in heights])
                            if heights_array.shape[0] == 4:
                                heights = np.zeros(2, dtype=np.float32)
                                heights[0] = heights_array[0]
                                heights[1] = heights_array[2]
                            elif heights_array.shape[0] == 3:
                                heights = np.zeros(2, dtype=np.float32)
                                heights[0] = heights_array[1]
                                heights[1] = heights_array[2] - heights_array[0]
                            else:
                                heights = heights_array
                            new_textline.heights = heights.tolist()

                if 'index' in line.attrib:
                    try:
                        new_textline.index = int(line.attrib['index'])
                    except ValueError:
                        pass

                if new_textline.index is None:
                    new_textline.index = line_i

                baseline = line.find(schema + 'Baseline')
                if baseline is not None:
                    new_textline.baseline = get_coords_form_page_xml(baseline, schema)
                else:
                    logger.warning(f'Warning: Baseline is missing in TextLine. '
                                   f'Skipping this line during import. Line ID: {new_textline.id} Page ID: {self.id}')
                    continue

                textline = line.find(schema + 'Coords')
                if textline is not None:
                    new_textline.polygon = get_coords_form_page_xml(textline, schema)

                if not new_textline.heights:
                    guess_line_heights_from_polygon(new_textline, use_center=False, n=len(new_textline.baseline))

                transcription = line.find(schema + 'TextEquiv')
                if transcription is not None:
                    t_unicode = transcription.find(schema + 'Unicode').text
                    if t_unicode is None:
                        t_unicode = ''
                    new_textline.transcription = t_unicode
                    conf = transcription.get('conf', None)
                    new_textline.transcription_confidence = float(conf) if conf is not None else None
                region_layout.lines.append(new_textline)

            self.regions.append(region_layout)

    def to_pagexml_string(self, creator: str = 'Pero OCR', validate_id: bool = False,
                          version: PAGEVersion = PAGEVersion.PAGE_2019_07_15):
        if version == PAGEVersion.PAGE_2019_07_15:
            attr_qname = ET.QName("http://www.w3.org/2001/XMLSchema-instance", "schemaLocation")
            root = ET.Element(
                'PcGts',
                {attr_qname: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd'},
                nsmap={
                    None: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15',
                    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                    })

            metadata = ET.SubElement(root, "Metadata")
            ET.SubElement(metadata, "Creator").text = creator
            now = datetime.now(timezone.utc)
            ET.SubElement(metadata, "Created").text = now.isoformat()
            ET.SubElement(metadata, "LastChange").text = now.isoformat()

        elif version == PAGEVersion.PAGE_2013_07_15:
            root = ET.Element("PcGts")
            root.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")

        else:
            raise ValueError(f"Unknown PAGE Version: '{version}'")

        page = ET.SubElement(root, "Page")
        page.set("imageFilename", self.id)
        page.set("imageWidth", str(self.page_size[1]))
        page.set("imageHeight", str(self.page_size[0]))

        if self.reading_order is not None:
            self.sort_regions_by_reading_order()
            self.reading_order_to_page_xml(page)

        for region_layout in self.regions:
            text_region = region_layout.to_page_xml(page, validate_id=validate_id)

            for i, line in enumerate(region_layout.lines):
                text_line = ET.SubElement(text_region, "TextLine")
                text_line.set("id", export_id(line.id, validate_id))
                if line.index is not None:
                    text_line.set("index", f'{line.index:d}')
                else:
                    text_line.set("index", f'{i:d}')
                if line.heights is not None:
                    text_line.set("custom", f"heights_v2:[{line.heights[0]:.1f},{line.heights[1]:.1f}]")

                coords = ET.SubElement(text_line, "Coords")

                if line.polygon is not None:
                    points = ["{},{}".format(int(np.round(coord[0])), int(np.round(coord[1]))) for coord in
                              line.polygon]
                    points = " ".join(points)
                    coords.set("points", points)

                if line.baseline is not None:
                    baseline_element = ET.SubElement(text_line, "Baseline")
                    points = ["{},{}".format(int(np.round(coord[0])), int(np.round(coord[1]))) for coord in
                              line.baseline]
                    points = " ".join(points)
                    baseline_element.set("points", points)

                if line.transcription is not None:
                    text_element = ET.SubElement(text_line, "TextEquiv")
                    if line.transcription_confidence is not None:
                        text_element.set("conf", f"{line.transcription_confidence:.3f}")
                    text_element = ET.SubElement(text_element, "Unicode")
                    text_element.text = line.transcription

        return ET.tostring(root, pretty_print=True, encoding="utf-8", xml_declaration=True).decode("utf-8")

    def to_pagexml(self, file_name: str, creator: str = 'Pero OCR',
                   validate_id: bool = False, version: PAGEVersion = PAGEVersion.PAGE_2019_07_15):
        xml_string = self.to_pagexml_string(version=version, creator=creator, validate_id=validate_id)
        with open(file_name, 'w', encoding='utf-8') as out_f:
            out_f.write(xml_string)

    def to_altoxml_string(self, ocr_processing_element: ET.SubElement = None, page_uuid: str = None, min_line_confidence: float = 0):
        arabic_helper = ArabicHelper()
        NSMAP = {"xlink": 'http://www.w3.org/1999/xlink',
                 "xsi": 'http://www.w3.org/2001/XMLSchema-instance'}
        root = ET.Element("alto", nsmap=NSMAP)
        root.set("xmlns", "http://www.loc.gov/standards/alto/ns-v2#")

        description = ET.SubElement(root, "Description")
        measurement_unit = ET.SubElement(description, "MeasurementUnit")
        measurement_unit.text = "pixel"
        source_image_information = ET.SubElement(description, "sourceImageInformation")
        file_name = ET.SubElement(source_image_information, "fileName")
        file_name.text = self.id
        if ocr_processing_element is not None:
            description.append(ocr_processing_element)
        else:
            ocr_processing_element = create_ocr_processing_element()
            description.append(ocr_processing_element)
        layout = ET.SubElement(root, "Layout")
        page = ET.SubElement(layout, "Page")
        if page_uuid is not None:
            page.set("ID", "id_" + page_uuid)
        else:
            page.set("ID", "id_" + re.sub('[!\"#$%&\'()*+,/:;<=>?@[\\]^`{|}~ ]', '_', self.id))
        page.set("PHYSICAL_IMG_NR", str(1))
        page.set("HEIGHT", str(self.page_size[0]))
        page.set("WIDTH", str(self.page_size[1]))

        top_margin = ET.SubElement(page, "TopMargin")
        left_margin = ET.SubElement(page, "LeftMargin")
        right_margin = ET.SubElement(page, "RightMargin")
        bottom_margin = ET.SubElement(page, "BottomMargin")
        print_space = ET.SubElement(page, "PrintSpace")

        print_space_height = 0
        print_space_width = 0
        print_space_vpos = self.page_size[0]
        print_space_hpos = self.page_size[1]

        for b, block in enumerate(self.regions):
            text_block = ET.SubElement(print_space, "TextBlock")
            text_block.set("ID", 'block_{}' .format(block.id))

            text_block_height, text_block_width, text_block_vpos, text_block_hpos = get_hwvh(block.polygon)
            text_block.set("HEIGHT", str(int(text_block_height)))
            text_block.set("WIDTH", str(int(text_block_width)))
            text_block.set("VPOS", str(int(text_block_vpos)))
            text_block.set("HPOS", str(int(text_block_hpos)))

            print_space_height = max([print_space_vpos + print_space_height, text_block_vpos + text_block_height])
            print_space_width = max([print_space_hpos + print_space_width, text_block_hpos + text_block_width])
            print_space_vpos = min([print_space_vpos, text_block_vpos])
            print_space_hpos = min([print_space_hpos, text_block_hpos])
            print_space_height = print_space_height - print_space_vpos
            print_space_width = print_space_width - print_space_hpos

            for l, line in enumerate(block.lines):
                if not line.transcription or line.transcription.strip() == "":
                    continue
                arabic_line = False
                if arabic_helper.is_arabic_line(line.transcription):
                    arabic_line = True
                text_line = ET.SubElement(text_block, "TextLine")
                text_line_baseline = int(np.average(np.array(line.baseline)[:, 1]))
                text_line.set("BASELINE", str(text_line_baseline))

                text_line_height, text_line_width, text_line_vpos, text_line_hpos = get_hwvh(line.polygon)

                text_line.set("VPOS", str(int(text_line_vpos)))
                text_line.set("HPOS", str(int(text_line_hpos)))
                text_line.set("HEIGHT", str(int(text_line_height)))
                text_line.set("WIDTH", str(int(text_line_width)))

                try:
                    chars = [i for i in range(len(line.characters))]
                    char_to_num = dict(zip(line.characters, chars))

                    blank_idx = line.logits.shape[1] - 1

                    label = []
                    for item in line.transcription:
                        if item in char_to_num.keys():
                            if char_to_num[item] >= blank_idx:
                                label.append(0)
                            else:
                                label.append(char_to_num[item])
                        else:
                            label.append(0)

                    logits = line.get_dense_logits()[line.logit_coords[0]:line.logit_coords[1]]
                    logprobs = line.get_full_logprobs()[line.logit_coords[0]:line.logit_coords[1]]
                    aligned_letters = align_text(-logprobs, np.array(label), blank_idx)
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(f'Error: Alto export, unable to align line {line.id} due to exception {e}.')
                    line.transcription_confidence = 0
                    average_word_width = (text_line_hpos + text_line_width) / len(line.transcription.split())
                    for w, word in enumerate(line.transcription.split()):
                        string = ET.SubElement(text_line, "String")
                        string.set("CONTENT", word)

                        string.set("HEIGHT", str(int(text_line_height)))
                        string.set("WIDTH", str(int(average_word_width)))
                        string.set("VPOS", str(int(text_line_vpos)))
                        string.set("HPOS", str(int(text_line_hpos + (w * average_word_width))))
                else:
                    crop_engine = EngineLineCropper(poly=2)
                    line_coords = crop_engine.get_crop_inputs(line.baseline, line.heights, 16)
                    space_idxs = [pos for pos, char in enumerate(line.transcription) if char == ' ']

                    words = []
                    space_idxs = [-1] + space_idxs + [len(aligned_letters)]
                    for i in range(len(space_idxs[1:])):
                        if space_idxs[i] != space_idxs[i+1]-1:
                            words.append([aligned_letters[space_idxs[i]+1], aligned_letters[space_idxs[i+1]-1]])
                    splitted_transcription = line.transcription.split()
                    lm_const = line_coords.shape[1] / logits.shape[0]
                    letter_counter = 0
                    confidences = get_line_confidence(line, np.array(label), aligned_letters, logprobs)
                    #if line.transcription_confidence is None:
                    line.transcription_confidence = np.quantile(confidences, .50)
                    for w, word in enumerate(words):
                        extension = 2
                        while line_coords.size > 0 and extension < 40:
                            all_x = line_coords[:, max(0, int((words[w][0]-extension) * lm_const)):int((words[w][1]+extension) * lm_const), 0]
                            all_y = line_coords[:, max(0, int((words[w][0]-extension) * lm_const)):int((words[w][1]+extension) * lm_const), 1]

                            if all_x.size == 0 or all_y.size == 0:
                                extension += 1
                            else:
                                break

                        if line_coords.size == 0 or all_x.size == 0 or all_y.size == 0:
                           all_x = line.baseline[:, 0]
                           all_y = np.concatenate([line.baseline[:, 1] - line.heights[0], line.baseline[:, 1] + line.heights[1]])

                        word_confidence = None
                        if line.transcription_confidence == 1:
                            word_confidence = 1
                        else:
                            if confidences.size != 0:
                                word_confidence = np.quantile(confidences[letter_counter:letter_counter+len(splitted_transcription[w])], .50)

                        string = ET.SubElement(text_line, "String")

                        if arabic_line:
                            string.set("CONTENT", arabic_helper.label_form_to_string(splitted_transcription[w]))
                        else:
                            string.set("CONTENT", splitted_transcription[w])

                        string.set("HEIGHT", str(int((np.max(all_y) - np.min(all_y)))))
                        string.set("WIDTH", str(int((np.max(all_x) - np.min(all_x)))))
                        string.set("VPOS", str(int(np.min(all_y))))
                        string.set("HPOS", str(int(np.min(all_x))))

                        if word_confidence is not None:
                            string.set("WC", str(round(word_confidence, 2)))

                        if w != (len(line.transcription.split())-1):
                            space = ET.SubElement(text_line, "SP")

                            space.set("WIDTH", str(4))
                            space.set("VPOS", str(int(np.min(all_y))))
                            space.set("HPOS", str(int(np.max(all_x))))
                        letter_counter += len(splitted_transcription[w])+1
                if line.transcription_confidence is not None:
                    if line.transcription_confidence < min_line_confidence:
                        text_block.remove(text_line)
        top_margin.set("HEIGHT", "{}" .format(int(print_space_vpos)))
        top_margin.set("WIDTH", "{}" .format(int(self.page_size[1])))
        top_margin.set("VPOS", "0")
        top_margin.set("HPOS", "0")

        left_margin.set("HEIGHT", "{}" .format(int(self.page_size[0])))
        left_margin.set("WIDTH", "{}" .format(int(print_space_hpos)))
        left_margin.set("VPOS", "0")
        left_margin.set("HPOS", "0")

        right_margin.set("HEIGHT", "{}" .format(int(self.page_size[0])))
        right_margin.set("WIDTH", "{}" .format(int(self.page_size[1] - (print_space_hpos + print_space_width))))
        right_margin.set("VPOS", "0")
        right_margin.set("HPOS", "{}" .format(int(print_space_hpos + print_space_width)))

        bottom_margin.set("HEIGHT", "{}" .format(int(self.page_size[0] - (print_space_vpos + print_space_height))))
        bottom_margin.set("WIDTH", "{}" .format(int(self.page_size[1])))
        bottom_margin.set("VPOS", "{}" .format(int(print_space_vpos + print_space_height)))
        bottom_margin.set("HPOS", "0")

        print_space.set("HEIGHT", str(int(print_space_height)))
        print_space.set("WIDTH", str(int(print_space_width)))
        print_space.set("VPOS", str(int(print_space_vpos)))
        print_space.set("HPOS", str(int(print_space_hpos)))

        return ET.tostring(root, pretty_print=True, encoding="utf-8", xml_declaration=True).decode("utf-8")

    def to_altoxml(self, file_name: str, ocr_processing_element: ET.SubElement = None, page_uuid: str = None):
        alto_string = self.to_altoxml_string(ocr_processing_element=ocr_processing_element, page_uuid=page_uuid)
        with open(file_name, 'w', encoding='utf-8') as out_f:
            out_f.write(alto_string)

    def from_altoxml_string(self, altoxml_string: str):
        self.from_altoxml(BytesIO(altoxml_string.encode('utf-8')))

    def from_altoxml(self, file: Union[str, BytesIO]):
        page_tree = ET.parse(file)
        schema = element_schema(page_tree.getroot())
        root = page_tree.getroot()

        layout = root.findall(schema + 'Layout')[0]
        page = layout.findall(schema + 'Page')[0]

        self.id = page.attrib['ID'][3:]
        self.page_size = (int(page.attrib['HEIGHT']), int(page.attrib['WIDTH']))

        print_space = page.findall(schema + 'PrintSpace')[0]
        for region in print_space.iter(schema + 'TextBlock'):
            region_coords = list()
            region_coords.append([int(region.get('HPOS')), int(region.get('VPOS'))])
            region_coords.append([int(region.get('HPOS')) + int(region.get('WIDTH')), int(region.get('VPOS'))])
            region_coords.append([int(region.get('HPOS')) + int(region.get('WIDTH')),
                                  int(region.get('VPOS')) + int(region.get('HEIGHT'))])
            region_coords.append([int(region.get('HPOS')), int(region.get('VPOS')) + int(region.get('HEIGHT'))])

            region_layout = RegionLayout(region.attrib['ID'], np.asarray(region_coords).tolist())

            for line in region.iter(schema + 'TextLine'):
                new_textline = TextLine(baseline=np.asarray(
                    [[int(line.attrib['HPOS']), int(line.attrib['BASELINE'])],
                       [int(line.attrib['HPOS']) + int(line.attrib['WIDTH']), int(line.attrib['BASELINE'])]]))
                polygon = []
                new_textline.heights = np.asarray([
                    int(line.attrib['HEIGHT']) + int(line.attrib['VPOS']) - int(line.attrib['BASELINE']),
                    int(line.attrib['BASELINE']) - int(line.attrib['VPOS'])])
                polygon.append([int(line.attrib['HPOS']), int(line.attrib['VPOS'])])
                polygon.append(
                    [int(line.attrib['HPOS']) + int(line.attrib['WIDTH']), int(line.attrib['VPOS'])])
                polygon.append([int(line.attrib['HPOS']) + int(line.attrib['WIDTH']),
                                             int(line.attrib['VPOS']) + int(line.attrib['HEIGHT'])])
                polygon.append(
                    [int(line.attrib['HPOS']), int(line.attrib['VPOS']) + int(line.attrib['HEIGHT'])])
                new_textline.polygon = np.asarray(polygon)
                word = ''
                start = True
                for text in line.iter(schema + 'String'):
                    if start:
                        start = False
                        word = word + text.get('CONTENT')
                    else:
                        word = word + " " + text.get('CONTENT')
                new_textline.transcription = word
                region_layout.lines.append(new_textline)

            self.regions.append(region_layout)

    def sort_regions_by_reading_order(self):
        self.regions = sorted(self.regions, key=lambda k: self.reading_order[k] if k in self.reading_order else float("inf"))

    def reading_order_to_page_xml(self, page_element: ET.SubElement):
        reading_order_element = ET.SubElement(page_element, "ReadingOrder")
        ordered_group_element = ET.SubElement(reading_order_element, "OrderedGroup")
        ordered_group_element.set("id", "reading_order")

        for region_id, region_index in self.reading_order.items():
            indexed_region_element = ET.SubElement(ordered_group_element, "RegionRefIndexed")
            indexed_region_element.set("regionRef", region_id)
            indexed_region_element.set("index", str(region_index))

    def _gen_logits(self, missing_line_logits_ok=False):
        """
        Generates logits as dictionary of sparse matrices
        :return: logit dictionary
        """
        logits = []
        characters = []
        logit_coords = []
        for region in self.regions:
            for line in region.lines:
                if missing_line_logits_ok and \
                        (line.logits is None or line.characters is None or line.logit_coords is None):
                    continue
                if line.logits is None:
                    raise Exception(f'Missing logits for line {line.id}.')
                if line.characters is None:
                    raise Exception(f'Missing logits mapping to characters for line {line.id}.')
                if line.logit_coords is None:
                    raise Exception(f'Missing logits coords for line {line.id}.')
            logits += [(line.id, line.logits) for line in region.lines]
            characters += [(line.id, line.characters) for line in region.lines]
            logit_coords += [(line.id, line.logit_coords) for line in region.lines]
        logits_dict = dict(logits)
        logits_dict['line_characters'] = dict(characters)
        logits_dict['logit_coords'] = dict(logit_coords)
        return logits_dict

    def save_logits(self, file_name: str, missing_line_logits_ok=False):
        """Save page logits as a pickled dictionary of sparse matrices.
        :param file_name: to pickle into.
        """
        logits_dict = self._gen_logits(missing_line_logits_ok=missing_line_logits_ok)
        with open(file_name, 'wb') as f:
            pickle.dump(logits_dict, f, protocol=4)

    def save_logits_bytes(self, missing_line_logits_ok=False):
        """
        Return page logits as pickled dictionary bytes.
        :return: pickled logits as bytes like object
        """
        logist_dict = self._gen_logits(missing_line_logits_ok=missing_line_logits_ok)
        return pickle.dumps(logist_dict, protocol=pickle.HIGHEST_PROTOCOL)

    def load_logits(self, file: str):
        """Load pagelogits as a pickled dictionary of sparse matrices.
        :param file: file name to pickle into, or already loaded bytes like object
        """
        if isinstance(file, bytes):
            logits_dict = pickle.loads(file)
        else:
            with open(file, 'rb') as f:
                logits_dict = pickle.load(f)

        if 'line_characters' in logits_dict:
            characters = logits_dict['line_characters']
        else:
            characters = dict([(k, None) for k in logits_dict])

        if 'logit_coords' in logits_dict:
            logit_coords = logits_dict['logit_coords']
        else:
            logit_coords = dict([(k, [None, None]) for k in logits_dict])

        for region in self.regions:
            for line in region.lines:
                if line.id not in logits_dict:
                    continue
                line.logits = logits_dict[line.id]
                line.characters = characters[line.id]
                line.logit_coords = logit_coords[line.id]

    def render_to_image(self, image, thickness: int = 2, circles: bool = True, render_order: bool = False):
        """Render layout into image.
        :param image: image to render layout into
        """
        for region_layout in self.regions:
            image = draw_lines(
                image,
                [line.baseline for line in region_layout.lines if line.baseline is not None], color=(0, 0, 255),
                circles=(circles, circles, False), thickness=thickness)
            image = draw_lines(
                image,
                [line.polygon for line in region_layout.lines if line.polygon is not None], color=(0, 255, 0),
                close=True, thickness=thickness)
            image = draw_lines(
                image,
                [region_layout.polygon], color=(255, 0, 0), circles=(circles, circles, circles), close=True,
                thickness=thickness)

        if render_order:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 4
            font_thickness = 5

            for idx, region in enumerate(self.regions):
                min = region.polygon.min(axis=0)
                max = region.polygon.max(axis=0)

                text_w, text_h = cv2.getTextSize(f"{idx}", font, font_scale, font_thickness)[0]

                mid_coords = (int((min[0] + max[0]) // 2 - text_w // 2), int((min[1] + max[1]) // 2 + text_h // 2))

                cv2.putText(image, f"{idx}", mid_coords, font, font_scale,
                            (0, 0, 0), thickness=font_thickness, lineType=cv2.LINE_AA)

        return image

    def lines_iterator(self):
        for region in self.regions:
            for line in region.lines:
                yield line

    def get_quality(self, x: int = None, y: int = None, width: int = None, height: int = None, power: int = 6):
        bbox_confidences = []
        for b, block in enumerate(self.regions):
            for l, line in enumerate(block.lines):
                if not line.transcription:
                    continue

                chars = [i for i in range(len(line.characters))]
                char_to_num = dict(zip(line.characters, chars))

                blank_idx = line.logits.shape[1] - 1

                label = []
                for item in line.transcription:
                    if item in char_to_num.keys():
                        if char_to_num[item] >= blank_idx:
                            label.append(0)
                        else:
                            label.append(char_to_num[item])
                    else:
                        label.append(0)

                logits = line.get_dense_logits()[line.logit_coords[0]:line.logit_coords[1]]
                logprobs = line.get_full_logprobs()[line.logit_coords[0]:line.logit_coords[1]]
                try:
                    aligned_letters = align_text(-logprobs, np.array(label), blank_idx)
                except (ValueError, IndexError) as e:
                    pass
                else:
                    crop_engine = EngineLineCropper(poly=2)
                    line_coords = crop_engine.get_crop_inputs(line.baseline, line.heights, 16)
                    space_idxs = [pos for pos, char in enumerate(line.transcription) if char == ' ']

                    words = []
                    space_idxs = [-1] + space_idxs + [len(aligned_letters)]

                    only_letters = dict()
                    counter = 0
                    for i, letter in enumerate(aligned_letters):
                        if i not in space_idxs:
                            words.append([letter, letter])
                            only_letters[counter] = i
                            counter += 1

                    lm_const = line_coords.shape[1] / logits.shape[0]
                    confidences = get_line_confidence(line, np.array(label), aligned_letters, logprobs)
                    line.transcription_confidence = np.quantile(confidences, .50)
                    for w, word in enumerate(words):
                        extension = 2
                        while True:
                            all_x = line_coords[:, max(0, int((words[w][0]-extension) * lm_const)):int((words[w][1]+extension) * lm_const), 0]
                            all_y = line_coords[:, max(0, int((words[w][0]-extension) * lm_const)):int((words[w][1]+extension) * lm_const), 1]

                            if all_x.size == 0 or all_y.size == 0:
                                extension += 1
                            else:
                                break

                        vpos = int(np.min(all_y))
                        hpos = int(np.min(all_x))
                        if x and y and height and width:
                            if vpos >= y and vpos <= (y+height) and hpos >= x and hpos <= (x+width):
                                bbox_confidences.append(confidences[only_letters[w]])
                        else:
                            bbox_confidences.append(confidences[only_letters[w]])

        if len(bbox_confidences) != 0:
            return (1 / len(bbox_confidences) * (np.power(bbox_confidences, power).sum())) ** (1 / power)
        else:
            return -1


def draw_lines(img, lines, color=(255, 0, 0), circles=(False, False, False), close=False, thickness=2):
    """Draw a line into image.
    :param img: input image
    :param lines: list of arrays of line coords
    :param color: RGB color of line
    :param circles: where to draw circles (start, mid steps, end)
    :param close: whether the rendered line should be a closed polygon
    """
    for line in lines:
        first = line[0]
        last = first
        if circles[0]:
            cv2.circle(img, (int(np.round(last[0])), int(np.round(last[1]))), 3, color, 4)
        for p in line[1:]:
            cv2.line(img, (int(np.round(last[0])), int(np.round(last[1]))), (int(np.round(p[0])), int(np.round(p[1]))),
                     color, thickness)
            if circles[1]:
                cv2.circle(img, (int(np.round(last[0])), int(np.round(last[1]))), 3, color, 4)
            last = p
        if circles[1]:
            cv2.circle(img, (int(np.round(line[-1][0])), int(np.round(line[-1][1]))), 3, color, 4)
        if close:
            cv2.line(img, (int(np.round(last[0])), int(np.round(last[1]))),
                     (int(np.round(first[0])), int(np.round(first[1]))), color, thickness)
    return img


def element_schema(elem):
    if elem.tag[0] == "{":
        schema, _, _ = elem.tag[1:].partition("}")
    else:
        schema = None
    return '{' + schema + '}'


def points_string_to_array(coords):
    coords = coords.split(' ')
    coords = [t.split(",") for t in coords]
    coords = [[int(round(float(x))), int(round(float(y)))] for x, y in coords]
    return np.asarray(coords)


def find_optimal(logit, positions, idx):
    maximum = -100
    highest = -1
    for i, item in enumerate(positions):
        if maximum < logit[item][idx]:
            maximum = logit[item][idx]
            highest = item

    return highest


def get_hwvh(polygon):
    xy = list(zip(*polygon))

    height = max(xy[1]) - min(xy[1])
    width = max(xy[0]) - min(xy[0])

    vpos = min(xy[1])
    hpos = min(xy[0])

    return height, width, vpos, hpos


def create_ocr_processing_element(id: str = "IdOcr",
                                  software_creator_str: str = "Project PERO",
                                  software_name_str: str = "PERO OCR",
                                  software_version_str: str = "v0.1.0",
                                  processing_datetime=None):
    ocr_processing = ET.Element("OCRProcessing")
    ocr_processing.set("ID", id)
    ocr_processing_step = ET.SubElement(ocr_processing, "ocrProcessingStep")
    processing_date_time = ET.SubElement(ocr_processing_step, "processingDateTime")
    if processing_datetime is not None:
        processing_date_time.text = processing_datetime
    else:
        processing_date_time.text = datetime.utcnow().isoformat()
    processing_software = ET.SubElement(ocr_processing_step, "processingSoftware")
    processing_creator = ET.SubElement(processing_software, "softwareCreator")
    processing_creator.text = software_creator_str
    software_name = ET.SubElement(processing_software, "softwareName")
    software_name.text = software_name_str
    software_version = ET.SubElement(processing_software, "softwareVersion")
    software_version.text = software_version_str

    return ocr_processing

