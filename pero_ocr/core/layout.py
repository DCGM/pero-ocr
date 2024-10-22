import logging
import re
import pickle
import json
from io import BytesIO
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union, List, Tuple
import unicodedata

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

class ALTOVersion(Enum):
    ALTO_v2_x = 1
    ALTO_v4_4 = 2

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
                 characters: Optional[List[str]] = None,
                 logit_coords: Optional[Union[List[Tuple[int]], List[Tuple[None]]]] = None,
                 transcription_confidence: Optional[Num] = None,
                 index: Optional[int] = None,
                 category: Optional[str] = None,
                 metadata: Optional[list[object]] = None):
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
        self.category = category
        self.metadata = metadata

    def get_dense_logits(self, zero_logit_value: int = -80):
        dense_logits = self.logits.toarray()
        dense_logits[dense_logits == 0] = zero_logit_value
        return dense_logits

    def get_full_logprobs(self, zero_logit_value: int = -80):
        dense_logits = self.get_dense_logits(zero_logit_value)
        return log_softmax(dense_logits)

    def to_pagexml(self, region_element: ET.SubElement, fallback_id: int, validate_id: bool = False):
        text_line = ET.SubElement(region_element, "TextLine")
        text_line.set("id", export_id(self.id, validate_id))
        if self.index is not None:
            text_line.set("index", f'{self.index:d}')
        else:
            text_line.set("index", f'{fallback_id:d}')

        custom = {}
        if self.heights is not None:
            heights_out = [np.float64(x) for x in self.heights]
            custom['heights'] = list(np.round(heights_out, decimals=1))
        if self.category is not None:
            custom['category'] = self.category
        if len(custom) > 0:
            text_line.set("custom", json.dumps(custom))

        coords = ET.SubElement(text_line, "Coords")

        if self.polygon is not None:
            coords.set("points", coords_to_pagexml_points(self.polygon))

        if self.baseline is not None:
            baseline_element = ET.SubElement(text_line, "Baseline")
            baseline_element.set("points", coords_to_pagexml_points(self.baseline))

        if self.transcription is not None:
            text_element = ET.SubElement(text_line, "TextEquiv")
            if self.transcription_confidence is not None:
                text_element.set("conf", f"{self.transcription_confidence:.3f}")
            text_element = ET.SubElement(text_element, "Unicode")
            text_element.text = self.transcription

    @classmethod
    def from_pagexml(cls, line_element: ET.SubElement, schema, fallback_index: int):
        new_textline = cls(id=line_element.attrib['id'])
        if 'custom' in line_element.attrib:
            new_textline.from_pagexml_parse_custom(line_element.attrib['custom'])

        if 'index' in line_element.attrib:
            try:
                new_textline.index = int(line_element.attrib['index'])
            except ValueError:
                pass

        if new_textline.index is None:
            new_textline.index = fallback_index

        baseline = line_element.find(schema + 'Baseline')
        if baseline is not None:
            new_textline.baseline = get_coords_from_pagexml(baseline, schema)
        else:
            logger.warning(f'Warning: Baseline is missing in TextLine. '
                           f'Skipping this line during import. Line ID: {new_textline.id}')
            return None

        textline = line_element.find(schema + 'Coords')
        if textline is not None:
            new_textline.polygon = get_coords_from_pagexml(textline, schema)

        if not new_textline.heights:
            guess_line_heights_from_polygon(new_textline, use_center=False, n=len(new_textline.baseline))

        transcription = line_element.find(schema + 'TextEquiv')
        if transcription is not None:
            t_unicode = transcription.find(schema + 'Unicode').text
            if t_unicode is None:
                t_unicode = ''
            new_textline.transcription = t_unicode
            conf = transcription.get('conf', None)
            new_textline.transcription_confidence = float(conf) if conf is not None else None
        return new_textline

    def from_pagexml_parse_custom(self, custom_str):
        try:
            custom = json.loads(custom_str)
            self.category = custom.get('category', None)
            self.heights = custom.get('heights', None)
        except json.decoder.JSONDecodeError:
            if 'heights_v2' in custom_str:
                for word in custom_str.split():
                    if 'heights_v2' in word:
                        self.heights = json.loads(word.split(":")[1])
            else:
                if re.findall("heights", custom_str):
                    heights = re.findall(r"\d+", custom_str)
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
                    self.heights = heights.tolist()

    def to_altoxml(self, text_block, tags, mods_namespace, arabic_helper, min_line_confidence, version: ALTOVersion):
        if self.transcription_confidence is not None and self.transcription_confidence < min_line_confidence:
            return

        text_line = ET.SubElement(text_block, "TextLine")
        text_line.set("ID", f'line_{self.id}')
        text_line.set("BASELINE", self.to_altoxml_baseline(version))

        text_line_height, text_line_width, text_line_vpos, text_line_hpos = get_hwvh(self.polygon)

        text_line.set("VPOS", str(int(text_line_vpos)))
        text_line.set("HPOS", str(int(text_line_hpos)))
        text_line.set("HEIGHT", str(int(text_line_height)))
        text_line.set("WIDTH", str(int(text_line_width)))

        if self.category == 'text':
            self.to_altoxml_text(text_line, arabic_helper,
                                 text_line_height, text_line_width, text_line_vpos, text_line_hpos)
        else:
            string = ET.SubElement(text_line, "String")
            string.set("CONTENT", self.transcription)

            string.set("HEIGHT", str(int(text_line_height)))
            string.set("WIDTH", str(int(text_line_width)))
            string.set("VPOS", str(int(text_line_vpos)))
            string.set("HPOS", str(int(text_line_hpos)))

            if self.transcription_confidence is not None:
                string.set("WC", str(round(self.transcription_confidence, 2)))

        if self.metadata is not None:
            tag_references = []
            for metadata in self.metadata:
                tag_references.append(metadata.metadata_id)

                exist = False
                for child in tags.getchildren():
                    if "ID" in child.attrib and child.attrib["ID"] == metadata.metadata_id:
                        exist = True
                        break

                if not exist:
                    metadata.to_altoxml(tags, mods_namespace)

            text_line.set("TAGREFS", ' '.join(tag_references))

    def get_labels(self):
        chars = [i for i in range(len(self.characters))]
        char_to_num = dict(zip(self.characters, chars))

        blank_idx = self.logits.shape[1] - 1

        labels = []
        for item in self.transcription:
            if item in char_to_num.keys():
                if char_to_num[item] >= blank_idx:
                    labels.append(0)
                else:
                    labels.append(char_to_num[item])
            else:
                labels.append(0)
        return np.array(labels)

    def to_altoxml_text(self, text_line, arabic_helper,
                        text_line_height, text_line_width, text_line_vpos, text_line_hpos):
        arabic_line = False
        if arabic_helper.is_arabic_line(self.transcription):
            arabic_line = True

        logits = None
        logprobs = None
        aligned_letters = None
        try:
            label = self.get_labels()
            blank_idx = self.logits.shape[1] - 1

            logits = self.get_dense_logits()[self.logit_coords[0]:self.logit_coords[1]]
            logprobs = self.get_full_logprobs()[self.logit_coords[0]:self.logit_coords[1]]
            aligned_letters = align_text(-logprobs, np.array(label), blank_idx)
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(f'Error: Alto export, unable to align line {self.id} due to exception: {e}.')

            if logits is not None:
                max_val = np.max(logits, axis=1)
                logits = logits - max_val[:, np.newaxis]
                probs = np.exp(logits)
                probs = probs / np.sum(probs, axis=1, keepdims=True)
                probs = np.max(probs, axis=1)
                self.transcription_confidence = np.quantile(probs, .50)
            else:
                self.transcription_confidence = 0.0

            average_word_width = (text_line_hpos + text_line_width) / len(self.transcription.split())
            for w, word in enumerate(self.transcription.split()):
                string = ET.SubElement(text_line, "String")
                string.set("CONTENT", word)

                string.set("HEIGHT", str(int(text_line_height)))
                string.set("WIDTH", str(int(average_word_width)))
                string.set("VPOS", str(int(text_line_vpos)))
                string.set("HPOS", str(int(text_line_hpos + (w * average_word_width))))
        else:
            crop_engine = EngineLineCropper(poly=2)
            line_coords = crop_engine.get_crop_inputs(self.baseline, self.heights, 16)
            space_idxs = [pos for pos, char in enumerate(self.transcription) if char == ' ']

            words = []
            space_idxs = [-1] + space_idxs + [len(aligned_letters)]
            for i in range(len(space_idxs[1:])):
                if space_idxs[i] != space_idxs[i + 1] - 1:
                    words.append([aligned_letters[space_idxs[i] + 1], aligned_letters[space_idxs[i + 1] - 1]])
            splitted_transcription = self.transcription.split()
            lm_const = line_coords.shape[1] / logits.shape[0]
            letter_counter = 0
            confidences = get_line_confidence(self, np.array(label), aligned_letters, logprobs)
            # if self.transcription_confidence is None:
            self.transcription_confidence = np.quantile(confidences, .50)
            for w, word in enumerate(words):
                extension = 2
                while line_coords.size > 0 and extension < 40:
                    all_x = line_coords[:,
                            max(0, int((words[w][0] - extension) * lm_const)):int((words[w][1] + extension) * lm_const),
                            0]
                    all_y = line_coords[:,
                            max(0, int((words[w][0] - extension) * lm_const)):int((words[w][1] + extension) * lm_const),
                            1]

                    if all_x.size == 0 or all_y.size == 0:
                        extension += 1
                    else:
                        break

                if line_coords.size == 0 or all_x.size == 0 or all_y.size == 0:
                    all_x = self.baseline[:, 0]
                    all_y = np.concatenate(
                        [self.baseline[:, 1] - self.heights[0], self.baseline[:, 1] + self.heights[1]])

                word_confidence = None
                if self.transcription_confidence == 1:
                    word_confidence = 1
                else:
                    if confidences.size != 0:
                        word_confidence = np.quantile(
                            confidences[letter_counter:letter_counter + len(splitted_transcription[w])], .50)

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

                if w != (len(self.transcription.split()) - 1):
                    space = ET.SubElement(text_line, "SP")

                    space.set("WIDTH", str(4))
                    space.set("VPOS", str(int(np.min(all_y))))
                    space.set("HPOS", str(int(np.max(all_x))))
                letter_counter += len(splitted_transcription[w]) + 1

    def to_altoxml_baseline(self, version: ALTOVersion) -> str:
        if version == ALTOVersion.ALTO_v2_x:
            # ALTO 4.1 and older accept baseline only as a single point
            baseline = int(np.round(np.average(np.array(self.baseline)[:, 1])))
            return str(baseline)
        elif version == ALTOVersion.ALTO_v4_4:
            # ALTO 4.2 and newer accept baseline as a string with list of points. Recommended "x1,y1 x2,y2 ..." format.
            baseline_points = [f"{x},{y}" for x, y in np.round(self.baseline).astype('int')]
            baseline_points = " ".join(baseline_points)
            return baseline_points
        else:
            return ""

    @classmethod
    def from_altoxml(cls, line: ET.SubElement, schema):
        hpos = int(line.attrib['HPOS'])
        vpos = int(line.attrib['VPOS'])
        width = int(line.attrib['WIDTH'])
        height = int(line.attrib['HEIGHT'])
        baseline_str = line.attrib['BASELINE']
        baseline, heights, polygon = cls.from_altoxml_polygon(baseline_str, hpos, vpos, width, height)

        new_textline = cls(id=line.attrib['ID'], baseline=baseline, heights=heights, polygon=polygon)

        word = ''
        start = True
        for text in line.iter(schema + 'String'):
            if start:
                start = False
                word = word + text.get('CONTENT')
            else:
                word = word + " " + text.get('CONTENT')
        new_textline.transcription = word
        return new_textline

    @staticmethod
    def from_altoxml_polygon(baseline_str, hpos, vpos, width, height) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        baseline = baseline_str.strip().split(' ')

        if len(baseline) == 1:
            # baseline is only one number (probably ALTOversion = 2.x)
            try:
                baseline = float(baseline[0])
            except ValueError:
                baseline = vpos + height  # fallback: baseline is at the bottom of the bounding box, heights[1] = 0

            baseline_arr = np.asarray([[hpos, baseline], [hpos + width, baseline]])
            heights = np.asarray([baseline - vpos, vpos + height - baseline])
            polygon = np.asarray([[hpos, vpos],
                                  [hpos + width, vpos],
                                  [hpos + width, vpos + height],
                                  [hpos, vpos + height]])
            return baseline_arr, heights, polygon
        else:
            # baseline is list of points (probably ALTOversion = 4.4)
            baseline_coords = [t.split(",") for t in baseline]
            baseline = np.asarray([[int(round(float(x))), int(round(float(y)))] for x, y in baseline_coords])

            # count heights from the FIRST element of baseline
            heights = np.asarray([baseline[0, 1] - vpos, vpos + height - baseline[0, 1]])

            coords_top = [[x, y - heights[0]] for x, y in baseline]
            coords_bottom = [[x, y + heights[1]] for x, y in baseline]
            # reverse coords_bottom to create polygon in clockwise order
            coords_bottom.reverse()
            polygon = np.concatenate([coords_top, coords_bottom, coords_top[:1]])

            return baseline, heights, polygon


class RegionLayout(object):
    def __init__(self, id: str,
                 polygon: np.ndarray,
                 region_type: Optional[str] = None,
                 category: Optional[str] = None,
                 detection_confidence: Optional[float] = None,
                 metadata: Optional[object] = None):
        self.id = id  # ID string
        self.polygon = polygon  # bounding polygon
        self.region_type = region_type
        self.category = category
        self.lines: List[TextLine] = []
        self.transcription = None
        self.detection_confidence = detection_confidence
        self.metadata = metadata

    def get_lines_of_category(self, categories: Union[str, list]):
        if isinstance(categories, str):
            categories = [categories]

        return [line for line in self.lines if line.category in categories]

    def replace_id(self, new_id):
        """Replace region ID and all IDs in TextLines which has region ID inside them."""
        for line in self.lines:
            line.id = line.id.replace(self.id, new_id)
        self.id = new_id

    def get_polygon_bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box of region polygon which includes all polygon points.
        :return: tuple[int, int, int, int]: (x_min, y_min, x_max, y_max)
        """
        x_min = min(self.polygon[:, 0])
        x_max = max(self.polygon[:, 0])
        y_min = min(self.polygon[:, 1])
        y_max = max(self.polygon[:, 1])

        return x_min, y_min, x_max, y_max

    def to_pagexml(self, page_element: ET.SubElement, validate_id: bool = False):
        region_element = ET.SubElement(page_element, "TextRegion")
        coords = ET.SubElement(region_element, "Coords")
        region_element.set("id", export_id(self.id, validate_id))

        if self.region_type is not None:
            region_element.set("type", self.region_type)

        custom = {}
        if self.category is not None:
            custom['category'] = self.category
        if self.detection_confidence is not None:
            custom['detection_confidence'] = round(self.detection_confidence, 3)
        if len(custom) > 0:
            custom = json.dumps(custom)
            region_element.set("custom", custom)

        coords.set("points", coords_to_pagexml_points(self.polygon))

        if self.transcription is not None:
            text_element = ET.SubElement(region_element, "TextEquiv")
            text_element = ET.SubElement(text_element, "Unicode")
            text_element.text = self.transcription

        for i, line in enumerate(self.lines):
            line.to_pagexml(region_element, fallback_id=i, validate_id=validate_id)

        return region_element

    @classmethod
    def from_pagexml(cls, region_element: ET.SubElement, schema):
        coords_element = region_element.find(schema + 'Coords')
        region_coords = get_coords_from_pagexml(coords_element, schema)

        region_type = None
        if "type" in region_element.attrib:
            region_type = region_element.attrib["type"]

        category = None
        detection_confidence = None
        if "custom" in region_element.attrib:
            custom = json.loads(region_element.attrib["custom"])
            category = custom.get('category', None)
            detection_confidence = custom.get('detection_confidence', None)

        layout_region = cls(region_element.attrib['id'], region_coords, region_type,
                            category=category,
                            detection_confidence=detection_confidence)

        transcription = region_element.find(schema + 'TextEquiv')
        if transcription is not None:
            layout_region.transcription = transcription.find(schema + 'Unicode').text
            if layout_region.transcription is None:
                layout_region.transcription = ''

        for i, line in enumerate(region_element.iter(schema + 'TextLine')):
            new_textline = TextLine.from_pagexml(line, schema, fallback_index=i)
            if new_textline is not None:
                layout_region.lines.append(new_textline)

        return layout_region

    def to_altoxml(self, print_space, tags, mods_namespace, arabic_helper, min_line_confidence,
                   print_space_coords: Tuple[int, int, int, int], version: ALTOVersion) -> Tuple[int, int, int, int]:
        print_space_height, print_space_width, print_space_vpos, print_space_hpos = print_space_coords

        if self.category is None or self.category == 'text':
            block = ET.SubElement(print_space, "TextBlock")

            if self.category is None or self.category == 'text':
                block.set("ID", 'block_{}'.format(self.id))
            else:
                block.set("ID", self.id)

        else:
            from orbis_pictus.anno_page.layout import region_to_altoxml
            block = region_to_altoxml(self, print_space)

        block_height, block_width, block_vpos, block_hpos = get_hwvh(self.polygon)
        block.set("HEIGHT", str(int(block_height)))
        block.set("WIDTH", str(int(block_width)))
        block.set("VPOS", str(int(block_vpos)))
        block.set("HPOS", str(int(block_hpos)))

        print_space_height = max([print_space_vpos + print_space_height, block_vpos + block_height])
        print_space_width = max([print_space_hpos + print_space_width, block_hpos + block_width])
        print_space_vpos = min([print_space_vpos, block_vpos])
        print_space_hpos = min([print_space_hpos, block_hpos])
        print_space_height = print_space_height - print_space_vpos
        print_space_width = print_space_width - print_space_hpos

        if self.metadata is not None:
            self.metadata.to_altoxml(tags,
                                     category=self.category,
                                     bounding_box=self.get_polygon_bounding_box(),
                                     confidence=self.detection_confidence,
                                     mods_namespace=mods_namespace)

        for line in self.lines:
            if not line.transcription or line.transcription.strip() == "":
                continue
            line.to_altoxml(block, tags, mods_namespace, arabic_helper, min_line_confidence, version)
        
        return print_space_height, print_space_width, print_space_vpos, print_space_hpos

    @classmethod
    def from_altoxml(cls, text_block: ET.SubElement, schema):
        region_coords = list()
        region_coords.append([int(text_block.get('HPOS')), int(text_block.get('VPOS'))])
        region_coords.append([int(text_block.get('HPOS')) + int(text_block.get('WIDTH')), int(text_block.get('VPOS'))])
        region_coords.append([int(text_block.get('HPOS')) + int(text_block.get('WIDTH')),
                              int(text_block.get('VPOS')) + int(text_block.get('HEIGHT'))])
        region_coords.append([int(text_block.get('HPOS')), int(text_block.get('VPOS')) + int(text_block.get('HEIGHT'))])

        region_layout = cls(text_block.attrib['ID'], np.asarray(region_coords).tolist())

        for line in text_block.iter(schema + 'TextLine'):
            new_textline = TextLine.from_altoxml(line, schema)
            region_layout.lines.append(new_textline)

        return region_layout


def get_coords_from_pagexml(coords_element, schema):
    if 'points' in coords_element.attrib:
        coords = points_string_to_array(coords_element.attrib['points'])
    else:
        coords = []
        for point in coords_element.findall(schema + 'Point'):
            x, y = point.attrib['x'], point.attrib['y']
            coords.append([float(x), float(y)])
        coords = np.asarray(coords)
    return coords


def coords_to_pagexml_points(polygon: np.ndarray) -> str:
    polygon = np.round(polygon).astype(np.dtype('int'))
    points = [f"{x},{y}" for x, y in np.maximum(polygon, 0)]
    points = " ".join(points)
    return points


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
    def __init__(self, id: str = None, page_size: List[Tuple[int]] = (0, 0), file: str = None):
        self.id = id
        self.page_size = page_size  # (height, width)
        self.regions: List[RegionLayout] = []
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
            region_layout = RegionLayout.from_pagexml(region, schema)
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

        if self.reading_order is not None and self.reading_order != {}:
            self.sort_regions_by_reading_order()
            self.reading_order_to_pagexml(page)

        for region_layout in self.regions:
            region_layout.to_pagexml(page, validate_id=validate_id)

        return ET.tostring(root, pretty_print=True, encoding="utf-8", xml_declaration=True).decode("utf-8")

    def to_pagexml(self, file_name: str, creator: str = 'Pero OCR',
                   validate_id: bool = False, version: PAGEVersion = PAGEVersion.PAGE_2019_07_15):
        xml_string = self.to_pagexml_string(version=version, creator=creator, validate_id=validate_id)
        with open(file_name, 'w', encoding='utf-8') as out_f:
            out_f.write(xml_string)

    def to_altoxml_string(self, ocr_processing_element: ET.SubElement = None, page_uuid: str = None,
                          min_line_confidence: float = 0, version: ALTOVersion = ALTOVersion.ALTO_v2_x):
        arabic_helper = ArabicHelper()

        mods_namespace_url = "http://www.loc.gov/mods/v3"

        NSMAP = {"xlink": 'http://www.w3.org/1999/xlink',
                 "mods": mods_namespace_url,
                 "xsi": 'http://www.w3.org/2001/XMLSchema-instance'}
        root = ET.Element("alto", nsmap=NSMAP)

        if version == ALTOVersion.ALTO_v4_4:
            root.set("xmlns", "http://www.loc.gov/standards/alto/ns-v4#")
        elif version == ALTOVersion.ALTO_v2_x:
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
            ocr_processing_element = create_ocr_processing_element(alto_version=version)
            description.append(ocr_processing_element)
        tags = ET.SubElement(root, "Tags")
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
        print_space_coords = (print_space_height, print_space_width, print_space_vpos, print_space_hpos)

        text_regions = []
        nontext_regions = []
        for region in self.regions:
            if region.category is None or region.category == 'text':
                text_regions.append(region)
            else:
                nontext_regions.append(region)

        for region in nontext_regions:
            print_space_coords = region.to_altoxml(print_space, tags, mods_namespace_url, arabic_helper, min_line_confidence, print_space_coords, version)

        for region in text_regions:
            print_space_coords = region.to_altoxml(print_space, tags, mods_namespace_url, arabic_helper, min_line_confidence, print_space_coords, version)

        print_space_height, print_space_width, print_space_vpos, print_space_hpos = print_space_coords

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

    def to_altoxml(self, file_name: str, ocr_processing_element: ET.SubElement = None, page_uuid: str = None,
                   version: ALTOVersion = ALTOVersion.ALTO_v2_x):
        alto_string = self.to_altoxml_string(ocr_processing_element=ocr_processing_element, page_uuid=page_uuid, version=version)
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
            region_layout = RegionLayout.from_altoxml(region, schema)
            self.regions.append(region_layout)

    def sort_regions_by_reading_order(self):
        self.regions = sorted(self.regions, key=lambda k: self.reading_order[k] if k in self.reading_order else float("inf"))

    def reading_order_to_pagexml(self, page_element: ET.SubElement):
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

    def render_to_image(self, image, thickness: int = 2, circles: bool = True,
                        render_order: bool = False, render_category: bool = False):
        """Render layout into image.
        :param image: image to render layout into
        :param render_order: render region order number given by enumerate(regions) to the middle of given region
        :param render_region_id: render region id to the upper left corner of given region
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

            if region_layout.category is None or region_layout.category == "text":
                region_color = (255, 0, 0)
                region_circles = (circles, circles, circles)
            else:
                region_color = (255, 0, 255)
                region_circles = (False, False, False)

            image = draw_lines(
                image,
                [region_layout.polygon], color=region_color, circles=region_circles, close=True,
                thickness=thickness)

        if render_order or render_category:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            font_thickness = 1

            for idx, region in enumerate(self.regions):
                min_p = region.polygon.min(axis=0)
                max_p = region.polygon.max(axis=0)

                if render_order:
                    text = f"{idx}"
                    text_w, text_h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    mid_x = int((min_p[0] + max_p[0]) // 2 - text_w // 2)
                    mid_y = int((min_p[1] + max_p[1]) // 2 + text_h // 2)
                    cv2.putText(image, text, (mid_x, mid_y), font, font_scale,
                                color=(0, 0, 0), thickness=font_thickness, lineType=cv2.LINE_AA)
                if render_category and region.category not in [None, 'text']:
                    text = f"{normalize_text(region.category)}"
                    text_w, text_h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    start_point = (int(min_p[0]), int(min_p[1]))
                    end_point = (int(min_p[0]) + text_w, int(min_p[1]) - text_h)
                    cv2.rectangle(image, start_point, end_point, color=(255, 0, 0), thickness=-1)
                    cv2.putText(image, text, start_point, font, font_scale,
                                color=(255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)

        return image

    def lines_iterator(self, categories: list = None):
        for region in self.regions:
            for line in region.lines:
                if not categories or line.category in categories:
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

    def rename_region_id(self, old_id, new_id):
        for region in self.regions:
            if region.id == old_id:
                region.replace_id(new_id)
                break
        else:
            raise ValueError(f'Region with id {old_id} not found.')


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


def normalize_text(text: str) -> str:
    """Normalize text to ASCII characters. (e.g. Obrázek -> Obrazek)"""
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')