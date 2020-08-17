import re
import pickle
import json
from io import BytesIO
from datetime import datetime

import numpy as np
import lxml.etree as ET
import cv2

from pero_ocr.document_ocr.crop_engine import EngineLineCropper
from pero_ocr.force_alignment import align_text


def log_softmax(x):
    a = np.logaddexp.reduce(x, axis=1)[:, np.newaxis]
    return x - a


class TextLine(object):
    def __init__(self, id=None, baseline=None, polygon=None, heights=None, transcription=None, logits=None, crop=None,
                 characters=None):
        self.id = id
        self.baseline = baseline
        self.polygon = polygon
        self.heights = heights
        self.transcription = transcription
        self.logits = logits
        self.crop = crop
        self.characters = characters

    def get_dense_logits(self, zero_logit_value=-80):
        dense_logits = self.logits.toarray()
        dense_logits[dense_logits == 0] = zero_logit_value
        return dense_logits

    def get_full_logprobs(self, zero_logit_value=-80):
        dense_logits = self.get_dense_logits(zero_logit_value)
        return log_softmax(dense_logits)


class RegionLayout(object):
    def __init__(self, id, polygon):
        self.id = id  # ID string
        self.polygon = polygon  # bounding polygon
        self.lines = []
        self.transcription = None

    def to_page_xml(self, page_element):
        region_element = ET.SubElement(page_element, "TextRegion")
        coords = ET.SubElement(region_element, "Coords")
        region_element.set("id", self.id)
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
    layout_region = RegionLayout(region_element.attrib['id'], region_coords)
    transcription = region_element.find(schema + 'TextEquiv')
    if transcription is not None:
        layout_region.transcription = transcription.find(schema + 'Unicode').text
        if layout_region.transcription is None:
            layout_region.transcription = ''
    return layout_region


class PageLayout(object):
    def __init__(self, id=None, page_size=(0, 0), file=None):
        self.id = id
        self.page_size = page_size  # (height, width)
        self.regions = []  # list of RegionLayout objects
        if file is not None:
            self.from_pagexml(file)

    def from_pagexml_string(self, pagexml_string):
        self.from_pagexml(BytesIO(pagexml_string))

    def from_pagexml(self, file):
        page_tree = ET.parse(file)
        schema = element_schema(page_tree.getroot())

        page = page_tree.findall(schema + 'Page')[0]
        self.id = page.attrib['imageFilename']
        self.page_size = (int(page.attrib['imageHeight']), int(page.attrib['imageWidth']))

        for region in page_tree.iter(schema + 'TextRegion'):
            region_layout = get_region_from_page_xml(region, schema)

            for line in region.iter(schema + 'TextLine'):
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

                baseline = line.find(schema + 'Baseline')
                if baseline is not None:
                    new_textline.baseline = get_coords_form_page_xml(baseline, schema)

                textline = line.find(schema + 'Coords')
                if textline is not None:
                    new_textline.polygon = get_coords_form_page_xml(textline, schema)

                transcription = line.find(schema + 'TextEquiv')
                if transcription is not None:
                    t_unicode = transcription.find(schema + 'Unicode').text
                    if t_unicode is None:
                        t_unicode = ''
                    new_textline.transcription = t_unicode
                region_layout.lines.append(new_textline)

            self.regions.append(region_layout)

    def to_pagexml_string(self):
        root = ET.Element("PcGts")
        root.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")

        page = ET.SubElement(root, "Page")
        page.set("imageFilename", self.id)
        page.set("imageWidth", str(self.page_size[1]))
        page.set("imageHeight", str(self.page_size[0]))

        for region_layout in self.regions:
            text_region = region_layout.to_page_xml(page)

            for line in region_layout.lines:
                text_line = ET.SubElement(text_region, "TextLine")
                text_line.set("id", line.id)
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
                    text_element = ET.SubElement(text_element, "Unicode")
                    text_element.text = line.transcription

        return ET.tostring(root, pretty_print=True, encoding="utf-8").decode("utf-8")

    def to_pagexml(self, file_name):
        xml_string = self.to_pagexml_string()
        with open(file_name, 'w', encoding='utf-8') as out_f:
            out_f.write(xml_string)

    def to_altoxml_string(self):
        NSMAP = {"xlink": 'http://www.w3.org/1999/xlink',
                 "xsi": 'http://www.w3.org/2001/XMLSchema-instance'}
        root = ET.Element("alto", nsmap=NSMAP)
        root.set("xmlns", "http://www.loc.gov/standards/alto/ns-v2#")

        description = ET.SubElement(root, "Description")
        measurement_unit = ET.SubElement(description, "MeasurementUnit")
        measurement_unit.text = "pixel"
        ocr_processing = ET.SubElement(description, "OCRProcessing")
        ocr_processing.set("ID", "IdOcr")
        ocr_processing_step = ET.SubElement(ocr_processing, "ocrProcessingStep")
        processing_date_time = ET.SubElement(ocr_processing_step, "processingDateTime")
        processing_date_time.text = datetime.today().strftime('%Y-%m-%d')
        processing_software = ET.SubElement(ocr_processing_step, "processingSoftware")
        processing_creator = ET.SubElement(processing_software, "softwareCreator")
        processing_creator.text = "Project PERO"
        software_name = ET.SubElement(processing_software, "softwareName")
        software_name.text = "PERO OCR"
        software_version = ET.SubElement(processing_software, "softwareVersion")
        software_version.text = "v0.1.0"

        layout = ET.SubElement(root, "Layout")
        page = ET.SubElement(layout, "Page")
        page.set("ID", "id_" + self.id)
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
                if not line.transcription:
                    continue
                text_line = ET.SubElement(text_block, "TextLine")
                text_line_baseline = int(np.average(np.array(line.baseline)[:, 1]))
                text_line.set("BASELINE", str(text_line_baseline))

                text_line_height, text_line_width, text_line_vpos, text_line_hpos = get_hwvh(line.polygon)

                text_line.set("VPOS", str(int(text_line_vpos)))
                text_line.set("HPOS", str(int(text_line_hpos)))
                text_line.set("HEIGHT", str(int(text_line_height)))
                text_line.set("WIDTH", str(int(text_line_width)))

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

                logits = line.get_dense_logits()
                logprobs = line.get_full_logprobs()
                try:
                    aligned_letters = align_text(-logprobs, np.array(label), blank_idx)
                except ValueError as _:
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
                    word = []
                    words = []
                    counter = 0
                    for i, elem in enumerate(aligned_letters):
                        try:
                            if i == space_idxs[counter]:
                                if counter != (len(space_idxs) - 1):
                                    counter += 1
                                words.append(word)
                                word = []
                            else:
                                word.append(elem)
                        except IndexError as _:
                            word.append(elem)

                    words.append(word)

                    try:
                        words.remove([])
                    except ValueError:
                        pass

                    start_end = []
                    for i in words:
                        start_end.append(tuple([i[0], i[-1]]))

                    splitted_transcription = line.transcription.split()
                    lm_const = line_coords.shape[1] / logits.shape[0]
                    for w, word in enumerate(start_end):
                        string = ET.SubElement(text_line, "String")
                        string.set("CONTENT", splitted_transcription[w])
                        all_x = line_coords[:, int((start_end[w][0]-2) * lm_const):int((start_end[w][1]+2) * lm_const), 0]
                        all_y = line_coords[:, int((start_end[w][0]-2) * lm_const):int((start_end[w][1]+2) * lm_const), 1]

                        string.set("HEIGHT", str(int((np.max(all_y) - np.min(all_y)))))
                        string.set("WIDTH", str(int((np.max(all_x) - np.min(all_x)))))
                        string.set("VPOS", str(int(np.min(all_y))))
                        string.set("HPOS", str(int(np.min(all_x))))

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

        return ET.tostring(root, pretty_print=True, encoding="utf-8").decode("utf-8")

    def to_altoxml(self, file_name):
        alto_string = "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"yes\"?>\n" + self.to_altoxml_string()
        with open(file_name, 'w', encoding='utf-8') as out_f:
            out_f.write(alto_string)

    def from_altoxml_string(self, pagexml_string):
        self.from_pagexml(BytesIO(pagexml_string))

    def from_altoxml(self, file):
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

            region_layout = RegionLayout(region.attrib['ID'], np.asarray(region_coords))

            for line in region.iter(schema + 'TextLine'):
                new_textline = TextLine(baseline=[[int(line.attrib['HPOS']), int(line.attrib['BASELINE'])],
                                                  [int(line.attrib['HPOS']) + int(line.attrib['WIDTH']),
                                                   int(line.attrib['BASELINE'])]], polygon=[])
                new_textline.heights = [
                    int(line.attrib['HEIGHT']) + int(line.attrib['VPOS']) - int(line.attrib['BASELINE']),
                    int(line.attrib['BASELINE']) - int(line.attrib['VPOS'])]
                new_textline.polygon.append([int(line.attrib['HPOS']), int(line.attrib['VPOS'])])
                new_textline.polygon.append(
                    [int(line.attrib['HPOS']) + int(line.attrib['WIDTH']), int(line.attrib['VPOS'])])
                new_textline.polygon.append([int(line.attrib['HPOS']) + int(line.attrib['WIDTH']),
                                             int(line.attrib['VPOS']) + int(line.attrib['HEIGHT'])])
                new_textline.polygon.append(
                    [int(line.attrib['HPOS']), int(line.attrib['VPOS']) + int(line.attrib['HEIGHT'])])
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

    def save_logits(self, file_name):
        """Save page logits as a pickled dictionary of sparse matrices.
        :param file_name: to pickle into.
        """
        logits = []
        characters = []
        for region in self.regions:
            for line in region.lines:
                if line.logits is None:
                    raise Exception(f'Missing logits for line {line.id}.')
                if line.characters is None:
                    raise Exception(f'Missing logit mapping to characters for line {line.id}.')
            logits += [(line.id, line.logits) for line in region.lines]
            characters += [(line.id, line.characters) for line in region.lines]
        logits_dict = dict(logits)
        logits_dict['line_characters'] = dict(characters)
        with open(file_name, 'wb') as f:
            pickle.dump(logits_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_logits(self, file_name):
        """Load pagelogits as a pickled dictionary of sparse matrices.
        :param file_name: to pickle into.
        """
        with open(file_name, 'rb') as f:
            logits_dict = pickle.load(f)

        if 'line_characters' in logits_dict:
            characters = logits_dict['line_characters']
        else:
            characters = dict([(k, None) for k in logits_dict])

        for region in self.regions:
            for line in region.lines:
                if line.id not in logits_dict:
                    raise Exception(f'Missing line id {line.id} in logits {file_name}.')
                line.logits = logits_dict[line.id]
                line.characters = characters[line.id]

    def render_to_image(self, image, thickness=2, circles=True, render_order=False):
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
        for r in self.regions:
            for l in r.lines:
                yield l


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


if __name__ == '__main__':
    l = PageLayout(
        file='/home/ikohut/data/pero_ocr_web_data/ocr_client/0fb06b7c-92b3-41cd-9523-5a869dccd7dc/output/page/9baa3b0d-3a6c-41b9-86b3-a012ea0ed378.xml')
    l.load_logits(
        '/home/ikohut/data/pero_ocr_web_data/ocr_client/0fb06b7c-92b3-41cd-9523-5a869dccd7dc/output/logits/9baa3b0d-3a6c-41b9-86b3-a012ea0ed378.logits')
    print(l.to_altoxml_string())


    # test_layout = PageLayout(file='/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624_gt.xml')
    # test_layout.to_pagexml('/mnt/matylda1/ikodym/junk/refactor_test/test.xml')
    # image = cv2.imread('/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624.jpg')
    # img = test_layout.render_to_image(image)
    # cv2.imwrite('/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624_RENDER.jpg', img)

    def save():
        test_layout = PageLayout()
        test_layout.from_pagexml('C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/00cfab43-a5bc-4af0-b1c4-b26925679afd.xml')
        test_layout.load_logits('C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/00cfab43-a5bc-4af0-b1c4-b26925679afd.logits')
        test_layout.to_altoxml("C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/test_alto.xml")

    save()
