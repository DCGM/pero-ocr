import re
import pickle
from io import BytesIO

import numpy as np
import lxml.etree as ET
import cv2


class TextLine(object):
    def __init__(self, id=None, baseline=None, polygon=None, heights=None, transcription=None, logits=None, crop=None):
        self.id = id
        self.baseline = baseline
        self.polygon = polygon
        self.heights = heights
        self.transcription = transcription
        self.logits = logits
        self.crop = crop


class RegionLayout(object):
    def __init__(self, id, polygon):
        self.id = id  # ID string
        self.polygon = polygon  # bounding polygon
        self.lines = []


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
            region_coords = list()

            coords = region.find(schema + 'Coords')
            if 'points' in coords.attrib:
                region_coords = points_string_to_array(coords)
            else:
                for point in coords.findall(schema + 'Point'):
                    x, y = point.attrib['x'], point.attrib['y']
                    region_coords.append([int(round(float(x))), int(round(float(y)))])

            region_layout = RegionLayout(region.attrib['id'], np.asarray(region_coords))
            for line in region.iter(schema + 'TextLine'):
                new_textline = TextLine(id=line.attrib['id'])
                if 'custom' in line.attrib:
                    heights = re.findall("\d+", line.attrib['custom'])
                    if re.findall("heights", line.attrib['custom']):
                        heights_array = np.asarray([float(x) for x in heights])
                        if heights_array.shape[0] == 3:
                            heights = np.zeros(2, dtype=np.int32)
                            heights[0] = heights_array[1]
                            heights[1] = heights_array[2] - heights_array[0]
                        else:
                            heights = heights_array
                        new_textline.heights = heights.tolist()

                baseline = line.find(schema + 'Baseline')
                if baseline is not None:
                    new_textline.baseline = points_string_to_array(baseline)

                textline = line.find(schema + 'Coords')
                if textline is not None:
                    new_textline.polygon = points_string_to_array(textline)

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

            text_region = ET.SubElement(page, "TextRegion")
            coords = ET.SubElement(text_region, "Coords")
            text_region.set("id", region_layout.id)
            points = ["{},{}".format(int(coord[0]), int(coord[1])) for coord in region_layout.polygon]
            points = " ".join(points)
            coords.set("points", points)
            for line in region_layout.lines:
                text_line = ET.SubElement(text_region, "TextLine")
                text_line.set("id", line.id)
                if line.heights is not None:
                    text_line.set("custom", "heights {" + str(line.heights[0]) + ", " + str(line.heights[1]) + "}")
                coords = ET.SubElement(text_line, "Coords")

                if line.polygon is not None:
                    points = ["{},{}".format(int(coord[0]), int(coord[1])) for coord in line.polygon]
                    points = " ".join(points)
                    coords.set("points", points)

                if line.baseline is not None:
                    baseline_element = ET.SubElement(text_line, "Baseline")
                    points = ["{},{}".format(int(coord[0]), int(coord[1])) for coord in line.baseline]
                    points = " ".join(points)
                    baseline_element.set("points", points)

                if line.transcription is not None:
                    text_element = ET.SubElement(text_line, "TextEquiv")
                    text_element = ET.SubElement(text_element, "Unicode")
                    text_element.text = line.transcription

        return ET.tostring(root, pretty_print=True, encoding="utf-8").decode("utf-8")

    def to_pagexml(self, file_name):
        xml_string = self.to_pagexml_string()
        with open(file_name, 'w') as out_f:
            out_f.write(xml_string)

    def save_logits(self, file_name):
        """Save page logits as a pickled dictionary of sparse matrices.
        :param file_name: to pickle into.
        """
        logits = []
        for region in self.regions:
            for line in region.lines:
                if line.logits is None:
                    raise Exception(f'Missing logits for line {line.id}.')
            logits += [(line.id, line.logits) for line in region.lines]
        logits_dict = dict(logits)
        with open(file_name, 'wb') as f:
            pickle.dump(logits_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_logits(self, file_name):
        """Load pagelogits as a pickled dictionary of sparse matrices.
        :param file_name: to pickle into.
        """
        with open(file_name, 'rb') as f:
            logits_dict = pickle.load(f)

        for region in self.regions:
            for line in region.lines:
                if line.id not in logits_dict:
                    raise Exception(f'Missing line id {line.id} in logits {file_name}.')
                line.logits = logits_dict[line.id]

    def render_to_image(self, image):
        """Render layout into image.
        :param image: image to render layout into
        """
        for region_layout in self.regions:
            image = draw_lines(
                image,
                [region_layout.polygon], color=(255, 0, 0), circles=(True, True, True), close=True)
            image = draw_lines(
                image,
                [line.baseline for line in region_layout.lines], color=(0,0,255), circles=(True, True, False))
            image = draw_lines(
                image,
                [line.polygon for line in region_layout.lines], color=(0,255,0), close=True)
        return image

    def lines_iterator(self):
        for r in self.regions:
            for l in r.lines:
                yield l


def draw_lines(img, lines, color=(255,0,0), circles=(False, False, False), close=False):
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
            cv2.circle(img, (int(last[0]), int(last[1])), 3, color, 4)
        for p in line[1:]:
            cv2.line(img, (int(last[0]), int(last[1])), (int(p[0]), int(p[1])), color, 2)
            if circles[1]:
                cv2.circle(img, (int(last[0]), int(last[1])), 3, color, 4)
            last = p
        if circles[1]:
            cv2.circle(img, (int(line[-1][0]), int(line[-1][1])), 3, color, 4)
        if close:
            cv2.line(img, (int(last[0]), int(last[1])), (int(first[0]), int(first[1])), color, 2)
    return img


def element_schema(elem):
    if elem.tag[0] == "{":
        schema, _, _ = elem.tag[1:].partition("}")
    else:
        schema = None
    return '{' + schema + '}'


def points_string_to_array(coords):
    coords = coords.attrib['points'].split(' ')
    coords = [t.split(",") for t in coords]
    coords = [[int(round(float(x))), int(round(float(y)))] for x, y in coords]
    return np.asarray(coords)


if __name__ == '__main__':
    test_layout = PageLayout(file='/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624_gt.xml')
    test_layout.to_pagexml('/mnt/matylda1/ikodym/junk/refactor_test/test.xml')
    image = cv2.imread('/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624.jpg')
    img = test_layout.render_to_image(image)
    cv2.imwrite('/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624_RENDER.jpg', img)
