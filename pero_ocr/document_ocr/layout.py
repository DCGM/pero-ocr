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

            for coords in region.findall(schema + 'Coords'):
                if 'points' in coords.attrib:
                    points_string = coords.attrib['points'].split(' ')
                    for points in points_string:
                        x, y = points.split(',')
                        region_coords.append([int(round(float(x))), int(round(float(y)))])
                else:
                    for point in coords.findall(schema + 'Point'):
                        x, y = point.attrib['x'], point.attrib['y']
                        region_coords.append([int(round(float(x))), int(round(float(y)))])

            region_layout = RegionLayout(region.attrib['id'], np.asarray(region_coords))
            for line in region.iter(schema + 'TextLine'):
                new_textline = TextLine(id=line.attrib['id'])
                heights = re.findall("\d+", line.attrib['custom'])
                if re.findall("heights", line.attrib['custom']):
                    heights_array = np.asarray([int(round(float(x))) for x in heights])
                    if heights_array.shape[0] == 3:
                        heights = np.zeros(2, dtype=np.int32)
                        heights[0] = heights_array[1]
                        heights[1] = heights_array[2] - heights_array[0]
                    else:
                        heights = heights_array
                    new_textline.heights = heights.tolist()

                baseline = line.find(schema + 'Baseline')
                if baseline is not None:
                    points_string = baseline.attrib['points'].split(' ')
                    baseline = list()
                    for point in points_string:
                        x, y = point.split(',')
                        baseline.append([int(round(float(y))), int(round(float(x)))])
                    new_textline.baseline = baseline

                textline = line.find(schema + 'Coords')
                if textline is not None:
                    points_string = textline.attrib['points'].split(' ')
                    textline = list()
                    for point in points_string:
                        x, y = point.split(',')
                        textline.append([int(round(float(y))), int(round(float(x)))])
                    new_textline.polygon = textline

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
            points = ["{},{}".format(int(x[0]), int(x[1])) for x in region_layout.polygon]
            points = " ".join(points)
            coords.set("points", points)
            for line in region_layout.lines:
                text_line = ET.SubElement(text_region, "TextLine")
                text_line.set("id", line.id)
                if line.heights is not None:
                    text_line.set("custom", "heights {" + str(int(line.heights[0])) + ", " + str(int(line.heights[1])) + "}")
                coords = ET.SubElement(text_line, "Coords")

                if line.polygon is not None:
                    points = ["{},{}".format(int(x[1]), int(x[0])) for x in line.polygon]
                    points = " ".join(points)
                    coords.set("points", points)

                if line.baseline is not None:
                    baseline_element = ET.SubElement(text_line, "Baseline")
                    points = ["{},{}".format(int(x[1]), int(x[0])) for x in line.baseline]
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
                [line.baseline for line in region_layout.lines], color=(0,0,255), circles=(True, True, False))
            image = draw_lines(
                image,
                [line.polygon for line in region_layout.lines], color=(0,255,0), close=True)
            image = draw_lines(
                image,
                [region_layout.polygon], color=(255, 0, 0), circles=(True, True, True), close=True)
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
            cv2.circle(img, (int(last[1]), int(last[0])), 3, color, 4)
        for p in line[1:]:
            cv2.line(img, (int(last[1]), int(last[0])), (int(p[1]), int(p[0])), color, 2)
            if circles[1]:
                cv2.circle(img, (int(last[1]), int(last[0])), 3, color, 4)
            last = p
        if circles[1]:
            cv2.circle(img, (int(line[-1][1]), int(line[-1][0])), 3, color, 4)
        if close:
            cv2.line(img, (int(last[1]), int(last[0])), (int(first[1]), int(first[0])), color, 2)
    return img


def element_schema(elem):
    if elem.tag[0] == "{":
        schema, _, _ = elem.tag[1:].partition("}")
    else:
        schema = None
    return '{' + schema + '}'


if __name__ == '__main__':
    test_layout = PageLayout(file='/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624_gt.xml')
    test_layout.to_pagexml('/mnt/matylda1/ikodym/junk/refactor_test/test.xml')
    image = cv2.imread('/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624.jpg')
    test_layout.render_to_image(image, '/mnt/matylda1/ikodym/junk/refactor_test/')

#
# def simple_line_extraction(self, img, element_size=2):
#     region = np.asarray(self.coords)
#
#     y1, y2, x1, x2 = np.amin(region[:, 1]), np.amax(region[:, 1]), np.amin(region[:, 0]), np.amax(region[:, 0])
#     column_width = x2 - x1
#     column_height = y2 - y1
#     img_crop = img[y1:y2, x1:x2, :]
#     img_crop = img_crop.mean(axis=2).astype(np.uint8)
#     img_crop = cv2.adaptiveThreshold(img_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 20) == 0
#
#     img_crop_labeled, num_features = sn.measurements.label(img_crop)
#     proj = np.sum(img_crop, axis=1)
#     corr = np.correlate(proj, proj, mode='full')[proj.shape[0]:]
#     corr_peaks = ss.find_peaks(corr, prominence=0, distance=1)[0]
#     if len(corr_peaks) > 0:
#         line_period = float(ss.find_peaks(corr, prominence=0, distance=1)[0][0])
#     else:
#         line_period = 1
#     target_signal = - np.diff(proj)
#     target_signal[target_signal < 0] = 0
#
#     baseline_coords = ss.find_peaks(target_signal, distance=int(round(0.85*line_period)))[0]
#
#     poly_coords = [coords[::-1] for coords in self.coords]
#     region = shapely.geometry.polygon.Polygon(poly_coords)
#     used_inds = []
#
#     for baseline_coord in baseline_coords[::-1]:
#         valid_baseline = True
#         matching_objects = np.unique(img_crop_labeled[baseline_coord-10, column_width//10:-column_width//10])[1:]
#         if len(matching_objects) > 0:
#             for ind in matching_objects:
#                 if ind in used_inds:
#                     valid_baseline = False
#                 used_inds.append(ind)
#
#             for yb1 in range(baseline_coord, 0, -3):
#                 line_inds_to_check = img_crop_labeled[yb1, column_width//10:-column_width//10]
#                 if not np.any(np.intersect1d(matching_objects, line_inds_to_check)):
#                     break
#
#             for yb2 in range(baseline_coord, column_height, 3):
#                 line_inds_to_check = img_crop_labeled[yb2, column_width//10:-column_width//10]
#                 if not np.any(np.intersect1d(matching_objects, line_inds_to_check)):
#                     break
#
#             xb1, xb2 = 0, column_width
#
#             if yb2 - yb1 < 6:
#                 valid_baseline = False
#
#             line = shapely.geometry.LineString([[y1+baseline_coord, x1+xb1-20], [y1+baseline_coord, x1+xb2+20]])
#             intersection = region.intersection(line)
#             if not intersection.is_empty:
#                 if valid_baseline:
#                     self.baselines.append(np.round(np.asarray(list(region.intersection(line).coords[:]))).astype(np.int16))
#                     self.heights.append([baseline_coord-yb1, yb2-baseline_coord])
#
