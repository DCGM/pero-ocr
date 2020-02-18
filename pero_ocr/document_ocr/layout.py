import re
import pickle
from io import BytesIO
from datetime import datetime

import numpy as np
import lxml.etree as ET
import cv2

from ocr_engine.softmax import softmax
from force_alignment import force_align

class TextLine(object):
    def __init__(self, id=None, baseline=None, polygon=None, heights=None, transcription=None, logits=None, crop=None, characters=None):
        self.id = id
        self.baseline = baseline
        self.polygon = polygon
        self.heights = heights
        self.transcription = transcription
        self.logits = logits
        self.crop = crop
        self.characters = characters


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

    def to_altoxml_string(self):
        NSMAP = {"xlink": 'http://www.w3.org/1999/xlink',
                 "xsi": 'http://www.w3.org/2001/XMLSchema-instance'}
        root = ET.Element("alto", nsmap=NSMAP)
        root.set("xmlns", "http://www.loc.gov/standards/alto/ns-v2#")

        description = ET.SubElement(root, "Description")
        measurement_unit = ET.SubElement(description, "MeasurementUnit")
        measurement_unit.text = "pixel"
        ocr_processing = ET.SubElement(description, "OCRProcessing")
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
        page.set("ID", self.id)
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
            text_block.set("ID", block.id)

            text_block_height = max(block.polygon[:,1]) - min(block.polygon[:,1])
            text_block.set("HEIGHT", str(text_block_height))

            text_block_width = max(block.polygon[:,0]) - min(block.polygon[:,0])
            text_block.set("WIDTH", str(text_block_width))

            text_block_vpos = min(block.polygon[:,0])
            text_block.set("VPOS", str(text_block_vpos))

            text_block_hpos = min(block.polygon[:,1])
            text_block.set("HPOS", str(text_block_hpos))

            print_space_height = max([print_space_vpos+print_space_height, text_block_vpos+text_block_height])
            print_space_width = max([print_space_hpos+print_space_width, text_block_hpos+text_block_width])
            print_space_vpos = min([print_space_vpos, text_block_vpos])
            print_space_hpos = min([print_space_hpos, text_block_hpos])
            print_space_height = print_space_height - print_space_vpos
            print_space_width = print_space_width - print_space_hpos

            for l, line in enumerate(block.lines):
                text_line = ET.SubElement(text_block, "TextLine")

                text_line_baseline = int(np.average(np.array(line.baseline)[:,0]))
                text_line.set("BASELINE", str(text_line_baseline))
                text_line_height = int(np.average((np.array(line.polygon)[:, 0])[len(line.polygon)//2:]))-int(np.average((np.array(line.polygon)[:,0])[:len(line.polygon)//2]))
                text_line.set("HEIGHT", str(text_line_height))
                text_line_width = max((np.array(line.polygon)[:, 1]))-min(np.array(line.polygon)[:, 1])
                text_line.set("WIDTH", str(text_line_width))
                text_line_vpos = int(np.average((np.array(line.polygon)[:, 0])[:len(line.polygon) // 2]))
                text_line.set("VPOS", str(text_line_vpos))
                text_line_hpos = min(np.array(line.polygon)[:, 1])
                text_line.set("HPOS", str(text_line_hpos))

                logits = np.array(line.logits[0].todense())

                output = softmax(np.array(logits), axis=1)
                aligned = force_align(-np.log(output), array, 254)

                for w, word in enumerate(line.transcription.split()):
                    string = ET.SubElement(text_line, "String")
                    string.set("CONTENT", word)

                    string.set("HEIGHT", "")
                    string.set("WIDTH", "")
                    string.set("VPOS", "")
                    string.set("HPOS", "")
                    if w != (len(line.transcription.split())-1):
                        space = ET.SubElement(text_line, "SP")
                        space.set("WIDTH", "")
                        space.set("VPOS", "")
                        space.set("HPOS", "")

        top_margin.set("HEIGHT", "{}" .format(print_space_vpos))
        top_margin.set("WIDTH", "{}" .format(self.page_size[1]))
        top_margin.set("VPOS", "0")
        top_margin.set("HPOS", "0")

        left_margin.set("HEIGHT", "{}" .format(self.page_size[0]))
        left_margin.set("WIDTH", "{}" .format(print_space_hpos))
        left_margin.set("VPOS", "0")
        left_margin.set("HPOS", "0")

        right_margin.set("HEIGHT", "{}" .format(self.page_size[0]))
        right_margin.set("WIDTH", "{}" .format(self.page_size[1]-(print_space_hpos+print_space_width)))
        right_margin.set("VPOS", "0")
        right_margin.set("HPOS", "{}" .format(print_space_hpos+print_space_width))

        bottom_margin.set("HEIGHT", "{}" .format(self.page_size[0]-(print_space_vpos+print_space_height)))
        bottom_margin.set("WIDTH", "{}" .format(self.page_size[1]))
        bottom_margin.set("VPOS", "{}" .format(print_space_vpos+print_space_height))
        bottom_margin.set("HPOS", "0")

        print_space.set("HEIGHT", str(print_space_height))
        print_space.set("WIDTH", str(print_space_width))
        print_space.set("VPOS", str(print_space_vpos))
        print_space.set("HPOS", str(print_space_hpos))

        return ET.tostring(root, pretty_print=True, encoding="utf-8").decode("utf-8")

    def to_altoxml(self, file_name):
        alto_string = self.to_altoxml_string()
        with open(file_name, 'w', encoding='utf-8') as out_f:
            out_f.write(alto_string)

    def from_altoxml_string(self, pagexml_string):
        self.from_pagexml(BytesIO(pagexml_string))

    def from_altoxml(self, file):
        page_tree = ET.parse(file)
        schema = element_schema(page_tree.getroot())
        root = page_tree.getroot()

        layout = root.findall(schema+'Layout')[0]
        page = layout.findall(schema+'Page')[0]

        self.id = page.attrib['ID']
        self.page_size = (int(page.attrib['HEIGHT']), int(page.attrib['WIDTH']))

        print_space = page.findall(schema+'PrintSpace')[0]
        for region in print_space.iter(schema + 'TextBlock'):
            region_coords = list()
            region_coords.append([int(region.get('VPOS')), int(region.get('HPOS'))])
            region_coords.append([int(region.get('WIDTH')), int(region.get('HPOS'))])
            region_coords.append([int(region.get('WIDTH')), int(region.get('HEIGHT'))])
            region_coords.append([int(region.get('VPOS')), int(region.get('HEIGHT'))])

            region_layout = RegionLayout(region.attrib['ID'], np.asarray(region_coords))

            for line in region.iter(schema + 'TextLine'):
                new_textline = TextLine(baseline=[[int(line.attrib['BASELINE']), int(line.attrib['HPOS'])], [int(line.attrib['BASELINE']), int(line.attrib['HPOS']) + int(line.attrib['WIDTH'])]], polygon=[])
                new_textline.heights = [int(line.attrib['BASELINE'])-int(line.attrib['VPOS']), int(line.attrib['HEIGHT'])+int(line.attrib['VPOS'])-int(line.attrib['BASELINE'])]
                new_textline.polygon.append([int(line.attrib['VPOS']), int(line.attrib['HPOS'])])
                new_textline.polygon.append([int(line.attrib['VPOS']), int(line.attrib['HPOS'])+int(line.attrib['WIDTH'])])
                new_textline.polygon.append([int(line.attrib['VPOS'])+int(line.attrib['HEIGHT']), int(line.attrib['HPOS'])+int(line.attrib['WIDTH'])])
                new_textline.polygon.append([int(line.attrib['VPOS'])+int(line.attrib['HEIGHT']), int(line.attrib['HPOS'])])
                word = ''
                start = True
                for text in line.iter(schema + 'String'):
                    if start:
                        start = False
                        word = word + text.get('CONTENT')
                    else:
                        word = word + " " + text.get('CONTENT')

                region_layout.lines.append(word)

            self.regions.append(region_layout)


    def save_logits(self, file_name):
        """Save page logits as a pickled dictionary of sparse matrices.
        :param file_name: to pickle into.
        """
        logits = []
        for region in self.regions:
            for line in region.lines:
                if line.logits is None:
                    raise Exception(f'Missing logits for line {line.id}.')
            logits += [(line.id, line.logits, line.characters) for line in region.lines]
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
    #test_layout = PageLayout(file='/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624_gt.xml')
    #test_layout.to_pagexml('/mnt/matylda1/ikodym/junk/refactor_test/test.xml')
    #image = cv2.imread('/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624.jpg')
    #test_layout.render_to_image(image, '/mnt/matylda1/ikodym/junk/refactor_test/')

    #test_layout.from_pagexml('C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/8e41ecc2-57ed-412a-aa4f-d945efa7c624_gt.xml')
    #test_layout.load_logits('C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/8e41ecc2-57ed-412a-aa4f-d945efa7c624.logits')

    def save():
        test_layout = PageLayout()
        test_layout.from_pagexml('C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/de0392e9-cdc2-42eb-aa74-a8c086c98bec.xml')
        test_layout.load_logits('C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/de0392e9-cdc2-42eb-aa74-a8c086c98bec.logits')
        #for i, item in enumerate(test_layout.regions):
        #    for e, elem in enumerate(item.lines):
        #        print(elem.logits)

        image = cv2.imread("C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/de0392e9-cdc2-42eb-aa74-a8c086c98bec.jpg")
        cv2.imwrite("C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/test.jpg", test_layout.render_to_image(image))
        #string = test_layout.to_altoxml_string()

        #test_layout.to_altoxml('test_alto.xml')

        print("XXX")
        print(test_layout.regions[0].lines[1])
        print(test_layout.regions[0].lines[0].baseline)
        print(test_layout.regions[0].lines[0].polygon)
        print(test_layout.regions[0].lines[0].heights)
        print(test_layout.regions[0].lines[0].crop)
        print("XXX")
    def load():
        test_layout = PageLayout()
        test_layout.from_altoxml('C:/Users/LachubCz_NTB/Documents/GitHub/pero-ocr/pero_ocr/document_ocr/test_alto.xml')

    save()
    #load()


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
