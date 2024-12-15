import argparse
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
import os


modern_universal = ["\ufffd", " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "!", "\u00a1", "?", "\u00bf", ",", "\u2014", ".", "\u00b7", ":", ";", "\\", "_", "&", "#", "@", "(", ")", "[", "]", "{", "}", "+", "-", "*", "/", "\u00b1", "=", "\u2260", "<", ">", "\u2264", "\u2265", "\u03f5", "\u221e", "%", "\u2030", "\u00a3", "\u20ac", "$", "\u00a7", "\u00a9", "\u00ae", "\u2125", "'", "\u2018", "\u2019", "`", "\u201e", "\u201c", "\"", "\u00bb", "\u00ab", "\u203a", "\u2039", "\u261e", "\u261c", "^", "~", "\u00b0", "\u02db", "\u2020", "|", "\u2042", "\u22a5", "\u00ac", "\u00a4", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\u00c1", "\u010c", "\u010e", "\u00c9", "\u011a", "\u00cd", "\u0147", "\u00d3", "\u0158", "\u0160", "\u0164", "\u00da", "\u016e", "\u00dd", "\u017d", "\u00e1", "\u010d", "\u010f", "\u00e9", "\u011b", "\u00ed", "\u0148", "\u00f3", "\u0159", "\u0161", "\u0165", "\u00fa", "\u016f", "\u00fd", "\u017e", "\u00c4", "\u00d6", "\u00dc", "\u1e9e", "\u00e4", "\u00f6", "\u00fc", "\u00df", "s", "\ua75b", "\u0410", "\u0411", "\u0412", "\u0413", "\u0414", "\u0415", "\u0401", "\u0416", "\u0417", "\u0418", "\u0419", "\u041a", "\u041b", "\u041c", "\u041d", "\u041e", "\u041f", "\u0420", "\u0421", "\u0422", "\u0423", "\u0424", "\u0425", "\u0426", "\u0427", "\u0428", "\u0429", "\u042a", "\u042b", "\u042c", "\u042d", "\u042e", "\u042f", "\u0430", "\u0431", "\u0432", "\u0433", "\u0434", "\u0435", "\u0451", "\u0436", "\u0437", "\u0438", "\u0439", "\u043a", "\u043b", "\u043c", "\u043d", "\u043e", "\u043f", "\u0440", "\u0441", "\u0442", "\u0443", "\u0444", "\u0445", "\u0446", "\u0447", "\u0448", "\u0449", "\u044a", "\u044b", "\u044c", "\u044d", "\u044e", "\u044f", "\u0404", "\u0405", "\ua642", "\ua640", "\u0406", "\u0407", "\ua64a", "\u0460", "\u0462", "\ua656", "\u0464", "\u046a", "\u046c", "\u0466", "\u0468", "\u046e", "\u0470", "\u0472", "\u0474", "\u0480", "\u0454", "\u0455", "\ua643", "\ua641", "\u0456", "\u0457", "\ua64b", "\u0461", "\u0463", "\ua657", "\u0465", "\u046b", "\u046d", "\u0467", "\u0469", "\u046f", "\u0471", "\u0473", "\u0475", "\u0481", "\u04e5", "\u04b0", "\u0458", "\u045a", "\u0459", "\u04e3", "\u045d", "\u0453", "\u045f", "\u045b", "\u0452", "\u04d9", "\u04d4", "\u04d5", "\u0391", "\u0392", "\u0393", "\u0394", "\u0395", "\u0396", "\u0397", "\u0398", "\u0399", "\u039a", "\u039b", "\u039c", "\u039d", "\u039e", "\u039f", "\u03a0", "\u03a1", "\u03a3", "\u03f9", "\u03a4", "\u03a5", "\u03a6", "\u03a7", "\u03a8", "\u03a9", "\u03b1", "\u03b2", "\u03b3", "\u03b4", "\u03b5", "\u03b6", "\u03b7", "\u03b8", "\u03b9", "\u03ba", "\u03bb", "\u03bc", "\u03bd", "\u03be", "\u03bf", "\u03c0", "\u03c1", "\u03c3", "\u03c2", "\u03f2", "\u03c4", "\u03c5", "\u03c6", "\u03c7", "\u03c8", "\u03c9", "\u03f3", "\u03dd", "\u00c0", "\u0104", "\u023a", "\u0100", "\u00c5", "\u00c2", "\u01e0", "\u00e0", "\u00e2", "\u00e3", "\u0101", "\u0105", "\u0227", "\u2c65", "\u00e5", "\u0103", "\u1ea1", "\u01ce", "\u0203", "\u1ea3", "\u00c7", "\u0106", "\u010a", "\u00e7", "\u0107", "\u010b", "\u0109", "\u0111", "\u00c8", "\u0118", "\u0246", "\u00ca", "\u00cb", "\u00e8", "\u00ea", "\u00eb", "\u0113", "\u0119", "\u0247", "\u1ebd", "\u0115", "\u0117", "\u0192", "\u0121", "\u01f5", "\u012e", "\u0130", "\u00cc", "\u00cf", "\u00ec", "\u00ee", "\u00ef", "\u012b", "\u0131", "\u012d", "\u01d0", "\u0129", "\u1e31", "\u0141", "\u013d", "\u013a", "\u013e", "\u0142", "\u1e3f", "\u1e41", "\u00d1", "\u00f1", "\u0144", "\u1e45", "\u01f9", "\u00d4", "\u014c", "\u014e", "\u00d2", "\u00f2", "\u00f4", "\u00f5", "\u014f", "\u014d", "\u020d", "\u0151", "\u022f", "\u1ecd", "\u00f8", "\u1d71", "\ua753", "\ua751", "\u1e57", "\u1e55", "\ua759", "\ua757", "\u0154", "\u1e59", "\u0155", "\u015e", "\u015a", "\u0218", "\u015b", "\u1e61", "\u015f", "\u015d", "\u00d9", "\u00f9", "\u00fb", "\u0169", "\u016b", "\u0171", "\u016d", "\u1eef", "\u01d4", "\ua75f", "\u1e82", "\u1e83", "\u1e87", "\u1e81", "\u1e89", "\u00ff", "\u0177", "\u1ef9", "\u1e8f", "\u0233", "\u1ef3", "\u017b", "\u0179", "\u017a", "\u017c", "\u00c6", "\u00e6", "\u0152", "\u0153", "\u0259", "\u014b", "\u0283", "\u0292", "\u0190", "\u00b6", "\u0364", "\u0304", "\u1e98", "\u033e", "\u0303", "\u1e7d", "\u01d2", "\u030a", "\udbc4\udc10", "\udbc4\udc11", "\udbc4\udc12", "\udbc4\udc13", "\udbc4\udc14", "\udbc4\udc15", "\udbc4\udc16", "\udbc4\udc17", "\udbc4\udc18", "\udbc4\udc19", "\udbc4\udc1a", "\udbc4\udc1b", "\udbc4\udc1c", "\udbc4\udc1d", "\udbc4\udc1e", "\udbc4\udc1f", "\udbc4\udc20", "\udbc4\udc21", "\udbc4\udc22", "\udbc4\udc23", "\udbc4\udc24", "\udbc4\udc25", "\udbc4\udc26", "\udbc4\udc27"]
transformer_michal = ["\ufffd", " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "!", "\u00a1", "?", "\u00bf", ",", "\u2014", ".", "\u00b7", ":", ";", "\\", "_", "&", "#", "@", "(", ")", "[", "]", "{", "}", "+", "-", "*", "/", "\u00b1", "=", "\u2260", "<", ">", "\u2264", "\u2265", "\u03f5", "\u221e", "%", "\u2030", "\u00a3", "\u20ac", "$", "\u00a7", "\u00a9", "\u00ae", "\u2125", "'", "\u2018", "\u2019", "`", "\u201e", "\u201c", "\"", "\u00bb", "\u00ab", "\u203a", "\u2039", "\u261e", "\u261c", "^", "~", "\u00b0", "\u02db", "\u2020", "|", "\u2042", "\u22a5", "\u00ac", "\u00a4", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\u00c1", "\u010c", "\u010e", "\u00c9", "\u011a", "\u00cd", "\u0147", "\u00d3", "\u0158", "\u0160", "\u0164", "\u00da", "\u016e", "\u00dd", "\u017d", "\u00e1", "\u010d", "\u010f", "\u00e9", "\u011b", "\u00ed", "\u0148", "\u00f3", "\u0159", "\u0161", "\u0165", "\u00fa", "\u016f", "\u00fd", "\u017e", "\u00c4", "\u00d6", "\u00dc", "\u1e9e", "\u00e4", "\u00f6", "\u00fc", "\u00df", "\u017f", "\ua75b", "\u0410", "\u0411", "\u0412", "\u0413", "\u0414", "\u0415", "\u0401", "\u0416", "\u0417", "\u0418", "\u0419", "\u041a", "\u041b", "\u041c", "\u041d", "\u041e", "\u041f", "\u0420", "\u0421", "\u0422", "\u0423", "\u0424", "\u0425", "\u0426", "\u0427", "\u0428", "\u0429", "\u042a", "\u042b", "\u042c", "\u042d", "\u042e", "\u042f", "\u0430", "\u0431", "\u0432", "\u0433", "\u0434", "\u0435", "\u0451", "\u0436", "\u0437", "\u0438", "\u0439", "\u043a", "\u043b", "\u043c", "\u043d", "\u043e", "\u043f", "\u0440", "\u0441", "\u0442", "\u0443", "\u0444", "\u0445", "\u0446", "\u0447", "\u0448", "\u0449", "\u044a", "\u044b", "\u044c", "\u044d", "\u044e", "\u044f", "\u0404", "\u0405", "\ua642", "\ua640", "\u0406", "\u0407", "\ua64a", "\u0460", "\u0462", "\ua656", "\u0464", "\u046a", "\u046c", "\u0466", "\u0468", "\u046e", "\u0470", "\u0472", "\u0474", "\u0480", "\u0454", "\u0455", "\ua643", "\ua641", "\u0456", "\u0457", "\ua64b", "\u0461", "\u0463", "\ua657", "\u0465", "\u046b", "\u046d", "\u0467", "\u0469", "\u046f", "\u0471", "\u0473", "\u0475", "\u0481", "\u04e5", "\u04b0", "\u0458", "\u045a", "\u0459", "\u04e3", "\u045d", "\u0453", "\u045f", "\u045b", "\u0452", "\u04d9", "\u04d4", "\u04d5", "\u0391", "\u0392", "\u0393", "\u0394", "\u0395", "\u0396", "\u0397", "\u0398", "\u0399", "\u039a", "\u039b", "\u039c", "\u039d", "\u039e", "\u039f", "\u03a0", "\u03a1", "\u03a3", "\u03f9", "\u03a4", "\u03a5", "\u03a6", "\u03a7", "\u03a8", "\u03a9", "\u03b1", "\u03b2", "\u03b3", "\u03b4", "\u03b5", "\u03b6", "\u03b7", "\u03b8", "\u03b9", "\u03ba", "\u03bb", "\u03bc", "\u03bd", "\u03be", "\u03bf", "\u03c0", "\u03c1", "\u03c3", "\u03c2", "\u03f2", "\u03c4", "\u03c5", "\u03c6", "\u03c7", "\u03c8", "\u03c9", "\u03f3", "\u03dd", "\u00c0", "\u0104", "\u023a", "\u0100", "\u00c5", "\u00c2", "\u01e0", "\u00e0", "\u00e2", "\u00e3", "\u0101", "\u0105", "\u0227", "\u2c65", "\u00e5", "\u0103", "\u1ea1", "\u01ce", "\u0203", "\u1ea3", "\u00c7", "\u0106", "\u010a", "\u00e7", "\u0107", "\u010b", "\u0109", "\u0111", "\u00c8", "\u0118", "\u0246", "\u00ca", "\u00cb", "\u00e8", "\u00ea", "\u00eb", "\u0113", "\u0119", "\u0247", "\u1ebd", "\u0115", "\u0117", "\u0192", "\u0121", "\u01f5", "\u012e", "\u0130", "\u00cc", "\u00cf", "\u00ec", "\u00ee", "\u00ef", "\u012b", "\u0131", "\u012d", "\u01d0", "\u0129", "\u1e31", "\u0141", "\u013d", "\u013a", "\u013e", "\u0142", "\u1e3f", "\u1e41", "\u00d1", "\u00f1", "\u0144", "\u1e45", "\u01f9", "\u00d4", "\u014c", "\u014e", "\u00d2", "\u00f2", "\u00f4", "\u00f5", "\u014f", "\u014d", "\u020d", "\u0151", "\u022f", "\u1ecd", "\u00f8", "\u1d71", "\ua753", "\ua751", "\u1e57", "\u1e55", "\ua759", "\ua757", "\u0154", "\u1e59", "\u0155", "\u015e", "\u015a", "\u0218", "\u015b", "\u1e61", "\u015f", "\u015d", "\u00d9", "\u00f9", "\u00fb", "\u0169", "\u016b", "\u0171", "\u016d", "\u1eef", "\u01d4", "\ua75f", "\u1e82", "\u1e83", "\u1e87", "\u1e81", "\u1e89", "\u00ff", "\u0177", "\u1ef9", "\u1e8f", "\u0233", "\u1ef3", "\u017b", "\u0179", "\u017a", "\u017c", "\u00c6", "\u00e6", "\u0152", "\u0153", "\u0259", "\u014b", "\u0283", "\u0292", "\u0190", "\u00b6", "\u0364", "\u0304", "\u1e98", "\u033e", "\u0303", "\u1e7d", "\u01d2", "\u030a", "\udbc4\udc10", "\udbc4\udc11", "\udbc4\udc12", "\udbc4\udc13", "\udbc4\udc14", "\udbc4\udc15", "\udbc4\udc16", "\udbc4\udc17", "\udbc4\udc18", "\udbc4\udc19", "\udbc4\udc1a", "\udbc4\udc1b", "\udbc4\udc1c", "\udbc4\udc1d", "\udbc4\udc1e", "\udbc4\udc1f", "\udbc4\udc20", "\udbc4\udc21", "\udbc4\udc22", "\udbc4\udc23", "\udbc4\udc24", "\udbc4\udc25", "\udbc4\udc26", "\udbc4\udc27"]

for i, (c1, c2) in enumerate(zip(modern_universal, transformer_michal)):
    different = "DIFFERENT" if c1 != c2 else ""
    c1 = c1.encode('utf-8', errors='replace').decode('utf-8')
    c2 = c2.encode('utf-8', errors='replace').decode('utf-8')
    print(f"{i}: {c1} -> {c2} {different}")


# 479: ͤ -> ͤ
exit(1)
def parse_args():
    parser = argparse.ArgumentParser(description='Merge ALTO files')
    parser.add_argument('--source-file', type=str, required=True, help='Take all text lines from this file.')
    parser.add_argument('--merge-file', type=str, required=True, help='Merge text lines from this file.')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output ALTO file')
    parser.add_argument('--image', type=str, help='Path to the image.')
    parser.add_argument('--output-image-path', type=str, default='./', help='Path to store rendered images.')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IoU threshold consider two text lines as the same.')
    return parser.parse_args()


class Textline:
    def __init__(self, x, y, width, height, element: ET.Element):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.element = element

    def intersection_over_union(self, bb):
        x1 = max(self.x, bb.x)
        y1 = max(self.y, bb.y)
        x2 = min(self.x + self.width, bb.x + bb.width)
        y2 = min(self.y + self.height, bb.y + bb.height)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.width * self.height + bb.width * bb.height - intersection
        return intersection / union

    def horizontal_iou(self, bb):
        x1 = max(self.x, bb.x)
        x2 = min(self.x + self.width, bb.x + bb.width)
        intersection = max(0, x2 - x1)
        union = self.width + bb.width - intersection
        return intersection / union


def parse_text_lines(root: ET.Element) -> list[Textline]:
    lines = root.findall('.//{*}TextLine')
    text_lines = []
    for line in lines:
        width = int(line.attrib['WIDTH'])
        height = int(line.attrib['HEIGHT'])
        x = int(line.attrib['HPOS'])
        y = int(line.attrib['VPOS'])
        text_lines.append(Textline(x, y, width, height, line))
    return text_lines


def find_best_iou(source_text_lines: list[Textline], merge_text_line: Textline) -> float:
    best_iou = 0
    for source_text_line in source_text_lines:
        iou = source_text_line.intersection_over_union(merge_text_line)
        best_iou = max(iou, best_iou)
    return best_iou


def find_closest_line(source_text_lines: list[Textline], merge_text_line: Textline, iou_threshold: float = 0.2) -> Textline:
    closest_line = None
    best_distance = 1000000
    for source_text_line in source_text_lines:
        horizontal_iou = source_text_line.horizontal_iou(merge_text_line)
        if horizontal_iou > iou_threshold:
            vertical_distance = abs(source_text_line.y - merge_text_line.y)
            if vertical_distance < best_distance:
                best_distance = vertical_distance
                closest_line = source_text_line
    return closest_line


def add_line_to_line(line_to_add: Textline, line: Textline):
    if line_to_add.y < line.y: # add before line
        line.element.addprevious(line_to_add.element)
    else:
        line.element.addnext(line_to_add.element)


def add_line_as_new_paragraph(merge_text_line: Textline, source_root: ET.Element):
    paragraph = ET.Element('TextBlock')
    paragraph_id = str(len(source_root.findall('.//{*}TextBlock')) + 100)
    paragraph.attrib['ID'] = f"block_{paragraph_id}"
    paragraph.attrib['HPOS'] = str(merge_text_line.x)
    paragraph.attrib['VPOS'] = str(merge_text_line.y)
    paragraph.attrib['WIDTH'] = str(merge_text_line.width)
    paragraph.attrib['HEIGHT'] = str(merge_text_line.height)
    paragraph.attrib['STYLEREFS'] = 'ParagraphStyle0'

    # Add text line into the paragraph
    paragraph.append(merge_text_line.element)

    # Add paragraph into the source root under "PrintSspace" element
    print_space = source_root.find('.//{*}PrintSpace')
    print_space.append(paragraph)

def render_image(source_image_path: str, source_root: ET.Element, output_image_path: str, sufix: str):
    lines = source_root.findall('.//{*}TextLine')
    img = cv2.imread(source_image_path)
    for line in lines:
        x = int(line.attrib['HPOS'])
        y = int(line.attrib['VPOS'])
        width = int(line.attrib['WIDTH'])
        height = int(line.attrib['HEIGHT'])
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)


    output_image_path = os.path.join(output_image_path, f"render_{sufix}.jpg")
    cv2.imwrite(output_image_path, img)

def main():
    args = parse_args()

    source_tree = ET.parse(args.source_file)
    source_root = source_tree.getroot()

    merge_tree = ET.parse(args.merge_file)
    merge_root = merge_tree.getroot()
    source_text_lines = parse_text_lines(source_root)
    merge_text_lines = parse_text_lines(merge_root)



    if args.image:
        image_base_bame = os.path.basename(args.image)
        render_image(args.image, source_root, args.output_image_path, f'{image_base_bame}_source')
        render_image(args.image, merge_root, args.output_image_path, f'{image_base_bame}_merge')

    for merge_text_line in tqdm(merge_text_lines):
        iou = find_best_iou(source_text_lines, merge_text_line)
        if iou > args.iou_threshold:
            continue
        closest_line = find_closest_line(source_text_lines, merge_text_line)
        if closest_line is not None:
            add_line_as_new_paragraph(merge_text_line, source_root)


    source_tree.write(args.output_file)
    if args.image:
        source_tree = ET.parse(args.output_file)
        source_root = source_tree.getroot()
        render_image(args.image, source_root, args.output_image_path, f'{image_base_bame}_output')


if __name__ == '__main__':
    main()

