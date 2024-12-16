import argparse
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
import os


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
    max_distance = 6 * merge_text_line.height
    for source_text_line in source_text_lines:
        horizontal_iou = source_text_line.horizontal_iou(merge_text_line)
        if horizontal_iou > iou_threshold:
            vertical_distance = abs(source_text_line.y - merge_text_line.y)
            if vertical_distance < best_distance and vertical_distance < max_distance:
                best_distance = vertical_distance
                closest_line = source_text_line
    return closest_line


def split_string_to_charachters(line: Textline) -> Textline:
    new_children = []
    for word in line.element:
        if word.tag != 'String':
            new_children.append(word)
        if len(text) == 1:
            new_children.append(word)

        text = word.attrib['CONTENT']
        x = int(word.attrib['HPOS'])
        y = int(word.attrib['VPOS'])
        width = int(word.attrib['WIDTH']) / len(text)
        height = int(word.attrib['HEIGHT'])

        for i, char in enumerate(text):
            new_word = ET.Element('String')
            new_word.attrib['CONTENT'] = char
            new_word.attrib['HPOS'] = str(int(x + i * width + 0.5))
            new_word.attrib['VPOS'] = str(y)
            new_word.attrib['WIDTH'] = str(width)
            new_word.attrib['HEIGHT'] = str(height)
            new_children.append(new_word)

    line.element.clear()
    for child in new_children:
        line.element.append(child)

    return line


def find_parent(root: ET.Element, target: ET.Element) -> tuple[int, ET.Element]:
    """Recursively find the parent of a target element."""
    for paragaraph in root.findall('.//{*}TextBlock'):
        for i, child in enumerate(paragaraph):
            if child is target:  # If the target is found as a child
                return i, paragaraph
    raise ValueError('Parent not found')

def add_line_to_paragraph(line_to_add: Textline, line: Textline, source_root: ET.Element):
    index, parrent = find_parent(source_root, line.element)

    if line_to_add.y < line.y: # add before line
        parrent.insert(index, line_to_add.element)
    else:
        parrent.insert(index + 1, line_to_add.element)

    x = int(parrent.attrib['HPOS'])
    y = int(parrent.attrib['VPOS'])
    x2 = x + int(parrent.attrib['WIDTH'])
    y2 = y + int(parrent.attrib['HEIGHT'])
    x = min(x, line_to_add.x)
    y = min(y, line_to_add.y)
    x2 = max(x2, line_to_add.x + line_to_add.width)
    y2 = max(y2, line_to_add.y + line_to_add.height)
    width = x2 - x
    height = y2 - y
    parrent.attrib['HPOS'] = str(x)
    parrent.attrib['VPOS'] = str(y)
    parrent.attrib['WIDTH'] = str(width)
    parrent.attrib['HEIGHT'] = str(height)


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
    import seaborn as sns
    img = cv2.imread(source_image_path)
    paragraphs = source_root.findall('.//{*}TextBlock')
    paragraph_colors = sns.color_palette("hls", len(paragraphs))
    # shuffle colors
    import random
    paragraph_colors = random.sample(paragraph_colors, len(paragraphs))
    for color, paragraph in zip(paragraph_colors, paragraphs):
        color = tuple(int(x * 255) for x in color)
        x = int(paragraph.attrib['HPOS'])
        y = int(paragraph.attrib['VPOS'])
        width = int(paragraph.attrib['WIDTH'])
        height = int(paragraph.attrib['HEIGHT'])
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 0), 3)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        for line in paragraph.findall('.//{*}TextLine'):
            x = int(line.attrib['HPOS'])
            y = int(line.attrib['VPOS'])
            width = int(line.attrib['WIDTH'])
            height = int(line.attrib['HEIGHT'])
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)

    output_image_path = os.path.join(output_image_path, f"render_{sufix}.jpg")
    cv2.imwrite(output_image_path, img)


def main():
    args = parse_args()

    source_tree = ET.parse(args.source_file)
    source_root = source_tree.getroot()

    merge_tree = ET.parse(args.merge_file)
    merge_root = merge_tree.getroot()
    merge_text_lines = parse_text_lines(merge_root)
    image_base_bame = os.path.basename(args.image)

    #if args.image:
    #    image_base_bame = os.path.basename(args.image)
    #    render_image(args.image, source_root, args.output_image_path, f'{image_base_bame}_source')
    #    render_image(args.image, merge_root, args.output_image_path, f'{image_base_bame}_merge')
    for merge_text_line in tqdm(merge_text_lines):
        source_text_lines = parse_text_lines(source_root)
        iou = find_best_iou(source_text_lines, merge_text_line)
        if iou > args.iou_threshold:
            continue
        merge_text_line = split_string_to_charachters(merge_text_line)
        closest_line = find_closest_line(source_text_lines, merge_text_line)
        if closest_line:
            add_line_to_paragraph(merge_text_line, closest_line, source_root)
        else:
            add_line_as_new_paragraph(merge_text_line, source_root)

    source_tree.write(args.output_file, encoding='utf-8', xml_declaration=True)
    if args.image:
        source_tree = ET.parse(args.output_file)
        source_root = source_tree.getroot()
        render_image(args.image, source_root, args.output_image_path, f'{image_base_bame}_output')


if __name__ == '__main__':
    main()

