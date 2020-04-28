# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lxml.etree as ET
import skimage.draw
import skimage.transform
import skimage.measure
import random
import os
import time
import argparse
import cv2
from scipy import misc, ndimage
import sys
import math
import re
import pickle
import imageio
import json

from . import layout
from . import parser_utils as parser


def load_image(path):
    loaded_img_fullsize = imageio.imread(path)
    if len(loaded_img_fullsize.shape) == 2:
        loaded_img_fullsize = np.stack((loaded_img_fullsize, loaded_img_fullsize, loaded_img_fullsize), axis=-1)
    loaded_img_fullsize = loaded_img_fullsize[:,:,:3].copy()

    return(loaded_img_fullsize)


def load_and_preprocess(input_path, img_path, pad, downsample):
    loaded_img_fullsize = load_image(os.path.join(input_path, img_path))
    loaded_img = cv2.resize(loaded_img_fullsize, (0,0), fx=1/downsample, fy=1/downsample)
    img = np.pad(loaded_img, [(pad,pad), (pad,pad), (0,0)], 'constant')
    return loaded_img_fullsize, img


def draw_baselines(img, baselines, upsample=1, color=(255,0,0)):
    for baseline in baselines:
        last = baseline[0]
        cv2.circle(img, (upsample*int(last[0]), upsample*int(last[1])), 3, color, 4)
        for p in baseline[1:]:
            cv2.line(img, (upsample*int(last[0]), upsample*int(last[1])), (upsample*int(p[0]), upsample*int(p[1])), color, 2)
            last = p
        cv2.circle(img, (upsample*int(baseline[-1][0]), upsample*int(baseline[-1][1])), 3, color, 4)

    return img


def save_lines(page_layout, output_path, filename, quality):
    if not os.path.exists(output_path + '/lines'):
        os.mkdir(output_path + '/lines')
    for paragraph_layout in page_layout:
        for crop, name in zip(paragraph_layout.line_crops, paragraph_layout.names):
            imageio.imwrite(output_path + '/lines/' + filename + '-' + name + '.jpg', (crop).astype(np.uint8), quality=quality)


def save_render(page_layout, output_path, name, loaded_img_fullsize):
    if not os.path.exists(output_path + '/output_renders'):
        os.mkdir(output_path + '/output_renders')
    for paragraph_layout in page_layout:
        loaded_img_fullsize = draw_baselines(loaded_img_fullsize, paragraph_layout.baselines, color=(0,0,255))
        loaded_img_fullsize = draw_baselines(loaded_img_fullsize, paragraph_layout.textlines, color=(0,255,0))
        loaded_img_fullsize = draw_baselines(loaded_img_fullsize, paragraph_layout.coords, color=(255,0,0))
    imageio.imwrite(output_path + '/output_renders/' + name + '.jpg', loaded_img_fullsize)


def save_decoding_results(results, output_path, name):
    decoding_path = output_path + '/decoding'
    if not os.path.exists(decoding_path):
        os.mkdir(decoding_path)
    with open("{}/{}.json".format(decoding_path, name), 'w') as handle:
        json.dump(results, handle)


def save_page(name, page_layout, shape, output_path):
    if not os.path.exists(output_path + '/page'):
        os.mkdir(output_path + '/page')
    xml_string = layout_to_xml(name, page_layout, shape)
    with open(output_path + '/page/' + name + '.xml', 'w') as out_f:
        out_f.write(xml_string.decode("utf-8"))


def element_schema(elem):
    if elem.tag[0] == "{":
        schema, _, _ = elem.tag[1:].partition("}")
    else:
        schema = None
    return '{' + schema + '}'


def layout_to_xml(file_name, page_layout, shape):
    height, width = shape
    root = ET.Element("PcGts")
    root.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")

    page = ET.SubElement(root, "Page")
    page.set("imageFilename", file_name)
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))

    for paragraph_layout in page_layout:

        text_region = ET.SubElement(page, "TextRegion")
        coords = ET.SubElement(text_region, "Coords")
        text_region.set("id", paragraph_layout.r_id)
        points = ["{},{}".format(int(x[0]), int(x[1])) for x in paragraph_layout.coords]
        points = " ".join(points)
        coords.set("points", points)

        for baseline, textline, height, transcription, name in zip(paragraph_layout.baselines, paragraph_layout.textlines, paragraph_layout.heights, paragraph_layout.transcriptions, paragraph_layout.names):
            text_line = ET.SubElement(text_region, "TextLine")
            text_line.set("id", name)
            text_line.set("custom", "heights {" + str(int(height[0])) + ", " + str(int(height[1])) + "}")
            coords = ET.SubElement(text_line, "Coords")

            points = np.nan_to_num(textline)
            points = ["{},{}".format(int(x[0]), int(x[1])) for x in points]
            points = " ".join(points)
            coords.set("points", points)

            baseline_element = ET.SubElement(text_line, "Baseline")
            points = ["{},{}".format(int(x[0]), int(x[1])) for x in baseline]
            points = " ".join(points)
            baseline_element.set("points", points)

            text_element = ET.SubElement(text_line, "TextEquiv")
            text_element = ET.SubElement(text_element, "Unicode")
            text_element.text = transcription

    return ET.tostring(root, pretty_print=True, encoding="utf-8")


def xml_to_paragraphs(path_to_xml):
    tree = ET.parse(path_to_xml)
    schema = element_schema(tree.getroot())

    regions_coords = list()
    region_names = list()

    for text_region in tree.iter(schema + 'TextRegion'):
        region_names.append(text_region.attrib['id'])
        region_coords = list()
        for coords in text_region.findall(schema + 'Coords'):
            if 'points' in coords.attrib:
                points_string = coords.attrib['points'].split(' ')
                for points in points_string:
                    x, y = points.split(',')
                    region_coords.append([int(round(float(x))), int(round(float(y)))])
            else:
                for point in coords.findall(schema + 'Point'):
                    x, y = point.attrib['x'], point.attrib['y']
                    region_coords.append([int(round(float(x))), int(round(float(y)))])
        regions_coords.append(region_coords)
    return regions_coords, region_names


def xml_to_layout(path_to_xml, downsample=1):
    page_tree = ET.parse(path_to_xml)
    schema = element_schema(page_tree.getroot())

    page_layout = list()

    for paragraph in page_tree.iter(schema + 'TextRegion'):
        region_coords = list()
        for coords in paragraph.findall(schema + 'Coords'):
            if 'points' in coords.attrib:
                points_string = coords.attrib['points'].split(' ')
                for points in points_string:
                    x, y = points.split(',')
                    region_coords.append([int(round(float(x)/downsample)), int(round(float(y)/downsample))])
            else:
                for point in coords.findall(schema + 'Point'):
                    x, y = point.attrib['x'], point.attrib['y']
                    region_coords.append([int(round(float(x)/downsample)), int(round(float(y)/downsample))])

        paragraph_layout = layout.RegionLayout(paragraph.attrib['id'], np.asarray(region_coords))
        for line in paragraph.iter(schema + 'TextLine'):
            paragraph_layout.names.append(line.attrib['id'])
            heights = re.findall("\d+", line.attrib['custom'])
            if len(re.findall("heights", line.attrib['custom']))==0:
                heights=np.asarray((10,5)).astype(np.int32)
            else:
                heights_array = np.asarray([int(round(float(x)/downsample)) for x in heights])
                if heights_array.shape[0] == 3:
                    heights = np.zeros(2, dtype=np.int32)
                    heights[0] = heights_array[1]
                    heights[1] = heights_array[2] - heights_array[0]
                else:
                    heights = heights_array
            paragraph_layout.heights.append(heights.tolist())

            for baseline in line.findall(schema + 'Baseline'):
                points_string = baseline.attrib['points'].split(' ')
                baseline = list()
                for point in points_string:
                    x, y = point.split(',')
                    baseline.append([int(round(float(x)/downsample)), int(round(float(y)/downsample))])
            paragraph_layout.baselines.append(baseline)

            for textline in line.findall(schema + 'Coords'):
                points_string = textline.attrib['points'].split(' ')
                textline = list()
                for point in points_string:
                    x, y = point.split(',')
                    textline.append([int(round(float(x)/downsample)), int(round(float(y)/downsample))])
            paragraph_layout.textlines.append(textline)

        page_layout.append(paragraph_layout)

    return(page_layout)


def match_annotations(source_xml, target_xml):
    source_page_is_readable = True
    try:
        ET.parse(source_xml)
    except:
        source_page_is_readable = False

    if source_page_is_readable:
        source_root = ET.parse(source_xml).getroot()
        source_schema = element_schema(source_root)
        source_page = source_root.find(source_schema + 'Page')
        source_text_regions = source_page.findall(source_schema + 'TextRegion')

        target_root = ET.parse(target_xml).getroot()
        target_schema = element_schema(target_root)
        target_page = target_root.find(target_schema + 'Page')
        target_text_regions = target_page.findall(target_schema + 'TextRegion')

        for text_region in source_text_regions:
            text_equiv = text_region.find(source_schema + 'TextEquiv')

            if text_equiv is not None:
                region_transcription = text_equiv.find(source_schema + 'Unicode').text
                for coords in text_region.findall(source_schema + 'Coords'):
                    if 'points' in coords.attrib:
                        source_xy = np.asarray(coords.attrib['points'].split(' ')[0].split(','))
                    else:
                        for point in coords.findall(source_schema + 'Point')[:1]:
                            source_xy = np.asarray((int(point.attrib['x']), int(point.attrib['y'])))
                best_match = np.inf
                for r_num, target_region in enumerate(target_text_regions):
                    for coords in target_region.findall(target_schema + 'Coords'):
                        target_xy = np.asarray(coords.attrib['points'].split(' ')[0].split(','))
                    match = math.sqrt(np.sum((int(source_xy[0])-int(target_xy[0]))**2 + (int(source_xy[1])-int(target_xy[1]))**2))
                    if match < best_match:
                        best_match = match
                        best_num = r_num
                matched_region = target_text_regions[best_num]
                target_text_lines = matched_region.findall(target_schema + 'TextLine')
                print('lines: {:01d}, transcriptions: {:01d}'.format(len(target_text_lines), len(region_transcription.split('\n'))))
                if len(target_text_lines) == len(region_transcription.split('\n')):
                    for target_line, transcription_line in zip(target_text_lines, region_transcription.split('\n')):
                        text_element = ET.SubElement(target_line, "TextEquiv")
                        text_element = ET.SubElement(text_element, "Unicode")
                        text_element.text = transcription_line
                else:
                    text_element = ET.SubElement(matched_region, "TextEquiv")
                    text_element = ET.SubElement(text_element, "Unicode")
                    text_element.text = region_transcription

        xml_string = ET.tostring(target_root, pretty_print=True, encoding="utf-8")
        with open(target_xml, 'w') as out_f:
            out_f.write(xml_string.decode("utf-8"))
