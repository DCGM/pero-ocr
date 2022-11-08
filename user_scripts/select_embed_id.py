#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import configparser
import argparse
import cv2

import time
import random
import Levenshtein
import numpy as np
import lmdb
import sys
from sklearn.cluster import KMeans

from safe_gpu import safe_gpu

from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser
from pero_ocr.utils import compose_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to input config file.')
    parser.add_argument('-i', '--input-image-path', help='')
    parser.add_argument('-x', '--input-xml-path', help='')
    parser.add_argument('-l', '--input-lmdb-path', help='')
    parser.add_argument('-t', '--input-data-path', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--n-clusters', type=int, default=100, help='')
    parser.add_argument('--n-lines', type=int, default=100, help='')
    parser.add_argument('--mean-cluster-embed', action='store_true', help='Do not pick representative embed for'
                                                                          'cluster randomly, instead pick the nearest'
                                                                          'to the cluster center.')
    parser.add_argument('--representative-embed-ids', type=str, help='Clustering is not performed.')
    parser.add_argument('--set-gpu', action='store_true', help='Sets visible CUDA device to first unused GPU.')
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    config = configparser.ConfigParser()
    config.read(args.config)

    if args.set_gpu:
        gpu_owner = safe_gpu.GPUOwner()

    page_parser = PageParser(config, config_path=os.path.dirname(args.config))
    page_parser.ocr.ocr_engine.batch_size = args.batch_size
    page_parser.ocr.ocr_engine.max_input_horizontal_pixels = 480 * args.batch_size

    if args.input_image_path is not None and args.input_xml_path is not None:
        line_crops, gts = get_line_crops_and_transcriptions_from_images_and_xmls(page_parser, args.input_image_path,
                                                                                 args.input_xml_path, args.n_lines)
    elif args.input_lmdb_path is not None and args.input_data_path is not None:
        line_crops, gts = get_line_crops_and_transcriptions_from_lmdb_and_data(args.input_lmdb_path,
                                                                               args.input_data_path, args.n_lines)
    else:
        print("Invalid inputs.")
        sys.exit(-1)

    t_start = time.time()

    if args.representative_embed_ids is not None:
        representative_embeddings_ids = [int(x) for x in args.representative_embed_ids.split(",")]
    elif args.n_clusters < page_parser.ocr.ocr_engine.embed_num:
        representative_embeddings_ids = select_representative_embeddings(page_parser.ocr.ocr_engine, args.n_clusters)
    else:
        representative_embeddings_ids = list(range(page_parser.ocr.ocr_engine.embed_num))
    print("REPRESENTATIVE EMBEDDING IDS: {}".format(",".join([str(x) for x in representative_embeddings_ids])))
    print()

    embed_id_cers = []
    for embed_id in representative_embeddings_ids:
        page_parser.ocr.ocr_engine.embed_id = embed_id

        t1 = time.time()

        transcriptions, _, _ = page_parser.ocr.ocr_engine.process_lines(line_crops, no_logits=True)
        ref_char_sum = 0
        ref_gt_char_dist = 0
        if args.out is not None:
            with open(os.path.join(args.out, "{}.gt".format(embed_id)), "w") as f:
                f.writelines(["{}\n".format(x) for x in gts])
            with open(os.path.join(args.out, "{}.trans".format(embed_id)), "w") as f:
                f.writelines(["{}\n".format(x) for x in transcriptions])
        for gt, trans in zip(gts, transcriptions):
            ref_char_sum += len(gt)
            ref_gt_char_dist += Levenshtein.distance(gt, trans)
        if ref_char_sum > 0:
            embed_id_cers.append(100.0 * ref_gt_char_dist / ref_char_sum)
            print(
                f'{embed_id} {embed_id_cers[-1]:.2f} % CER [ {ref_gt_char_dist} / {ref_char_sum} ] Time: {time.time() - t1:.2f}')
        else:
            embed_id_cers.append(1000000000000)
            print(f'{embed_id} N/A % CER [ {ref_gt_char_dist} / {ref_char_sum} ] Time: {time.time() - t1:.2f}')

    embed_id_with_minimal_cer = representative_embeddings_ids[np.argmin(embed_id_cers)]

    print()
    print(f'SELECTED EMBED ID WITH MIN CER: {embed_id_with_minimal_cer}')
    print(f'PROCESSING TIME {(time.time() - t_start)}')

    page_parser.ocr.ocr_engine.config["embed_id"] = str(embed_id_with_minimal_cer)
    with open(compose_path(config['OCR']['OCR_JSON'], os.path.dirname(args.config)), 'w', encoding='utf8') as f:
        json.dump(page_parser.ocr.ocr_engine.config, f, indent=4)


def select_representative_embeddings(ocr_engine, n_clusters, mean_cluster_embedding=False):
    for name, child in ocr_engine.model.named_modules():
        if name == "embeddings_layer" and child.original_name == "Embedding":
            embeddings_layer = child
            break
    embeddings = next(embeddings_layer.parameters())
    embeddings = embeddings.cpu().detach().numpy()
    print("EMBEDDINGS SHAPE: {}".format(embeddings.shape))
    representative_embeddings_ids = []
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    for i in range(n_clusters):
        if mean_cluster_embedding:
            representative_embeddings_ids.append(np.argmin(embeddings - embeddings[np.where(kmeans.labels_ == i)[0]].mean()))
        else:
            representative_embeddings_ids.append(np.random.choice(np.where(kmeans.labels_ == i)[0]))
    return representative_embeddings_ids


def get_line_crops_and_transcriptions_from_images_and_xmls(page_parser, input_image_path, input_xml_path, n_lines,
                                                           max_lines=5000):
    ignored_extensions = ['', '.xml', '.logits']
    images_to_process = [f for f in os.listdir(input_image_path) if
                         os.path.splitext(f)[1].lower() not in ignored_extensions]
    page_ids_to_process = [os.path.splitext(os.path.basename(file))[0] for file in images_to_process]

    valid_lines = []
    valid_lines_counter = 0
    for image_file, page_id in zip(images_to_process, page_ids_to_process):
        page_layout = PageLayout(file=os.path.join(input_xml_path, page_id + '.xml'))
        for line in page_layout.lines_iterator():
            if line.transcription != "" and line.transcription is not None:
                valid_lines.append([image_file, line])
                valid_lines_counter += 1
                if valid_lines_counter == max_lines:
                    break
        if valid_lines_counter == max_lines:
            break
    random.shuffle(valid_lines)
    valid_lines = valid_lines[:n_lines]

    image_file_to_lines = {}
    for valid_line in valid_lines:
        image_file, line = valid_line
        if image_file in image_file_to_lines:
            image_file_to_lines[image_file].append(line)
        else:
            image_file_to_lines[image_file] = [line]

    line_crops = []
    transcriptions = []
    for image_file, lines in image_file_to_lines.items():
        image = cv2.imread(os.path.join(input_image_path, image_file), 1)
        if image is None:
            raise Exception(f'Unable to read image "{os.path.join(input_image_path, image_file)}"')

        page_parser.line_cropper.crop_lines(image, lines)
        for line in lines:
            line_crops.append(line.crop)
            transcriptions.append(line.transcription)

    return line_crops, transcriptions


def get_line_crops_and_transcriptions_from_lmdb_and_data(input_lmdb_path, input_data_path, n_lines,
                                                         max_lines=5000):
    print(input_lmdb_path)
    print(input_data_path)
    line_crops_and_transcriptions = []
    valid_lines_counter = 0
    txn = lmdb.open(input_lmdb_path, readonly=True).begin()
    with open(input_data_path) as f:
        lines = f.readlines()
    for l in lines:
        line_id, embed_id, transcription = l.split(" ", 2)
        data = txn.get(line_id.encode())
        if data is None:
            print(f"Unable to load image '{line_id}' specified in '{input_data_path}' from DB '{input_lmdb_path}'.")
        line_crop = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
        if line_crop is None:
            print(f"Unable to decode image '{line_id}'.")
        else:
            if transcription != "":
                line_crops_and_transcriptions.append((line_crop, transcription))
                valid_lines_counter += 1
                if valid_lines_counter == max_lines:
                    break

    random.shuffle(line_crops_and_transcriptions)
    line_crops_and_transcriptions = line_crops_and_transcriptions[:n_lines]

    line_crops = []
    transcriptions = []
    for line_crop, transcription in line_crops_and_transcriptions:
        line_crops.append(line_crop)
        transcriptions.append(transcription)

    return line_crops, transcriptions


def load_images_and_page_layouts(input_image_path, input_xml_path):
    ignored_extensions = ['', '.xml', '.logits']
    images_to_process = [f for f in os.listdir(input_image_path) if
                         os.path.splitext(f)[1].lower() not in ignored_extensions]
    images_to_process = sorted(images_to_process)
    ids_to_process = [os.path.splitext(os.path.basename(file))[0] for file in images_to_process]
    images = []
    page_layouts = []
    for index, (file_id, image_file_name) in enumerate(zip(ids_to_process, images_to_process)):
        image = cv2.imread(os.path.join(input_image_path, image_file_name), 1)
        if image is None:
            raise Exception(f'Unable to read image "{os.path.join(input_image_path, image_file_name)}"')
        images.append(image)
        page_layout = PageLayout(file=os.path.join(input_xml_path, file_id + '.xml'))
        page_layouts.append(page_layout)

    return images, page_layouts


if __name__ == "__main__":
    main()
