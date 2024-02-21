#!/usr/bin/env python3

import argparse
import os
import logging
import re
from pero_ocr.document_ocr import pdf_production

img_extensions = ['jpg', 'jpeg', 'png']
img_regex = re.compile(f'.*\.({"|".join(img_extensions)})', re.IGNORECASE)


def drop_suffix(fn):
    return fn.rsplit('.', maxsplit=1)[0]


def discover_files(folder, is_relevant, key_postprocess=lambda x: x):
    fns = [
        fn
        for fn in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fn)) and is_relevant(fn)
    ]
    return {key_postprocess(drop_suffix(fn)): os.path.join(folder, fn) for fn in fns}


def intersect_keys(dict_a, dict_b):
    intersection = [k for k in dict_a if k in dict_b]

    if len(dict_a) != len(intersection) or len(dict_b) != len(intersection):
        non_matched = [v for k, v in dict_a.items() if k not in intersection] + [v for k, v in dict_b.items() if k not in intersection]
        logging.warning(f'Not matched: {non_matched}')

    return intersection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-file', action='store_true',
                        help='Interpret paths as single files instead of whole folders')
    parser.add_argument('--xml-drop-suffix', default='',
                        help='String to drop from xml filename, e.g. give "_alto" to normalize "043-099_alto.xml"')
    parser.add_argument('xml')
    parser.add_argument('image')
    parser.add_argument('out')
    args = parser.parse_args()

    merger = pdf_production.Merger()

    if args.single_file:
        merger.merge(args.xml, args.image, args.out)
    else:
        xml_dict = discover_files(args.xml, lambda fn: fn.endswith('.xml'), lambda fn: fn.removesuffix(args.xml_drop_suffix))
        img_dict = discover_files(args.image, lambda fn: img_regex.fullmatch(fn) is not None)

        matched_keys = intersect_keys(xml_dict, img_dict)

        os.makedirs(args.out, exist_ok=True)

        for k in matched_keys:
            logging.info(f'Merging {k}')
            merger.merge(
                xml_dict[k],
                img_dict[k],
                os.path.join(args.out, f'{k}.pdf'),
            )


if __name__ == '__main__':
    main()
