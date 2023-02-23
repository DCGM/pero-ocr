#!/usr/bin/env python3

import argparse
from pero_ocr.document_ocr import pdf_production

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml')
    parser.add_argument('image')
    parser.add_argument('out')
    args = parser.parse_args()

    merger = pdf_production.Merger()
    merger.merge(args.xml, args.image, args.out)


if __name__ == '__main__':
    main()
