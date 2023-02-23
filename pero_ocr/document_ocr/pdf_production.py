import lxml.etree as ET
import os
import shutil
import subprocess
import sys
import tempfile

import PIL
import fpdf

from pero_ocr.document_ocr.layout import PageLayout, element_schema


class Merger:
    def __init__(self):
        pass

    def merge(self, xml_path, img_path, out_path):
        xml_processor = self.get_xml_processor(xml_path)

        with tempfile.TemporaryDirectory() as tmp_dir_fn:
            resolution = 72.0
            img = PIL.Image.open(img_path)
            img_pdf_fn = os.path.join(tmp_dir_fn, 'img.pdf')
            img.save(img_pdf_fn, 'PDF', resolution=resolution)

            xml_pdf = xml_processor(xml_path)
            xml_pdf_fn = os.path.join(tmp_dir_fn, 'ocr.pdf')
            xml_pdf_fn = 'ocr.pdf'
            xml_pdf.output(xml_pdf_fn)

            merge_2_pdfs(img_pdf_fn, xml_pdf_fn, out_path, tmp_dir_fn)

    def get_xml_processor(self, xml_path):
        page_tree = ET.parse(xml_path)
        schema = element_schema(page_tree.getroot())

        if 'alto' in schema.lower():
            return pdf_from_alto_xml
        elif 'page' in schema.lower():
            return pdf_from_page_xml
        else:
            raise ValueError(f'Unsupported XML type {schema}')


def parse_page_xml(xml_path):
    layout = PageLayout()
    layout.from_pagexml(xml_path)

    h, w = layout.page_size

    return (h, w), layout.lines_iterator()


def pdf_from_page_xml(xml_path):
    (h, w), line_iterator = parse_page_xml(xml_path)

    pdf_writer = PDFWriter(w, h)

    for line in line_iterator:
        left = line.baseline[0, 0]
        right = line.baseline[-1, 0]
        bottom = line.baseline[0, 1]

        width = right - left

        pdf_writer.put_line(left, bottom, width, line.heights[0], line.transcription)

    return pdf_writer.pdf


def pdf_from_alto_xml(xml_path):
    page_tree = ET.parse(xml_path)
    schema = element_schema(page_tree.getroot())
    root = page_tree.getroot()

    layout = root.findall(schema + 'Layout')[0]
    page = layout.findall(schema + 'Page')[0]

    height = int(page.attrib['HEIGHT'])
    width = int(page.attrib['WIDTH'])

    pdf_writer = PDFWriter(width, height)

    print_space = page.findall(schema + 'PrintSpace')[0]
    lines = []
    for region in print_space.iter(schema + 'TextBlock'):
        for line in region.iter(schema + 'TextLine'):
            line_left = int(line.attrib['HPOS'])
            line_bottom = int(line.attrib['BASELINE'])
            line_width = int(line.attrib['WIDTH'])
            line_height = int(line.attrib['HEIGHT'])

            words = ' '.join(word.get('CONTENT') for word in line.iter(schema + 'String'))
            pdf_writer.put_line(line_left, line_bottom, line_width, line_height, words)

    return pdf_writer.pdf


class PDFWriter:
    def __init__(self, width, height):
        self.pdf = fpdf.FPDF(orientation='Portrait', unit='pt', format=(width, height))
        self.pdf.add_font('DeJavu', '', '/usr/share/fonts/dejavu/DejaVuSans.ttf', uni=True)
        self.pdf.add_page()

        self.font = 'dejavu'

    def put_line(self, left, bottom, width, height, text):
        self.pdf.set_xy(left, bottom)
        font_size = self.get_font_size(height, width, text)
        self.pdf.set_font(self.font, '', font_size)

        default_text_width = self.pdf.get_string_width(text)
        self.pdf.set_stretching(100.0 * width / default_text_width)
        self.pdf.cell(width, txt=text)
        self.pdf.set_stretching(100.0)

    def get_font_size(self, line_height, line_width, text):
        h_estimate = line_height 

        def height_ok(height):
            self.pdf.set_font(self.font, '', height)
            return self.pdf.get_string_width(text) <= line_width

        if not height_ok(h_estimate):
            h_estimate = bisect_max(0.1, h_estimate, height_ok, 0.25)

        return h_estimate

def bisect_max(lower, upper, predicate, max_error):
    assert upper > lower
    assert predicate(lower)
    assert not predicate(upper)

    while upper - lower > max_error:
        estimate = (upper + lower) / 2
        if predicate(estimate):
            lower = estimate
        else:
            upper = estimate

    return lower


def merge_2_pdfs(img_pdf_fn, xml_pdf_fn, out, tmp_dir_fn):
    core_fn = 'py_template'
    tex_fn = os.path.join(tmp_dir_fn, f'{core_fn}.tex')
    with open(tex_fn, 'w') as f:
        f.write(PREAMBULE)
        f.write(DOCUMENT_BEGIN)
        f.write(print_page_with_overlay(img_pdf_fn, xml_pdf_fn))
        f.write(DOCUMENT_END)

    subprocess.run(
        [shutil.which('pdflatex'), '-output-directory', f'{tmp_dir_fn}', tex_fn],
        stdout=subprocess.DEVNULL
    )

    shutil.copyfile(os.path.join(tmp_dir_fn, f'{core_fn}.pdf'), out)


def print_page_with_overlay(img_path, ocr_path):
    return (
        "\\begin{overpic}{" + ocr_path + "}\n"
        "    \\put(0, 0){\n"
        "        \\begin{ocg}{OCR outputs}{ocr}{1}\n"
        "            \\includegraphics{" + img_path + "}\n"
        "        \\end{ocg}\n"
        "    }\n"
        "\\end{overpic}\n"
    )


PREAMBULE = (
   "\\documentclass{standalone}\n"
   "\\usepackage{ocgx}\n"
   "\\usepackage{graphicx}\n"
   "\n"
   "\\usepackage{overpic}\n"
   "\n"
)

DOCUMENT_BEGIN = (
    "\\begin{document}\n"
    "\n"
)

DOCUMENT_END = (
    "\n"
    "\\end{document}\n"
)
