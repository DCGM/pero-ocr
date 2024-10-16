# This script should expand functionality of layout.py (region and lines) with options to represent tables.
# The script should also support loading and exporting in different formats
# - original page XML (regions and lines with categories)
# - table page XML with table and cell XML elements
# - label studio JSON format

# Table page XML example:
# <?xml version='1.0' encoding='UTF-8'?>
# <document filename="cTDaR_t00091.jpg">
#   <table>
#     <Coords points="389,379 5811,379 5811,4201 389,4201"/>
#     <cell start-row="0" end-row="0" start-col="0" end-col="0">
#       <Coords points="393,376 394,840 798,844 800,379"/>
#     </cell>
#     <cell start-row="0" end-row="0" start-col="1" end-col="1">
#       <Coords points="800,379 798,844 1177,848 1177,382"/>
#     </cell>
#     ...
#  </table>
# </document>

# From notion:
# - Page = page, 1 velký region = tabulka, text line=buňka
# - Tabulka = region s kategorií “tabulka”
# - Typ buňky (textLine) (heading, ID, normal) - **může být category**
# - Vztahy: vertical + horizontal (ID odkaz na následovníka?? - label studio má jakoby vztah ID1 a ID2 zvlášť)
# - Jak udělat hodnoty col a row
#     - Musí tam být nějaká abstraktní struktura tabulky asi…

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.core.force_alignment import align_text
from pero_ocr.core.confidence_estimation import get_line_confidence
from pero_ocr.core.arabic_helper import ArabicHelper


class Cell(TextLine):
    ...

class Table(RegionLayout):
    def __init__(self):
        super().__init__()
        self.lines = []


    def from_label_studio_json(self, json):
        ...

    def to_label_studio_json(self):
        ...