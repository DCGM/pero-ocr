import xml.etree.ElementTree as ET


def sort_attributes(attributes: ET.Element):
    """Sorts children in an <attributes> element"""
    tag_order = [
        # https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/attributes/
        "footnote", "level", "divisions", "key", "time", "staves",
        "part-symbol", "instruments", "clef", "staff-details", "transpose",
        "for-part", "directive", "measure-style"
    ]
    attributes[:] = sorted(
        attributes,
        key=lambda e: (tag_order.index(e.tag), e.get("number", "1"))
    )
