import xml.etree.ElementTree as ET
from typing import Optional


def get_head_attributes(measure: ET.Element, create_if_missing=False) -> Optional[ET.Element]:
    """Returns the head <attributes> element in the tag. This is the first occurence
    of this element, that typically contains clefs, signatures and other head data"""

    _IGNORE = {"print", "barline"}

    position = 0
    for i, child in enumerate(measure):
        position = i
        if child.tag in _IGNORE:
            continue
        elif child.tag == "attributes":
            return child
        else:
            # we hit a note/forward/backup or similar content
            break
    
    if create_if_missing:
        attributes_element = ET.Element("attributes")
        measure.insert(position, attributes_element)
        return attributes_element
    
    return None
