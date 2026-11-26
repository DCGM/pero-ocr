import xml.etree.ElementTree as ET
from fractions import Fraction
import math
from .get_head_attributes import get_head_attributes
from .sort_attributes import sort_attributes


def fractional_durations_to_actual(part: ET.Element):
    assert part.tag == "part"
    
    if len(part) == 0:
        return

    # make sure there are no divisions
    divisions = part.findall("measure/attributes/divisions")
    assert len(divisions) == 0, "Fractional part should not contain <divisions>"

    # get all duration values
    duration_values = set(
        Fraction(e.text) for e in part.iter("duration")
    )

    # compute divisions
    denominators = [v.denominator for v in duration_values]
    
    # LCM magic algorithm
    lcm = denominators[0]
    for d in denominators[1:]:
        lcm = lcm // math.gcd(lcm, d) * d
    divisions = lcm

    # write divisions element
    attributes_element = get_head_attributes(part[0], create_if_missing=True)
    divisions_element = ET.Element("divisions")
    divisions_element.text = str(divisions)
    attributes_element.append(divisions_element)
    sort_attributes(attributes_element)

    # update all duration elements
    for e in part.iter("duration"):
        d = Fraction(e.text) * divisions
        assert d.denominator == 1
        e.text = str(d.numerator)
