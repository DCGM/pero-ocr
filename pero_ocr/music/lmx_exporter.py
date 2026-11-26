"""Export of piano systems transcribed in LMX (Linearized MusicXML) to MusicXML."""

import io
import logging
import xml.etree.ElementTree as ET
from typing import List

from pero_ocr.music.lmx.linearization.Delinearizer import Delinearizer

logger = logging.getLogger(__name__)


def lmx_systems_to_musicxml(systems: List[str]) -> str:
    """Return a MusicXML piano score with the measures of all systems one after another.

    Each system is converted on its own, so a system that cannot be converted is skipped
    instead of breaking the whole score.
    """
    part = ET.Element('part', id='P1')
    for index, lmx in enumerate(systems, start=1):
        errors = io.StringIO()
        try:
            system = Delinearizer(errout=errors).process_text(lmx)
        except Exception as e:
            logger.warning(f'Skipping system {index}, its LMX cannot be converted to MusicXML: {e}')
            continue
        if errors.getvalue():
            logger.warning(f'System {index} has invalid LMX: {errors.getvalue().splitlines()[0]}')
        part.extend(system)

    for number, measure in enumerate(part, start=1):
        measure.set('number', str(number))

    score = ET.Element('score-partwise', version='3.1')
    part_list = ET.SubElement(score, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    ET.SubElement(score_part, 'part-name').text = 'Piano'
    instrument = ET.SubElement(score_part, 'score-instrument', id='P1-I1')
    ET.SubElement(instrument, 'instrument-name').text = 'Piano'
    midi_instrument = ET.SubElement(score_part, 'midi-instrument', id='P1-I1')
    ET.SubElement(midi_instrument, 'midi-channel').text = '1'
    ET.SubElement(midi_instrument, 'midi-program').text = '1'
    score.append(part)

    return ET.tostring(score, encoding='UTF-8', xml_declaration=True).decode('utf-8')
