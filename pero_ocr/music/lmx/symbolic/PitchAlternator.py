import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Tuple
from fractions import Fraction
import copy


class PitchAlternator:
    """Updates the <alter> pitch tag to match the key signature and accidentals."""

    def __init__(self):
        # part-scoped state
        self._key_signature: int = 0
        self._measure_number = 0

        # measure-scoped state
        self._onset: Fraction = Fraction(0)
        self._measure_accidentals: Dict[str, List[Tuple[Fraction, str]]] = {}
        self._tie_starts: Dict[str, ET.Element] = {}
        self._previous_tie_starts: Dict[str, ET.Element] = {}

    def process_measure(self, measure: ET.Element):
        assert measure.tag == "measure"

        self._measure_number += 1

        self._onset = Fraction(0)
        self._measure_accidentals = {}
        self.collect_measure_accidentals(measure)
        # print(self._measure_number, self._measure_accidentals)
        self._onset = Fraction(0)

        self._previous_tie_starts = self._tie_starts
        self._tie_starts = {}

        for element in measure:
            if element.tag == "note":
                self.process_note(element)
                self.process_ties(element)
            elif element.tag == "attributes":
                self.process_attributes(element)
            
            self.update_onset(element)
    
    def process_ties(self, note: ET.Element):
        pitch = get_visual_pitch(note)
        if pitch is None:
            return

        has_start = False  # can be both!
        has_stop = False
        for tie in note.iterfind("tie"):
            if tie.get("type") == "start":
                has_start = True
            if tie.get("type") == "stop":
                has_stop = True

        if has_stop:
            if pitch in self._tie_starts:
                # within-measure match, fire event
                start_note = self._tie_starts[pitch]
                del self._tie_starts[pitch]
                self.handle_tie(start_note, note)
            elif pitch in self._previous_tie_starts:
                # cross-measure match, fire event
                start_note = self._previous_tie_starts[pitch]
                del self._previous_tie_starts[pitch]
                self.handle_tie(start_note, note)
        
        if has_start:
            self._tie_starts[pitch] = note
    
    def handle_tie(self, start_note: ET.Element, stop_note: ET.Element):
        # copy the pitch element children over, so that any <alter> element is set as well
        start_pitch = start_note.find("pitch")
        stop_pitch = stop_note.find("pitch")
        stop_pitch[:] = copy.deepcopy(list(start_pitch[:]))

    def collect_measure_accidentals(self, measure: ET.Element):
        for element in measure:
            if element.tag != "note":
                self.update_onset(element)
                continue

            pitch = get_visual_pitch(element)
            if pitch is None:
                self.update_onset(element)
                continue
            
            accidental_element = element.find("accidental")
            if accidental_element is None:
                self.update_onset(element)
                continue

            note_onset = self.get_note_onset(element)
            
            # record the accidental
            if pitch not in self._measure_accidentals:
                self._measure_accidentals[pitch] = []
            self._measure_accidentals[pitch].append(
                (note_onset, accidental_element.text)
            )

            self.update_onset(element)
        
        # sort accidentals by onset
        for pitch in self._measure_accidentals.keys():
            self._measure_accidentals[pitch].sort()

    def update_onset(self, element: ET.Element):
        if element.tag not in {"note", "forward", "backup"}:
            return

        duration_element = element.find("duration")
        if duration_element is None:
            return
        
        if element.tag == "note":
            if element.find("chord") is not None:
                return # chord notes have duration but do not move onset

        duration = Fraction(duration_element.text)
        if element.tag == "backup":
            duration = -duration
        
        self._onset += duration
    
    def get_note_onset(self, note: ET.Element):
        note_duration = Fraction(0)
        duration_element = note.find("duration")
        if duration_element is not None:
            note_duration = Fraction(duration_element.text)

        note_onset = self._onset
        if note.find("chord") is not None:
            note_onset -= note_duration
        
        return note_onset
    
    def process_attributes(self, attributes: ET.Element):
        fifths_element = attributes.find("key/fifths")
        if fifths_element is not None:
            self._key_signature = int(fifths_element.text)

    def process_note(self, note: ET.Element):
        pitch = get_visual_pitch(note)
        if pitch is None:
            return

        pitch_element = note.find("pitch")
        alter_element = pitch_element.find("alter")
        if alter_element is None:
            alter_element = ET.Element("alter")
            pitch_element.insert(1, alter_element)
        
        # compute the alter
        note_onset = self.get_note_onset(note)
        alter = self.get_note_alter(pitch, note_onset)
        alter_element.text = str(alter)
        
        if alter == 0:
            pitch_element.remove(alter_element)
    
    def get_note_alter(self, pitch: Tuple[str, str, str], note_onset: Fraction) -> int:
        # accidental carried from this! and previous notes in the measure
        carried_accidental = self.get_carried_accidental(pitch, note_onset)

        # accidental carried from key signature
        key_signature_accidental = self.get_key_signature_accidental(pitch[1])

        # combine all the accidentals
        if carried_accidental is not None:
            actual_accidental = carried_accidental
        elif key_signature_accidental is not None:
            actual_accidental = key_signature_accidental
        else:
            actual_accidental = None
        
        return accidental_to_alteration(actual_accidental)
    
    def get_carried_accidental(self, pitch: Tuple[str, str, str], note_onset: Fraction) -> Optional[str]:
        accidentals = [
            accidental for onset, accidental
            in self._measure_accidentals.get(pitch, [])
            if onset <= note_onset
        ]
        if len(accidentals) == 0:
            return None
        return accidentals[-1]
    
    def get_key_signature_accidental(self, step: str) -> Optional[str]:
        _SHARPS = ["F", "C", "G", "D", "A", "E", "B"] # fis, cis, gis, ...
        _FLATS = ["B", "E", "A", "D", "G", "C", "F"] # b, es, as, des, ...
        
        if self._key_signature > 0:
            if step in _SHARPS[0:self._key_signature]:
                return "sharp"
        elif self._key_signature < 0:
            if step in _FLATS[0:-self._key_signature]:
                return "flat"

        return None


def get_visual_pitch(note: ET.Element) -> Optional[Tuple[str, str, str]]:
    """Staff + step + octave"""
    pitch_element = note.find("pitch")
    if pitch_element is None:
        return None

    step_element = pitch_element.find("step")
    assert step_element is not None
    octave_element = pitch_element.find("octave")
    assert octave_element is not None

    staff_element = note.find("staff")
    staff = "1"
    if staff_element is not None:
        staff = staff_element.text

    return (staff, step_element.text, octave_element.text)


def accidental_to_alteration(accidental: Optional[str]) -> int:
    if accidental is None:
        return 0

    if accidental == "sharp":
        return 1
    elif accidental == "flat":
        return -1
    elif accidental == "natural":
        return 0
    elif accidental == "double-sharp":
        return 2
    elif accidental == "flat-flat":
        return -2
    elif accidental == "natural-sharp":
        return 1
    elif accidental == "natural-flat":
        return -1

    raise Exception("Unknown accidental: " + str(accidental))
