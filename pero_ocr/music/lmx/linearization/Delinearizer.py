import xml.etree.ElementTree as ET
from typing import List, Optional, TextIO, Set
from .vocabulary import *
import io
from fractions import Fraction
from ..symbolic.PitchAlternator import PitchAlternator
from ..symbolic.get_head_attributes import get_head_attributes
from ..symbolic.sort_attributes import sort_attributes
from ..symbolic.fractional_durations_to_actual import fractional_durations_to_actual


MEASURE_ITEM_ROOTS = set([
    *NOTE_ROOT_TOKENS,
    "time",
    *KEY_TOKENS,
    *CLEF_TOKENS
])

MEASURE_ITEM_PREFIXES = set([
    *NOTE_PREFIX_TOKENS
])

MEASURE_ITEM_SUFFIXES = set([
    *NOTE_SUFFIX_TOKENS,
    *BEATS_TOKENS,
    *BEAT_TYPE_TOKENS
])

ATTRIBUTES_ROOTS = set([
    "time",
    *KEY_TOKENS,
    *CLEF_TOKENS
])


class Token:
    def __init__(self, terminal: str, position: int):
        self.terminal = terminal
        self.position = position


class Tree:
    def __init__(self, root: Token, prefixes: List[Token], suffixes: List[Token]):
        self.root = root
        self.prefixes = prefixes
        self.suffixes = suffixes


class Delinearizer:
    def __init__(
        self,
        errout: Optional[TextIO] = None,
        keep_fractional_durations=False
    ):
        self._errout = errout or io.StringIO()
        """Print errors and warnings here"""

        self.keep_fractional_durations = keep_fractional_durations

        self.part_element = ET.Element("part")

        # within-part state
        self._fractional_measure_duration: Optional[Fraction] = None
        self._open_slur_count = 0
        self._pitch_alternator = PitchAlternator()

        # within-measure state
        self._stem_orientation: Optional[str] = None # "up", "down", None
        self._staff: Optional[str] = None # "1", "2", None
        self._voice: Optional[str] = None # "1", "2", ... "8", None
        self._open_beam_count = 0
        self._open_beam_count_grace = 0

    def _error(self, token: Token, *values):
        header = f"[ERROR][Token '{token.terminal}' at position {token.position}]:"
        print(header, *values, file=self._errout)

    def process_text(self, text: str) -> ET.Element:
        # reset within-part state
        self._fractional_measure_duration = None
        self._open_slur_count = 0
        self._pitch_alternator = PitchAlternator()

        # process LMX
        tokens = self.lex(text)
        self.process_system(tokens)

        # add the <staves> element if 2 or more staves present
        self._add_staves_head_element()

        # add <divisions> and update all <duration> elements
        if not self.keep_fractional_durations:
            fractional_durations_to_actual(self.part_element)
        
        return self.part_element
    
    def _add_staves_head_element(self):
        max_clef_number = max(
            (int(clef.get("number", "1"))
            for clef in self.part_element.iterfind("measure/attributes/clef")),
            default=1
        )
        if max_clef_number > 1 and len(self.part_element) > 0:
            attributes = get_head_attributes(
                self.part_element[0],
                create_if_missing=True
            )
            staves_element = ET.Element("staves")
            staves_element.text = str(max_clef_number)
            attributes.append(staves_element)
            sort_attributes(attributes)
    
    def lex(self, text: str) -> List[Token]:
        """Lexer phase that converts the input string to token sequence"""
        terminals = text.split()
        tokens = []
        for i, terminal in enumerate(terminals):
            position = i + 1
            token = Token(terminal, position)
            if terminal not in ALL_TOKENS:
                self._error(token, "Token not present in the vocabulary.")
                continue
            tokens.append(token)
        return tokens

    def process_system(self, tokens: List[Token]):
        """Takes all tokens produced by the recognition model"""
        measure_clusters = self.cluster_measures(tokens)

        for cluster_tokens in measure_clusters:
            measure_element = self.process_measure(cluster_tokens)
            self.part_element.append(measure_element)
    
    def cluster_measures(self, tokens: List[Token]) -> List[List[Token]]:
        clusters: List[List[Token]] = []

        this_cluster: List[Token] = []
        before_first_measure = True

        def _flush():
            nonlocal before_first_measure, clusters, this_cluster
            if not before_first_measure:
                clusters.append(this_cluster)
            elif len(this_cluster) > 0:
                self._error("There are tokens before the first 'measure' token.")

        for token in tokens:
            if token.terminal == "measure":
                _flush()
                this_cluster = []
                before_first_measure = False
                continue

            this_cluster.append(token)
        
        if len(this_cluster) > 0:
            _flush()
        
        return clusters

    def process_measure(self, tokens: List[Token]) -> ET.Element:
        """Takes tokens within a measure, without the measure token itself"""
        assert all(token.terminal != "measure" for token in tokens), \
            "Measure tokens should not contain the 'measure' token"
        
        # reset within-measure state
        self._stem_orientation = None
        self._staff = None
        self._voice = None
        self._open_beam_count = 0
        self._open_beam_count_grace = 0

        measure_element = ET.Element("measure")
        attributes_element: Optional[ET.Element] = None
        last_notelike: Optional[ET.Element] = None

        trees = self.build_trees(tokens)
        for tree in trees:
            if tree.root.terminal in ATTRIBUTES_ROOTS:
                attributes_element_in = attributes_element
                attributes_element = self.process_attributes(
                    tree, attributes_element_in
                )
                if attributes_element_in is None:
                    measure_element.append(attributes_element)
            else:
                attributes_element = None
            
            if tree.root.terminal in NOTE_ROOT_TOKENS:
                notelike_element = self.process_notelike(tree)
                if (notelike_element.tag in {"forward", "backup"}
                        and last_notelike is not None
                        and last_notelike.tag == notelike_element.tag):
                    # merge forward-backup
                    last_duration = last_notelike.find("duration")
                    this_duration = notelike_element.find("duration")
                    last_duration.text = str(
                        Fraction(last_duration.text) +
                        Fraction(this_duration.text)
                    )
                else:
                    # append new forward,backup,note
                    measure_element.append(notelike_element)
                    last_notelike = notelike_element
            else:
                last_notelike = None
        
        # infer pitch alterations
        self._pitch_alternator.process_measure(measure_element)

        return measure_element
    
    def build_trees(self, tokens: List[Token]) -> List[Tree]:
        trees: List[Tree] = []
        
        old_prefixes = [] # behind an already consumed root
        old_root = None # root that has been already consumed
        suffixes = [] # for a root that has already been consumed
        prefixes = [] # for a root to be consumed

        def _first_shift(current_root: Token):
            nonlocal trees, old_prefixes, prefixes, old_root, suffixes
            assert old_root is None
            for token in suffixes:
                self._error(token, "Dangling suffix token.")
            old_prefixes = prefixes
            old_root = current_root
            suffixes = []
            prefixes = []

        def _normal_shift(current_root: Token):
            nonlocal trees, old_prefixes, prefixes, old_root, suffixes
            assert old_root is not None
            trees.append(Tree(old_root, old_prefixes, suffixes))
            old_prefixes = prefixes
            old_root = current_root
            suffixes = []
            prefixes = []
        
        def _last_shift():
            nonlocal trees, old_prefixes, prefixes, old_root, suffixes
            if old_root is not None:
                trees.append(Tree(old_root, old_prefixes, suffixes))
            for token in prefixes:
                self._error(token, "Dangling prefix token.")

        for token in tokens:
            terminal = token.terminal

            if terminal in MEASURE_ITEM_PREFIXES:
                prefixes.append(token)
            elif terminal in MEASURE_ITEM_SUFFIXES:
                suffixes.append(token)
            elif terminal in MEASURE_ITEM_ROOTS:
                if old_root is None:
                    _first_shift(token)
                else:
                    _normal_shift(token)
            else:
                self._error(token, "Unexpected measure item type.")
        
        _last_shift()

        return trees
    
    def process_attributes(self, tree: Tree, attributes_element: Optional[ET.Element]) -> ET.Element:
        if attributes_element is None:
            attributes_element = ET.Element("attributes")
        
        if tree.root.terminal in KEY_TOKENS:
            key_element = self.process_key_signature(tree)
            attributes_element.append(key_element)
        elif tree.root.terminal == "time":
            time_element = self.process_time_signature(tree)
            attributes_element.append(time_element)
        elif tree.root.terminal in CLEF_TOKENS:
            clef_element = self.process_clef(tree)
            attributes_element.append(clef_element)
        else:
            raise Exception("Unknown attributes token: " + tree.root.terminal)
        
        return attributes_element
    
    def process_key_signature(self, tree: Tree) -> ET.Element:
        assert tree.root.terminal in KEY_TOKENS

        self._list_unexpected_valencies(tree)

        fifths: str = tree.root.terminal.split(":")[-1]

        key_element = ET.Element("key")
        fifths_element = ET.Element("fifths")
        fifths_element.text = fifths
        key_element.append(fifths_element)
        return key_element
    
    def process_time_signature(self, tree: Tree) -> ET.Element:
        assert tree.root.terminal == "time"

        beats_token = self._extract_suffix(tree, BEATS_TOKENS)
        beat_type_token = self._extract_suffix(tree, BEAT_TYPE_TOKENS)
        self._list_unexpected_valencies(tree)

        # defaults
        beats = "4"
        beat_type = "4"

        if beats_token is not None:
            beats = beats_token.terminal.split(":")[-1]
        else:
            self._error(tree.root, "Missing beats token from time signature.")
        
        if beat_type_token is not None:
            beat_type = beat_type_token.terminal.split(":")[-1]
        else:
            self._error(tree.root, "Missing beat type token from time signature.")
        
        time_element = ET.Element("time")
        beats_element = ET.Element("beats")
        beats_element.text = beats
        beat_type_element = ET.Element("beat-type")
        beat_type_element.text = beat_type
        time_element.append(beats_element)
        time_element.append(beat_type_element)

        # compute measure duration in quarter note multiples
        # (in fractional time)
        beat_type_fraction = Fraction(1, int(beat_type))
        quarters_per_beat_type = beat_type_fraction / Fraction(1, 4)
        quarters_per_measure = quarters_per_beat_type * int(beats)
        self._fractional_measure_duration = quarters_per_measure

        return time_element

    def process_clef(self, tree: Tree) -> ET.Element:
        assert tree.root.terminal in CLEF_TOKENS

        staff_token = self._extract_suffix(tree, STAFF_TOKENS)
        self._list_unexpected_valencies(tree)
        
        sign, line = tree.root.terminal.split(":")[-1]
        
        clef_element = ET.Element("clef")
        sign_element = ET.Element("sign")
        sign_element.text = sign
        line_element = ET.Element("line")
        line_element.text = line
        clef_element.append(sign_element)
        clef_element.append(line_element)

        if staff_token is not None:
            staff_number = staff_token.terminal.split(":")[-1]
            clef_element.attrib["number"] = staff_number

        return clef_element

    def process_notelike(self, tree: Tree) -> ET.Element:
        assert tree.root.terminal in NOTE_ROOT_TOKENS

        backup_token = self._extract_prefix(tree, {"backup"})
        forward_token = self._extract_prefix(tree, {"forward"})

        if backup_token is not None:
            backup_element = ET.Element("backup")
            self.process_forward_backup(backup_element, tree)
            return backup_element
        
        if forward_token is not None:
            forward_element = ET.Element("forward")
            self.process_forward_backup(forward_element, tree)
            return forward_element

        return self.process_note(tree)
    
    def process_forward_backup(self, element: ET.Element, tree: Tree):
        assert tree.root.terminal in NOTE_ROOT_TOKENS

        # backup element resets beams,
        # so that if there were ever to be beams crossing measures,
        # we break them and the next voice sees a blank slate
        self._open_beam_count = 0
        self._open_beam_count_grace = 0

        note_type = tree.root.terminal
        if note_type == "rest:measure":
            self._error(tree.root, "Rest measure cannot be the root of a forward/backup element.")
            return # no duration element will be produced

        time_modification_token = self._extract_suffix(tree, TIME_MODIFICATION_TOKENS)
        self._list_unexpected_valencies(tree)

        if time_modification_token is not None:
            duration: str = self._get_fractional_duration(note_type, time_modification_token.terminal)
        else:
            duration: str = self._get_fractional_duration(note_type, None)
        
        duration_element = ET.Element("duration")
        duration_element.text = duration
        element.append(duration_element)
    
    def process_note(self, tree: Tree) -> ET.Element:
        assert tree.root.terminal in NOTE_ROOT_TOKENS

        # === parse the note tokens ====

        no_print_token = self._extract_prefix(tree, {"print-object:no"})
        grace_token = self._extract_prefix(tree, {"grace"})
        grace_slash_token = self._extract_prefix(tree, {"grace:slash"})
        chord_token = self._extract_prefix(tree, {"chord"})
        rest_token = self._extract_prefix(tree, {"rest"})
        pitch_token = self._extract_prefix(tree, PITCH_TOKENS)
        voice_token = self._extract_prefix(tree, VOICE_TOKENS)

        time_modification_token = self._extract_suffix(tree, TIME_MODIFICATION_TOKENS)
        dot_tokens = self._extract_suffixes(tree, {"dot"})
        accidental_token = self._extract_suffix(tree, ACCIDENTAL_TOKENS)
        stem_token = self._extract_suffix(tree, STEM_TOKENS)
        staff_token = self._extract_suffix(tree, STAFF_TOKENS)
        beam_tokens = self._extract_suffixes(tree, BEAM_TOKENS)
        tied_start_token = self._extract_suffix(tree, {"tied:start"})
        tied_stop_token = self._extract_suffix(tree, {"tied:stop"})
        tuplet_start_token = self._extract_suffix(tree, {"tuplet:start"})
        tuplet_stop_token = self._extract_suffix(tree, {"tuplet:stop"})
        slur_tokens = self._extract_suffixes(tree, {"slur:start", "slur:stop"})
        fermata_token = self._extract_suffix(tree, {"fermata"})
        arpeggiate_token = self._extract_suffix(tree, {"arpeggiate"})
        staccato_token = self._extract_suffix(tree, {"staccato"})
        staccatissimo_token = self._extract_suffix(tree, {"staccatissimo"})
        accent_token = self._extract_suffix(tree, {"accent"})
        strong_accent_token = self._extract_suffix(tree, {"strong-accent"})
        tenuto_token = self._extract_suffix(tree, {"tenuto"})
        tremolo_type_token = self._extract_suffix(tree, TREMOLO_TYPE_TOKENS)
        tremolo_marks_token = self._extract_suffix(tree, TREMOLO_MARKS_TOKENS)
        trill_mark_token = self._extract_suffix(tree, {"trill-mark"})

        self._list_unexpected_valencies(tree)

        # === extract higher-level information ====

        note_type = tree.root.terminal
        is_measure_rest = note_type == "rest:measure"
        if is_measure_rest:
            note_type = None
        
        is_grace = grace_token is not None
        is_rest = rest_token is not None
        is_chord = chord_token is not None

        has_articulations = any((t is not None) for t in [
            staccato_token, staccatissimo_token, accent_token, strong_accent_token, tenuto_token
        ])

        has_ornaments = any((t is not None) for t in [
            tremolo_type_token, trill_mark_token
        ])

        has_notations = any((t is not None) for t in [
            tied_start_token, tied_stop_token,
            *slur_tokens,
            tuplet_start_token, tuplet_stop_token,
            fermata_token,
            arpeggiate_token
        ]) or has_articulations or has_ornaments

        # === reconstruct the note element ====

        note_element = ET.Element("note")
        if no_print_token is not None:
            note_element.attrib["print-object"] = "no"
        
        # <grace>
        if grace_token is not None:
            grace_element = ET.Element("grace")
            if grace_slash_token is not None:
                grace_element.attrib["slash"] = "yes"
            note_element.append(grace_element)
        
        # <chord>
        if chord_token is not None:
            note_element.append(ET.Element("chord"))
        
        # <rest> or <pitch>
        if rest_token is not None:
            rest_element = ET.Element("rest")
            if is_measure_rest:
                rest_element.attrib["measure"] = "yes"
            note_element.append(rest_element)
        elif pitch_token is not None:
            step, octave = pitch_token.terminal
            pitch_element = ET.Element("pitch")
            step_element = ET.Element("step")
            step_element.text = step
            # <alter> element is added later in postprocessing
            octave_element = ET.Element("octave")
            octave_element.text = octave
            pitch_element.append(step_element)
            pitch_element.append(octave_element)
            note_element.append(pitch_element)
        
        # <duration>
        if is_grace:
            pass # no duration element in grace notes
        elif is_measure_rest and self._fractional_measure_duration is not None:
            duration_element = ET.Element("duration")
            duration_element.text = str(self._fractional_measure_duration)
            note_element.append(duration_element)
        elif note_type is not None:
            duration_element = ET.Element("duration")
            tm = (
                time_modification_token.terminal
                if time_modification_token is not None
                else None
            )
            duration_element.text = self._get_fractional_duration(
                note_type, tm, len(dot_tokens)
            )
            note_element.append(duration_element)

        # <tie> (generated just like the <tied> element)
        if tied_stop_token is not None:
            note_element.append(ET.Element("tie", {"type": "stop"}))
        if tied_start_token is not None:
            note_element.append(ET.Element("tie", {"type": "start"}))

        # <voice>
        if voice_token is not None:
            self._voice = voice_token.terminal.split(":")[1]
        if self._voice is not None:
            voice_element = ET.Element("voice")
            voice_element.text = self._voice
            note_element.append(voice_element)
        
        # <type>
        if note_type is not None:
            type_element = ET.Element("type")
            type_element.text = note_type
            note_element.append(type_element)

        # <dot>
        for _ in dot_tokens:
            note_element.append(ET.Element("dot"))

        # <accidental>
        if accidental_token is not None:
            accidental_element = ET.Element("accidental")
            accidental_element.text = accidental_token.terminal
            note_element.append(accidental_element)

        # <time-modification>
        if time_modification_token is not None:
            actual, normal = time_modification_token.terminal.split("in")
            tm_element = ET.Element("time-modification")
            actual_notes_element = ET.Element("actual-notes")
            actual_notes_element.text = actual
            normal_notes_element = ET.Element("normal-notes")
            normal_notes_element.text = normal
            tm_element.append(actual_notes_element)
            tm_element.append(normal_notes_element)
            note_element.append(tm_element)

        # <stem>
        if not is_rest and note_type not in {"whole"}:
            if stem_token is not None:
                self._stem_orientation = stem_token.terminal.split(":")[1]
            if self._stem_orientation is not None:
                stem_element = ET.Element("stem")
                stem_element.text = self._stem_orientation
                note_element.append(stem_element)

        # <staff>
        if staff_token is not None:
            self._staff = staff_token.terminal.split(":")[1]
        if self._staff is not None:
            staff_element = ET.Element("staff")
            staff_element.text = self._staff
            note_element.append(staff_element)

        # <beam>
        if not is_chord and not is_rest:
            if is_grace:
                self._open_beam_count_grace = self.emit_beams(
                    note_element, beam_tokens, self._open_beam_count_grace
                )
            else:
                self._open_beam_count = self.emit_beams(
                    note_element, beam_tokens, self._open_beam_count
                )

        # === reconstruct the notations element ====

        if has_notations:
            notations_element = ET.Element("notations")
            note_element.append(notations_element)

        # <tied>
        if tied_stop_token is not None:
            notations_element.append(ET.Element("tied", {"type": "stop"}))
        if tied_start_token is not None:
            notations_element.append(ET.Element("tied", {"type": "start"}))

        # <tuplet> (swapped with slur because MuseScore...)
        if tuplet_stop_token is not None:
            notations_element.append(ET.Element("tuplet", {"type": "stop"}))
        if tuplet_start_token is not None:
            notations_element.append(ET.Element("tuplet", {"type": "start"}))

        # <slur>
        slur_elements: List[ET.Element] = []
        for slur_token in slur_tokens:
            if slur_token.terminal == "slur:stop":
                e = ET.Element("slur", {"type": "stop"})
                slur_elements.append(e)
                if self._open_slur_count > 0:
                    e.attrib["number"] = str(self._open_slur_count)
                    self._open_slur_count -= 1
            if slur_token.terminal == "slur:start":
                self._open_slur_count += 1
                slur_elements.append(ET.Element("slur", {
                    "type": "start",
                    "number": str(self._open_slur_count)
                }))
            # NOTE: slur "continue" is not produced by MuseScore, so we also ignore it
        
        slur_elements.sort(key=lambda s: (
            1 if s.get("type") == "stop" else 2, # by start/stop primarily
            s.get("number", "1" # by number secondarily
        )))
        for se in slur_elements:
            notations_element.append(se)

        # <fermata> (before articulations and ornaments because MuseScore does it that way)
        if fermata_token is not None:
            notations_element.append(ET.Element("fermata"))

        # <articulations> (before ornaments because MuseScore does it that way)
        if has_articulations:
            articulations_element = ET.Element("articulations")
            notations_element.append(articulations_element)
        
        # <accent>
        if accent_token is not None:
            articulations_element.append(ET.Element("accent"))

        # <strong-accent>
        if strong_accent_token is not None:
            articulations_element.append(ET.Element("strong-accent"))

        # <tenuto> (before staccato, because MuseScore does it that way)
        if tenuto_token is not None:
            articulations_element.append(ET.Element("tenuto"))

        # <staccato>
        if staccato_token is not None:
            articulations_element.append(ET.Element("staccato"))

        # <staccatissimo>
        if staccatissimo_token is not None:
            articulations_element.append(ET.Element("staccatissimo"))

        # </articulation>

        # <ornaments>
        if has_ornaments:
            ornaments_element = ET.Element("ornaments")
            notations_element.append(ornaments_element)

        # <trill-mark>
        if trill_mark_token is not None:
            ornaments_element.append(ET.Element("trill-mark"))

        # <tremolo>
        if tremolo_type_token is not None:
            tremolo_element = ET.Element("tremolo", {
                "type": tremolo_type_token.terminal.split(":")[1]
            })
            if tremolo_marks_token is not None:
                tremolo_element.text = tremolo_marks_token.terminal.split(":")[1]
            else:
                tremolo_element.text = "3"
            ornaments_element.append(tremolo_element)
        
        # </ornaments>

        # <arpeggiate>
        if arpeggiate_token is not None:
            notations_element.append(ET.Element("arpeggiate"))

        # </notations>

        return note_element

    def emit_beams(self, note_element: ET.Element, beam_tokens: List[Token], open_beam_count: int) -> int:
        beam_begin_count = len(
            list(t for t in beam_tokens if t.terminal == "beam:begin")
        )
        beam_end_count = len(
            list(t for t in beam_tokens if t.terminal == "beam:end")
        )
        beam_count_delta = beam_begin_count - beam_end_count
        if beam_count_delta < -open_beam_count:
            # if we are about to close more beams then are open,
            # the we are probbably incorrect in how many beams we have open
            open_beam_count = -beam_count_delta # make it match to close to 0
        
        emitted_beam_count = None
        
        if beam_count_delta >= 0: # we are adding more beams
            for i in range(open_beam_count):
                e = ET.Element("beam", {"number": str(i + 1)})
                e.text = "continue"
                note_element.append(e)
            for _ in range(beam_count_delta):
                open_beam_count += 1
                e = ET.Element("beam", {"number": str(open_beam_count)})
                e.text = "begin"
                note_element.append(e)
            emitted_beam_count = open_beam_count
        elif beam_count_delta < 0: # we are removing beams
            emitted_beam_count = open_beam_count
            open_beam_count += beam_count_delta
            for i in range(open_beam_count):
                e = ET.Element("beam", {"number": str(i + 1)})
                e.text = "continue"
                note_element.append(e)
            for i in range(-beam_count_delta):
                e = ET.Element("beam", {"number": str(open_beam_count + i + 1)})
                e.text = "end"
                note_element.append(e)
        
        hook_tokens = list(
            t for t in beam_tokens if "hook" in t.terminal
        )
        for i, token in enumerate(hook_tokens):
            e = ET.Element("beam", {"number": str(emitted_beam_count + i + 1)})
            e.text = token.terminal.split(":")[1].replace("-", " ")
            note_element.append(e)
        
        # never return with negative beam count
        if open_beam_count < 0:
            open_beam_count = 0
        
        return open_beam_count
    
    def _get_fractional_duration(
        self,
        note_type: str,
        time_modification: Optional[str],
        dots: int = 0
    ) -> str:
        assert note_type in NOTE_TYPE_TOKENS
        if time_modification is not None:
            assert time_modification in TIME_MODIFICATION_TOKENS

        # simple conversion
        quarter_multiple = NOTE_TYPE_TO_QUARTER_MULTIPLE[note_type]

        # handle duration dots
        dot_duration = quarter_multiple / 2
        for _ in range(dots):
            quarter_multiple += dot_duration
            dot_duration /= 2

        # handle time modification
        if time_modification is not None:
            denominator, numerator = time_modification.split("in")
            quarter_multiple = quarter_multiple * Fraction(int(numerator), int(denominator))
        
        return str(quarter_multiple)
    
    def _extract_suffix(self, tree: Tree, terminals: Set[str]) -> Optional[Token]:
        target_token: Optional[Token] = None

        i = 0
        while i < len(tree.suffixes):
            token = tree.suffixes[i]
            if token.terminal in terminals:
                if target_token is None:
                    target_token = token
                else:
                    self._error(
                        token,
                        f"Additional suffix token '{token.terminal}' " + \
                            f"for the '{tree.root.terminal}' token."
                    )
                
                del tree.suffixes[i]
                i -= 1
            i += 1

        return target_token
    
    def _extract_suffixes(self, tree: Tree, terminals: Set[str]) -> List[Token]:
        target_tokens: List[Token] = []

        i = 0
        while i < len(tree.suffixes):
            token = tree.suffixes[i]
            if token.terminal in terminals:
                target_tokens.append(token)
                
                del tree.suffixes[i]
                i -= 1
            i += 1

        return target_tokens
    
    def _extract_prefix(self, tree: Tree, terminals: Set[str]) -> Optional[Token]:
        target_token: Optional[Token] = None

        i = len(tree.prefixes) - 1
        while i >= 0:
            token = tree.prefixes[i]
            if token.terminal in terminals:
                if target_token is None:
                    target_token = token
                else:
                    self._error(
                        token,
                        f"Additional prefix token for the '{tree.root.terminal}' token."
                    )
                
                del tree.prefixes[i]
            i -= 1

        return target_token
    
    def _list_unexpected_valencies(self, tree: Tree):
        for token in tree.prefixes:
            self._error(token, f"Unexpected prefix token for the '{tree.root.terminal}' token.")
        for token in tree.suffixes:
            self._error(token, f"Unexpected suffix token for the '{tree.root.terminal}' token.")
