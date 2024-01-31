#!/usr/bin/python3.10
"""Script containing classes for internal representation of Semantic labels.
Symbol class is default symbol for parsing and returning music21 representation.
Other classes are internal representations of different symbols.

Primarily used for compatibility with music21 library.

Author: Vojtěch Vlach
Contact: xvlach22@vutbr.cz
"""

from __future__ import annotations
import logging
from enum import Enum
import re

import music21 as music


class SymbolType(Enum):
    CLEF = 0
    GRACENOTE = 1
    KEY_SIGNATURE = 2
    MULTI_REST = 3
    NOTE = 4
    REST = 5
    TIE = 6
    TIME_SIGNATURE = 7
    UNKNOWN = 99


class Symbol:
    """Represents one label in a label group."""

    def __init__(self, label: str):
        self.label = label
        self.type, self.repr = Symbol.label_to_symbol(label)
        self.length = self.get_length()

    def __str__(self):
        return f'\t\t\t({self.type}) {self.repr}'

    def get_length(self) -> float:
        """Returns the length of the symbol in quarter notes.

        (half note: 2 quarter notes, eighth note: 0.5 quarter notes, ...)
        If the symbol does not have musical length, returns 0.
        """
        if self.type in [SymbolType.REST, SymbolType.NOTE, SymbolType.GRACENOTE]:
            return self.repr.duration.quarterLength
        else:
            return 0

    def set_key(self, altered_pitches: AlteredPitches):
        if self.type in [SymbolType.NOTE, SymbolType.GRACENOTE]:
            self.repr = self.repr.get_real_height(altered_pitches)

    @staticmethod
    def label_to_symbol(label: str):  # -> (SymbolType, music.object):
        """Converts one label to music21 format.

        Args:
            label (str): one symbol in semantic format as string
        """
        if label.startswith("clef-"):
            label = label[len('clef-'):]
            return SymbolType.CLEF, Symbol.clef_to_symbol(label)
        elif label.startswith("gracenote-"):
            label = label[len('gracenote-'):]
            return SymbolType.GRACENOTE, Symbol.note_to_symbol(label, gracenote=True)
        elif label.startswith("keySignature-"):
            label = label[len('keySignature-'):]
            return SymbolType.KEY_SIGNATURE, Symbol.keysignature_to_symbol(label)
        elif label.startswith("multirest-"):
            label = label[len('multirest-'):]
            return SymbolType.MULTI_REST, Symbol.multirest_to_symbol(label)
        elif label.startswith("note-"):
            label = label[len('note-'):]
            return SymbolType.NOTE, Symbol.note_to_symbol(label)
        elif label.startswith("rest-"):
            label = label[len('rest-'):]
            return SymbolType.REST, Symbol.rest_to_symbol(label)
        elif label.startswith("tie"):
            label = label[len('tie'):]
            return SymbolType.TIE, Symbol.tie_to_symbol(label)
        elif label.startswith("timeSignature-"):
            label = label[len('timeSignature-'):]
            return SymbolType.TIME_SIGNATURE, Symbol.timesignature_to_symbol(label)

        logging.info(f'Unknown label: {label}, returning None.')
        return SymbolType.UNKNOWN, None

    @staticmethod
    def clef_to_symbol(clef) -> music.clef:
        """Converts one clef label to music21 format.

        Args:
            clef (str): one symbol in semantic format as string

        Returns:
            music.clef: one clef in music21 format
        """
        if len(clef) != 2:
            logging.info(f'Unknown clef label: {clef}, returning default clef.')
            return music.clef.Clef()

        return music.clef.clefFromString(clef)

    @staticmethod
    def keysignature_to_symbol(keysignature) -> music.key.Key:
        """Converts one key signature label to music21 format.

        Args:
            keysignature (str): one symbol in semantic format as string

        Returns:
            music.key.Key: one key in music21 format
        """
        if not keysignature:
            logging.info(f'Unknown key signature label: {keysignature}, returning default key.')
            return music.key.Key()

        return music.key.Key(keysignature)

    @staticmethod
    def multirest_to_symbol(multirest: str) -> MultiRest:
        """Converts one multi rest label to internal MultiRest format.

        Args:
            multirest (str): one symbol in semantic format as string

        Returns:
            music.note.Rest: one rest in music21 format
        """
        def return_default_multirest() -> MultiRest:
            logging.info(f'Unknown multi rest label: {multirest}, returning default Multirest.')
            return MultiRest()

        if not multirest:
            return_default_multirest()

        try:
            return MultiRest(int(multirest))
        except ValueError:
            return return_default_multirest()

    @staticmethod
    def note_to_symbol(note, gracenote: bool = False) -> Note:
        """Converts one note label to internal note format.

        Args:
            note (str): one symbol in semantic format as string
            gracenote (bool, optional): if True, returns grace note. Defaults to False.

        Returns:
            music.note.Note: one note in music21 format
        """
        def return_default_note() -> music.note.Note:
            logging.info(f'Unknown note label: {note}, returning default note.')
            return Note(music.duration.Duration(1), 'C4', gracenote=gracenote)

        if not note:
            return_default_note()

        note, fermata = Symbol.check_fermata(note)

        note_height, note_length = re.split('_', note, maxsplit=1)

        if not note_length or not note_height:
            return_default_note()

        return Note(label_to_length(note_length),
                    note_height, fermata=fermata, gracenote=gracenote)

    @staticmethod
    def rest_to_symbol(rest) -> music.note.Rest:
        """Converts one rest label to music21 format.

        Args:
            rest (str): one symbol in semantic format as string

        Returns:
            music.note.Rest: one rest in music21 format
        """
        if not rest:
            logging.info(f'Unknown rest label: {rest}, returning default rest.')
            return music.note.Rest()

        rest, fermata = Symbol.check_fermata(rest)

        duration = label_to_length(rest)

        rest = music.note.Rest()
        rest.duration = duration
        if fermata is not None:
            rest.expressions.append(fermata)

        return rest

    @staticmethod
    def tie_to_symbol(label):
        return Tie

    @staticmethod
    def timesignature_to_symbol(timesignature) -> music.meter.TimeSignature:
        """Converts one time signature label to music21 format.

        Args:
            timesignature (str): one symbol in semantic format as string

        Returns:
            music.meter.TimeSignature: one time signature in music21 format
        """
        if not timesignature:
            logging.info(f'Unknown time signature label: {timesignature}, returning default time signature.')
            return music.meter.TimeSignature()

        if timesignature == 'C/':
            return music.meter.TimeSignature('cut')
        else:
            return music.meter.TimeSignature(timesignature)

    @staticmethod
    def check_fermata(label: str) -> (str, music.expressions.Fermata):
        """Check if note has fermata.

        Args:
            label (str): one symbol in semantic format as string

        Returns:
            str: note without fermata
            music.expressions.Fermata: fermata
        """
        fermata = None
        if label.endswith('_fermata'):
            label = label[:-len('_fermata')]
            fermata = music.expressions.Fermata()
            fermata.type = 'upright'

        return label, fermata


class Note:
    """Represents one note in a label group.

    In the order which semantic labels are represented, the real height of note depends on key signature
    of current measure. This class is used as an internal representation of a note before knowing its real height.
    Real height is then stored directly in `self.note` as music.note.Note object.
    """

    def __init__(self, duration: music.duration.Duration, height: str,
                 fermata: music.expressions.Fermata = None, gracenote: bool = False):
        self.duration = duration
        self.height = height
        self.fermata = fermata
        self.note = None
        self.gracenote = gracenote
        self.note_ready = False

    def get_real_height(self, altered_pitches: AlteredPitches) -> music.note.Note | None:
        """Returns the real height of the note.

        Args:
            key signature of current measure
        Returns:
             Final music.note.Note object representing the real height and other info.
        """
        if self.note_ready:
            return self.note

        # pitches = [pitch.name[0] for pitch in key.alteredPitches]

        if not self.height[1:-1]:
            # Note has no accidental on its own and takes accidental of the altered pitches.
            note_str = self.height[0] + altered_pitches[self.height[0]] + self.height[-1]
            self.note = music.note.Note(note_str, duration=self.duration)
        else:
            # Note has accidental which directly tells real note height.
            note_str = self.height[0] + self.height[1:-1].replace('b', '-') + self.height[-1]
            self.note = music.note.Note(note_str, duration=self.duration)
            # Note sets new altered pitch for future notes.
            altered_pitches[self.height[0]] = note_str[1:-1]

        if self.gracenote:
            self.note = self.note.getGrace()
        self.note_ready = True
        return self.note

    def __str__(self):
        return f'note {self.height} {self.duration}'


class MultiRest:
    """Represents one multi rest in a label group."""

    def __init__(self, duration: int = 0):
        self.duration = duration


class Tie:
    """Represents one tie in a label group."""

    def __str__(self):
        return 'tie'


class AlteredPitches:
    def __init__(self, key: music.key.Key):
        self.key = key
        self.alteredPitches = {}
        for pitch in self.key.alteredPitches:
            self.alteredPitches[pitch.name[0]] = pitch.name[1]

    def __repr__(self):
        return str(self.alteredPitches)

    def __str__(self):
        return str(self.alteredPitches)

    def __getitem__(self, pitch_name: str):
        """Gets name of pitch (e.g. 'C', 'G', ...) and returns its alternation."""
        if pitch_name not in self.alteredPitches:
            return ''
        return self.alteredPitches[pitch_name]

    def __setitem__(self, pitch_name: str, direction: str):
        """Sets item.

        Args:
            pitch_name (str): name of pitch (e.g. 'C', 'G',...)
            direction (str): pitch alternation sign (#, ##, b, bb, 0, N)
        """
        if not direction:
            return
        elif direction in ['0', 'N']:
            if pitch_name in self.alteredPitches:
                del self.alteredPitches[pitch_name]
            # del self.alteredPitches[pitch_name]
            return
        else:
            self.alteredPitches[pitch_name] = direction


SYMBOL_TO_LENGTH = {
    'hundred_twenty_eighth': 0.03125,
    'hundred_twenty_eighth.': 0.046875,
    'hundred_twenty_eighth..': 0.0546875,
    'sixty_fourth': 0.0625,
    'sixty_fourth.': 0.09375,
    'sixty_fourth..': 0.109375,
    'thirty_second': 0.125,
    'thirty_second.': 0.1875,
    'thirty_second..': 0.21875,
    'sixteenth': 0.25,
    'sixteenth.': 0.375,
    'sixteenth..': 0.4375,
    'eighth': 0.5,
    'eighth.': 0.75,
    'eighth..': 0.875,
    'quarter': 1.0,
    'quarter.': 1.5,
    'quarter..': 1.75,
    'half': 2.0,
    'half.': 3.0,
    'half..': 3.5,
    'whole': 4.0,
    'whole.': 6.0,
    'whole..': 7.0,
    'breve': 8.0,
    'breve.': 10.0,
    'breve..': 11.0,
    'double_whole': 8.0,
    'double_whole.': 12.0,
    'double_whole..': 14.0,
    'quadruple_whole': 16.0,
    'quadruple_whole.': 24.0,
    'quadruple_whole..': 28.0
}

LENGTH_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_LENGTH.items()}  # reverse dictionary


def label_to_length(length: str) -> music.duration.Duration:
    """Return length of label as music21 duration.

    Args:
        length (str): only length part of one label in semantic format as string

    Returns:
        music.duration.Duration: one duration in music21 format
    """
    if length in SYMBOL_TO_LENGTH:
        return music.duration.Duration(SYMBOL_TO_LENGTH[length])
    else:
        logging.info(f'Unknown duration label: {length}, returning default duration.')
        return music.duration.Duration(1)
