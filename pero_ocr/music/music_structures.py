#!/usr/bin/python3.10
"""Script for converting semantic sequential representation of music labels (produced by the model) to music21 stream
usable by music21 library to export to other formats.

Author: Vojtěch Vlach
Contact: xvlach22@vutbr.cz
"""

from __future__ import annotations
import re
from enum import Enum
import logging

import music21 as music
from music_symbols import Symbol, SymbolType, AlteredPitches, LENGTH_TO_SYMBOL


class Measure:
    _is_polyphonic = None
    keysignature = None
    repr = None
    new_system = False
    start_clef = None
    last_clef = None

    def __init__(self, labels: str):
        """Takes labels corresponding to a single measure."""
        self.labels = labels

        label_groups = re.split(r'\+', self.labels)
        stripped_label_groups = []
        for measure_label in label_groups:
            stripped = measure_label.strip().strip('+').strip()
            if stripped:
                stripped_label_groups.append(stripped)

        self.symbol_groups = [SymbolGroup(label_group) for label_group in stripped_label_groups]
        self._is_polyphonic = self.is_polyphonic

    def __str__(self):
        label_groups_str = '\n'.join([str(group) for group in self.symbol_groups])
        poly = 'polyphonic' if self.is_polyphonic else 'monophonic'
        return (f'MEASURE: ({poly}) \n'
                f'key signature: {self.keysignature}\n'
                f'labels: {self.labels}\n'
                f'{label_groups_str}')

    @property
    def is_polyphonic(self) -> bool:
        """Returns True if there are more than 1 notes in the same label_group with different lengths."""
        if self._is_polyphonic is not None:
            return self._is_polyphonic

        self._is_polyphonic = any(group.type == SymbolGroupType.TUPLE for group in self.symbol_groups)
        return self._is_polyphonic

    def get_key(self, previous_measure_key: music.key.Key) -> music.key.Key:
        """Returns the key of the measure.

        Args:
            previous_measure_key (music.key.Key): key of the previous measure.

        Returns:
            music.key.Key: key of the current measure.
        """
        if self.keysignature is not None:
            return self.keysignature

        for symbol_group in self.symbol_groups:
            key = symbol_group.get_key()
            if key is not None:
                self.set_key(key)
                break
        else:
            self.set_key(previous_measure_key)
        return self.keysignature

    def get_start_clef(self) -> music.clef.Clef | None:
        if self.start_clef is not None:
            return self.start_clef
        else:
            return self.symbol_groups[0].get_clef()

    def get_last_clef(self, previous_measure_last_clef: music.clef.Clef = music.clef.TrebleClef
                      ) -> music.clef.Clef | None:
        if self.last_clef is not None:
            return self.last_clef

        self.last_clef = previous_measure_last_clef

        for group in self.symbol_groups:
            new_clef = group.get_clef()
            if new_clef is not None:
                self.last_clef = new_clef

        return self.last_clef

    def delete_clef_symbol(self, position: int = 0) -> None:
        self.symbol_groups.pop(position)

    def set_key(self, key: music.key.Key):
        """Sets the key of the measure. Send key to all symbols groups to represent notes in real height.

        Args:
            key (music.key.Key): key of the current measure.
        """
        self.keysignature = key

        altered_pitches = AlteredPitches(key)
        for symbol_group in self.symbol_groups:
            symbol_group.set_key(altered_pitches)

    def encode_to_music21(self) -> music.stream.Measure:
        """Encodes the measure to music21 format.

        Returns:
            music.stream.Measure: music21 representation of the measure.
        """
        if self.repr is not None:
            return self.repr

        self.repr = music.stream.Measure()
        if not self._is_polyphonic:
            for symbol_group in self.symbol_groups:
                self.repr.append(symbol_group.encode_to_music21_monophonic())
        else:
            self.repr = self.encode_to_music21_polyphonic()

        if self.new_system:
            self.repr.insert(0, music.layout.SystemLayout(isNew=True))

        logging.debug('Current measure:')
        logging.debug(str(self))

        logging.debug('Current measure ENCODED:')
        return self.repr

    def encode_to_music21_polyphonic(self) -> music.stream.Measure:
        """Encodes POLYPHONIC MEASURE to music21 format.

        Returns:
            music.stream.Measure: music21 representation of the measure.
        """
        voice_count = max([symbol_group.get_voice_count() for symbol_group in self.symbol_groups])
        voices = [Voice() for _ in range(voice_count)]
        logging.debug('-------------------------------- NEW MEASURE --------------------------------')
        logging.debug(f'voice_count: {voice_count}')

        zero_length_symbol_groups = Measure.find_zero_length_symbol_groups(self.symbol_groups)
        remaining_symbol_groups = self.symbol_groups[len(zero_length_symbol_groups):]

        mono_start_symbol_groups = Measure.get_mono_start_symbol_groups(remaining_symbol_groups)
        voices = Measure.create_mono_start(voices, mono_start_symbol_groups)
        remaining_symbol_groups = remaining_symbol_groups[len(mono_start_symbol_groups):]

        # Groups to voices
        for symbol_group in remaining_symbol_groups:
            logging.debug('------------ NEW symbol_group ------------------------')
            groups_to_add = symbol_group.get_groups_to_add()
            shortest_voice_ids = Measure.pad_voices_to_n_shortest(voices, len(groups_to_add))

            logging.debug(
                f'Zipping {len(groups_to_add)} symbol groups to shortest voices ({len(shortest_voice_ids)}): {shortest_voice_ids}')
            for voice_id, group in zip(shortest_voice_ids, groups_to_add):
                logging.debug(f'Voice ({voice_id}) adding: {group}')
                voices[voice_id].add_symbol_group(group)

            for voice_id, voice in enumerate(voices):
                logging.debug(f'voice ({voice_id}) len: {voice.length}')

        zero_length_encoded = [group.encode_to_music21_monophonic() for group in zero_length_symbol_groups]
        voices_repr = [voice.encode_to_music21_monophonic() for voice in voices]
        return music.stream.Measure(zero_length_encoded + voices_repr)

    @staticmethod
    def find_shortest_voices(voices: list, ignore: list = None) -> list[int]:
        """Go through all voices and find the one with the current shortest duration.

        Args:
            voices (list): list of voices.
            ignore (list): indexes of voices to ignore.

        Returns:
            list: indexes of voices with the current shortest duration.
        """
        if ignore is None:
            ignore = []

        shortest_duration = 1_000_000
        shortest_voice_ids = [0]
        for voice_id, voice in enumerate(voices):
            if voice_id in ignore:
                continue
            if voice.length < shortest_duration:
                shortest_duration = voice.length
                shortest_voice_ids = [voice_id]
            elif voice.length == shortest_duration:
                shortest_voice_ids.append(voice_id)

        return shortest_voice_ids

    @staticmethod
    def find_zero_length_symbol_groups(symbol_groups: list[SymbolGroup]) -> list[SymbolGroup]:
        """Returns a list of zero-length symbol groups AT THE BEGGING OF THE MEASURE."""
        zero_length_symbol_groups = []
        for symbol_group in symbol_groups:
            if symbol_group.type == SymbolGroupType.TUPLE or symbol_group.length > 0:
                break
            zero_length_symbol_groups.append(symbol_group)
        return zero_length_symbol_groups

    @staticmethod
    def pad_voices_to_n_shortest(voices: list[Voice], n: int = 1) -> list[int]:
        """Pads voices (starting from the shortest) so there is n shortest voices with same length.

        Args:
            voices (list): list of voices.
            n (int): number of desired shortest voices.

        Returns:
            list: list of voice IDS with the current shortest duration.
        """
        shortest_voice_ids = Measure.find_shortest_voices(voices)

        while n > len(shortest_voice_ids):
            logging.debug(f'Found {len(shortest_voice_ids)} shortest voices, desired voices: {n}.')
            second_shortest_voice_ids = Measure.find_shortest_voices(voices, ignore=shortest_voice_ids)
            second_shortest_len = voices[second_shortest_voice_ids[0]].length
            for voice_id in shortest_voice_ids:
                desired_padding_length = second_shortest_len - voices[voice_id].length
                voices[voice_id].add_padding(desired_padding_length)

            shortest_voice_ids = Measure.find_shortest_voices(voices)

        return shortest_voice_ids

    @staticmethod
    def get_mono_start_symbol_groups(symbol_groups: list[SymbolGroup]) -> list[SymbolGroup]:
        """Get a list of monophonic symbol groups AT THE BEGINNING OF THE MEASURE.

        Returns:
            list: list of monophonic symbol groups AT THE BEGINNING OF THE MEASURE
        """
        mono_start_symbol_groups = []
        for symbol_group in symbol_groups:
            if symbol_group.type == SymbolGroupType.TUPLE:
                break
            mono_start_symbol_groups.append(symbol_group)
        return mono_start_symbol_groups

    @staticmethod
    def create_mono_start(voices: list[Voice], mono_start_symbol_groups: list[SymbolGroup]) -> list[Voice]:
        """Create monophonic start of measure in the first voice and add padding to the others.

        Args:
            voices (list[Voices]): list of voices
            mono_start_symbol_groups: list of monophonic symbol groups AT THE BEGINNING OF MEASURE.

        Returns:
            list[Voice]: list of voices
        """
        padding_length = 0
        for symbol_group in mono_start_symbol_groups:
            voices[0].add_symbol_group(symbol_group)
            padding_length += symbol_group.length

        for voice in voices[1:]:
            voice.add_padding(padding_length)

        return voices


class SymbolGroupType(Enum):
    SYMBOL = 0
    CHORD = 1
    TUPLE = 2
    EMPTY = 3
    UNKNOWN = 99


class SymbolGroup:
    """Represents one label group in a measure. Consisting of 1 to n labels/symbols."""
    tuple_data: list = None  # Tuple data consists of a list of symbol groups where symbols have same lengths.
    length: float = None   # Length of the symbol group in quarter notes.

    def __init__(self, labels: str):
        self.labels = labels
        self.type = SymbolGroupType.UNKNOWN

        label_group_parsed = re.split(r'\s', self.labels.strip())
        self.symbols = [Symbol(label_group) for label_group in label_group_parsed if label_group]

        self.type = self.get_type()
        if self.type == SymbolGroupType.TUPLE:
            self.create_tuple_data()

    def __str__(self):
        if not self.type == SymbolGroupType.TUPLE:
            symbols_str = '\n'.join([str(symbol) for symbol in self.symbols])
            return (f'\t({self.type}) {self.labels} (len: {self.length}) =>\n'
                    f'{symbols_str}')
        out = []
        for group in self.tuple_data:
            out.append(str(group))
        out = '\n'.join(out)

        return (f'\tTUPLE BEGIN:\n'
                f'{out}\n'
                f'\tTUPLE END')

    def get_type(self):
        if len(self.symbols) == 0:
            logging.warning(f'No symbols found in label group: {self.labels}')
            return SymbolGroupType.UNKNOWN
        elif len(self.symbols) == 1:
            self.length = self.symbols[0].get_length()
            return SymbolGroupType.SYMBOL
        else:
            same_length_notes = all((symbol.get_length() == self.symbols[0].get_length() and
                                     symbol.type in [SymbolType.NOTE, SymbolType.GRACENOTE])
                                    for symbol in self.symbols)
            if same_length_notes:
                self.length = self.symbols[0].get_length()
                return SymbolGroupType.CHORD
            else:
                return SymbolGroupType.TUPLE

    def get_key(self) -> music.key.Key | None:
        """Go through all labels and find key signature or return None.

        Returns:
            music.key.Key: key signature of the label group or None.
        """
        if not self.type == SymbolGroupType.SYMBOL:
            return None

        for symbol in self.symbols:
            if symbol.type == SymbolType.KEY_SIGNATURE:
                return symbol.repr

        return None

    def set_key(self, altered_pitches: AlteredPitches):
        if self.type == SymbolGroupType.TUPLE:
            for group in self.tuple_data:
                group.set_key(altered_pitches)
        else:
            for symbol in self.symbols:
                symbol.set_key(altered_pitches)

    def encode_to_music21_monophonic(self):
        """Encodes the label group to music21 format.

        Returns:
            music.object: music21 representation of the label group.
        """
        if self.type == SymbolGroupType.SYMBOL:
            return self.symbols[0].repr
        elif self.type == SymbolGroupType.CHORD:
            notes = [symbol.repr for symbol in self.symbols]
            logging.debug(f'notes: {notes}')
            return music.chord.Chord(notes)
            # return music.stream.Stream(music.chord.Chord(notes))
        elif self.type == SymbolGroupType.EMPTY:
            return music.stream.Stream()
        elif self.type == SymbolGroupType.TUPLE:
            logging.info(f'Tuple label group not supported yet, returning empty stream.')
            return music.stream.Stream()
        else:
            return music.stream.Stream()

    def create_tuple_data(self):
        """Create tuple data for the label group.

        Tuple data consists of a list of symbol groups where symbols have same lengths.
        """
        # logging.debug(f'Creating tuple data for label group: {self.labels}')
        list_of_groups = [[self.symbols[0]]]
        for symbol in self.symbols[1:]:
            if symbol.type == SymbolType.REST:
                list_of_groups.append([symbol])
                continue
            symbol_length = symbol.get_length()
            for group in list_of_groups:
                # if symbol_length == group[0].get_length() and symbol.type in [SymbolType.NOTE, SymbolType.GRACENOTE]:
                if group[0].type in [SymbolType.NOTE, SymbolType.GRACENOTE] and symbol_length == group[0].get_length():
                    group.append(symbol)
                    break
            else:
                list_of_groups.append([symbol])

        # logging.debug(list_of_groups)

        self.tuple_data = []
        for group in list_of_groups:
            labels = [symbol.label for symbol in group]
            labels = ' '.join(labels)
            self.tuple_data.append(SymbolGroup(labels))

    def get_voice_count(self):
        """Returns the number of voices in the label group (count of groups in tuple group)

        Returns:
            int: number of voices in the label group.
        """
        if not self.type == SymbolGroupType.TUPLE:
            return 1
        return len(self.tuple_data)

    def get_groups_to_add(self):
        """Returns list of symbol groups. Either self in list of symbol groups in tuple data."""
        if self.type == SymbolGroupType.TUPLE:
            groups_to_add = self.tuple_data.copy()
            groups_to_add.reverse()
            # return groups_to_add.reverse()
        else:
            groups_to_add = [self]
            # return [self]

        logging.debug(f'groups_to_add:')
        for group in groups_to_add:
            logging.debug(f'{group}')

        return groups_to_add

    def get_clef(self) -> music.clef.Clef | None:
        if self.type == SymbolGroupType.SYMBOL and self.symbols[0].type == SymbolType.CLEF:
            return self.symbols[0].repr
        return None


class Voice:
    """Internal representation of voice (list of symbol groups symbolizing one musical line)."""
    length: float = 0.0   # Accumulated length of symbol groups (in quarter notes).
    symbol_groups: list = []
    repr = None

    def __init__(self):
        self.length = 0.0
        self.symbol_groups = []
        self.repr = None

    def __str__(self):
        out = []
        for group in self.symbol_groups:
            out.append(str(group))
        out = '\n'.join(out)

        return (f'\tVOICE BEGIN:\n'
                f'{out}\n'
                f'\tVOICE END')

    def add_symbol_group(self, symbol_group: SymbolGroup) -> None:
        if symbol_group.type == SymbolGroupType.TUPLE:
            logging.warning(f'Can NOT add symbol group of type TUPLE to a voice.')
            return
        self.symbol_groups.append(symbol_group)
        self.length += symbol_group.length
        self.repr = None

    def encode_to_music21_monophonic(self) -> music.stream.Voice:
        """Encodes the voice to music21 format.

        Returns:
            music.Voice: music21 representation of the voice.
        """
        if self.repr is not None:
            return self.repr

        if len(self.symbol_groups) == 0:
            return music.stream.Voice()

        self.repr = music.stream.Voice()
        for group in self.symbol_groups:
            self.repr.append(group.encode_to_music21_monophonic())
        return self.repr

    def add_padding(self, padding_length: float) -> None:
        """Add padding symbols (rests) to the voice until it reaches the desired length.

        Args:
            padding_length (float): desired length of the padding in quarter notes.
        """
        lengths = list(LENGTH_TO_SYMBOL.values())
        min_length = min(LENGTH_TO_SYMBOL.keys())

        while padding_length > 0:
            if padding_length in LENGTH_TO_SYMBOL:
                length_label = LENGTH_TO_SYMBOL[padding_length]
                logging.debug(f'Completing padding with padding length {padding_length} to the voice.')
                self.add_symbol_group(SymbolGroup(f'rest-{length_label}'))
                padding_length -= padding_length
            elif padding_length < min_length:
                logging.error(f'Padding length {padding_length} is smaller than the minimum length {min}, breaking.')
                break
            else:
                # Step is the biggest number lower than desired padding length.
                step = lengths[lengths < padding_length].max()
                logging.debug(f'Adding padding STEP {step} to the voice.')

                length_label = LENGTH_TO_SYMBOL[step]
                self.add_symbol_group(SymbolGroup(f'rest-{length_label}'))
                padding_length -= step
