
from typing import Union
import re
import logging
import json
import os
from typing import Optional

logger = logging.getLogger(__name__)


class OutputTranslator:
    """Class for translating output from shorter form to longer form using simple dictionary.

    Used for example in Optical Music Recognition to translate shorter SSemantic encoding to Semantic encoding."""
    def __init__(self, dictionary: dict = None, filename: str = None, atomic: bool = False):
        self.dictionary = self.load_dictionary(dictionary, filename)
        self.dictionary_reversed = {v: k for k, v in self.dictionary.items()}
        self.n_existing_labels = set()

        # ensures atomicity on line level (if one symbol is not found, return None and let caller handle it)
        self.atomic = atomic

    def __call__(self, inputs: Union[str, list], reverse: bool = False) -> Union[str, list, None]:
        if isinstance(inputs, list):
            if len(inputs[0]) > 1:  # list of strings (lines)
                return self.translate_lines(inputs, reverse)
            else:  # list of chars (one line total)
                return self.translate_line(''.join(inputs), reverse)
        elif isinstance(inputs, str):  # one line
            return self.translate_line(inputs, reverse)
        else:
            raise ValueError(f'OutputTranslator: Unsupported input type: {type(inputs)}')

    def translate_lines(self, lines: list, reverse: bool = False) -> list:
        return [self.translate_line(line, reverse) for line in lines]

    def translate_line(self, line, reverse: bool = False):
        line_stripped = line.replace('"', ' ').strip()
        symbols = re.split(r'\s+', line_stripped)

        converted_symbols = []
        for symbol in symbols:
            translation = self.translate_symbol(symbol, reverse)
            if translation is None:
                if self.atomic:
                    return None  # return None and let caller handle it (e.g. by storing the original line or breaking)
                converted_symbols.append(symbol)
            else:
                converted_symbols.append(translation)

        return ' '.join(converted_symbols)

    def translate_symbol(self, symbol: str, reverse: bool = False) -> Optional[str]:
        dictionary = self.dictionary_reversed if reverse else self.dictionary

        translation = dictionary.get(symbol, None)
        if translation is not None:
            return translation

        if symbol not in self.n_existing_labels:
            logger.debug(f'Not existing label: ({symbol})')
        self.n_existing_labels.add(symbol)

        return None

    @staticmethod
    def load_dictionary(dictionary: dict = None, filename: str = None) -> dict:
        if dictionary is not None:
            return dictionary
        elif filename is not None:
            return OutputTranslator.read_json(filename)
        else:
            raise ValueError('OutputTranslator: Either dictionary or filename must be provided.')

    @staticmethod
    def read_json(filename) -> dict:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'Translator file ({filename}) not found. Cannot translate output.')

        with open(filename) as f:
            data = json.load(f)
        return data
