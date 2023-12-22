
from typing import Union
import re
import logging
import json
import os

logger = logging.getLogger(__name__)


class MusicTranslator:
    """MusicTranslator class for translating shorter SSemantic encoding to Semantic encoding using dictionary."""
    def __init__(self, dictionary: dict = None, filename: str = None):
        self.dictionary = self.load_dictionary(dictionary, filename)
        self.dictionary_reversed = {v: k for k, v in self.dictionary.items()}
        self.n_existing_labels = set()

    def __call__(self, inputs: Union[str, list], to_longer: bool = True) -> Union[str, list]:
        if isinstance(inputs, list):
            if len(inputs[0]) > 1:  # list of strings (lines)
                return self.translate_lines(inputs, to_longer)
            else:  # list of chars (one line total)
                return self.translate_line(''.join(inputs), to_longer)
        elif isinstance(inputs, str):  # one line
            return self.translate_line(inputs, to_longer)
        else:
            raise ValueError(f'MusicTranslator: Unsupported input type: {type(inputs)}')

    def translate_lines(self, lines: list, to_longer: bool = True) -> list:
        return [self.translate_line(line, to_longer) for line in lines]

    def translate_line(self, line, to_longer: bool = True):
        line = line.strip('"').strip()
        symbols = re.split(r'\s+', line)
        converted_symbols = [self.translate_symbol(symbol, to_longer) for symbol in symbols]

        return ' '.join(converted_symbols)

    def translate_symbol(self, symbol: str, to_longer: bool = True):
        dictionary = self.dictionary_reversed if to_longer else self.dictionary

        try:
            return dictionary[symbol]
        except KeyError:
            if symbol not in self.n_existing_labels:
                self.n_existing_labels.add(symbol)
                logger.info(f'Not existing label: ({symbol})')
            return ''

    def load_dictionary(self, dictionary: dict = None, filename: str = None) -> dict:
        if dictionary is not None:
            return dictionary
        elif filename is not None:
            return self.read_json(filename)
        else:
            raise ValueError('MusicTranslator: Either dictionary or filename must be provided.')

    @staticmethod
    def read_json(filename) -> dict:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'Translator file ({filename}) not found. Cannot export music.')

        with open(filename) as f:
            data = json.load(f)
        return data
