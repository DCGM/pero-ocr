
from typing import Union
import re
import logging
import json
import os

logger = logging.getLogger(__name__)


class OutputTranslator:
    """Class for translating output from shorter form to longer form using simple dictionary.

    Used for example in Optical Music Recognition to translate shorter SSemantic encoding to Semantic encoding."""
    def __init__(self, dictionary: dict = None, filename: str = None):
        self.dictionary = self.load_dictionary(dictionary, filename)
        self.dictionary_reversed = {v: k for k, v in self.dictionary.items()}
        self.n_existing_labels = set()

    def __call__(self, inputs: Union[str, list], reverse: bool = False) -> Union[str, list]:
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
        line = line.strip('"').strip()
        symbols = re.split(r'\s+', line)
        converted_symbols = [self.translate_symbol(symbol, reverse) for symbol in symbols]

        return ' '.join(converted_symbols)

    def translate_symbol(self, symbol: str, reverse: bool = False):
        dictionary = self.dictionary_reversed if reverse else self.dictionary

        try:
            return dictionary[symbol]
        except KeyError:
            if symbol not in self.n_existing_labels:
                self.n_existing_labels.add(symbol)
                logger.info(f'Not existing label: ({symbol})')
            return ''

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
