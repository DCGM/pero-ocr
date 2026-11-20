"""Config sections for engine tests: every engine (pipeline stage) is built from one config.ini section."""
import configparser
import json


def section(name='S', **options):
    """A configparser section with the given options, e.g. section(METHOD='LAYOUT_CNN', DOWNSAMPLE=4).
    Lists and tuples become JSON, as pero-ocr reads list options (CATEGORIES, IMAGE_SIZE) with json.loads."""
    config = configparser.ConfigParser()
    config.read_dict({name: {key: json.dumps(list(value)) if isinstance(value, (list, tuple)) else str(value)
                             for key, value in options.items()}})
    return config[name]


def sections(**named_sections):
    """A whole ConfigParser from {section name: {option: value}}, for factories that read several sections."""
    config = configparser.ConfigParser()
    config.read_dict({name: {key: json.dumps(list(value)) if isinstance(value, (list, tuple)) else str(value)
                             for key, value in options.items()} for name, options in named_sections.items()})
    return config
