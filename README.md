# pero-ocr

## Running stuff


Scripts (as well as tests) assume that it is possible to import ``pero_ocr`` and its components.

For the current shell session, this can be achieved by setting ``PYTHONPATH`` up:
```
export PYTHONPATH=/path/to/the/repo:$PYTHONPATH
```

As a more permanent solution, a very simplistic `setup.py` is prepared:
```
python setup.py develop
```
Beware that the `setup.py` does not promise to bring all the required stuff, e.g. setting CUDA up is up to you.

Pero can be later removed from your Python distribution by running:
```
python setup.py develop --uninstall
```

## Available models
General layout analysis (printed and handwritten) with european printed OCR specialized to czech newspapers can be [downloaded here](https://www.fit.vut.cz/~ihradis/pero/pero_eu_cz_print_newspapers_2020-10-09.tar.gz).

## Contributing
Working changes are expected to happen on `develop` branch, so if you plan to contribute, you better check it out right during cloning:

```
git clone -b develop git@github.com:DCGM/pero-ocr.git pero-ocr
```

### Testing
Currently, only unittests are provided with the code. Some of the code. So simply run your preferred test runner, e.g.:
```
~/pero-ocr $ green
```
