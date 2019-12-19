# pero

## Running stuff

Scripts (as well as tests) assume that it is possible to import ``pero`` and its components.

For the current shell session, this can be achieved by setting ``PYTHONPATH`` up:
```
export PYTHONPATH=/path/to/the/repo:$PYTHONPATH
```

As a more permanent solution, a very simplistic `setup.py` is prepared:
```
python setup.py develop
```
Beware that the `setup.py` does NOT check for dependencies in the current version.

Pero can be later removed from your Python distribution by running:
```
python setup.py develop --uninstall
```

## Testing
Currently, only unittests are provided with the code. Some of the code. So simply run your preferred test runner, e.g.:
```
~/pero $ green test
```
Beware that you need to specify the test folder as the automatic test discovery can find `active_learning/adaptation/test_adapt_net.py`, which is not a test module.
