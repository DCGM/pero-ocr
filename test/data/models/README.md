# Frozen tiny models

These are torchscript stand-ins from `test/tiny_models.py`, frozen as files. `test/engines/test_frozen_models.py` checks that models saved with an older pero-ocr and torch still load and produce the same results.

- Generated from: develop @ 81b496e
- torch: 2.14.1

Do not regenerate these files to make a failing test pass. A failure means that previously exported models no longer work.

Regenerate them only on purpose, after a deliberate break of the model format:

    python -m test.tiny_models test/data/models
