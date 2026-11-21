"""The stable dependency basis (requirements/lock.in): frozen data and golden files are generated only in an environment
that has exactly these versions, so that they are what every live worker stack produces (issue #94)."""
import os
import re
from importlib import metadata

LOCK_IN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'requirements', 'lock.in')


def pinned_versions(path=LOCK_IN):
    """{distribution name: version} of the top-level pins in ``path``."""
    pins = {}
    with open(path) as f:
        for line in f:
            match = re.match(r'\s*([A-Za-z0-9_.\-]+)==(\S+)', line)
            if match:
                pins[match.group(1)] = match.group(2)
    return pins


def mismatches(pins=None):
    """Descriptions of the pinned packages whose installed version differs (or that are missing)."""
    problems = []
    for name, version in sorted((pins or pinned_versions()).items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f'{name}: not installed, the basis pins {version}')
            continue
        if installed != version:
            problems.append(f'{name}: installed {installed}, the basis pins {version}')
    return problems


def require_basis(allow_other=False):
    """Stop a regeneration script unless it runs in the basis environment (or ``allow_other`` is set)."""
    problems = mismatches()
    if problems and not allow_other:
        raise SystemExit('Not the basis environment (requirements/lock.txt), refusing to regenerate:\n  ' +
                         '\n  '.join(problems) +
                         '\nInstall the lock (pip install -r requirements/lock.txt && pip install --no-deps -e .) '
                         'or pass --any-environment to regenerate anyway.')
    return problems
