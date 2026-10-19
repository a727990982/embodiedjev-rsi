"""Source text of the episode entry point and every harness module, for structural checks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / 'run_ep_fbpatch.py', *sorted((ROOT / 'harness').glob('*.py'))]


def read():
    return '\n'.join(path.read_text() for path in PATHS)
