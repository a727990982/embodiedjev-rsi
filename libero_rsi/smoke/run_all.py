"""Run every smoke test as its own process and summarize the results.

    python libero_rsi/smoke/run_all.py [-v]

Exit status: 0 all passed, 1 a test failed, 2 a required package is missing.
"""
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = {'numpy': 'numpy', 'PIL': 'Pillow', 'scipy': 'scipy'}


def missing_packages():
    return [package for module, package in REQUIRED.items() if importlib.util.find_spec(module) is None]


def main(argv):
    verbose = '-v' in argv
    missing = missing_packages()
    if missing:
        print(f"Missing Python packages: {', '.join(missing)}\n"
              f"Install them with: pip install -r requirements.txt", file=sys.stderr)
        return 2
    if not shutil.which('ffmpeg'):
        print('Note: ffmpeg not found; the episode video test will be skipped.')
    failed = []
    tests = sorted(HERE.glob('test_*.py'))
    for test in tests:
        start = time.monotonic()
        run = subprocess.run([sys.executable, str(test)], capture_output=True, text=True, timeout=300)
        status = 'ok' if run.returncode == 0 else 'FAIL'
        print(f'{status:4}  {test.name}  ({time.monotonic() - start:.1f}s)')
        if run.returncode != 0:
            failed.append(test.name)
        if run.returncode != 0 or verbose:
            print((run.stdout + run.stderr).rstrip())
    print(f'\n{len(tests) - len(failed)}/{len(tests)} smoke tests passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
