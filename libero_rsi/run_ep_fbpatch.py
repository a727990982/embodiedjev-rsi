"""Run one LIBERO RSI episode. The harness itself lives in the `harness` package.

python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIBRARY.json OUTDIR [nofallback]
"""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
# The bundled TypeSafe client is part of this checkout; no external text-only
# jev_choice module may shadow the production image transport.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.episode import main

if __name__ == '__main__':
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)
