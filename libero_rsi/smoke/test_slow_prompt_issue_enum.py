"""Smoke: Codex monitor issue enum aligned with applicator neighbour-block classes."""
import ast
import re
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py'
sys.path.insert(0, str(HARNESS.parent))
import coding_agent_slow as CAS

REQUIRED = (
    'blocked_by_neighbour',
    'object_pushed_by_descent',
    'wrong_object',
    'tip_push',
    'empty_grasp',
    'place_miss',
    'other',
    'ok',
)


def main():
    src = HARNESS.read_text()
    ast.parse(src)
    prompt = CAS.build_prompt({'language_goal': 'put object in basket'})
    for lab in REQUIRED:
        assert lab in CAS.ALLOWED_ISSUES
        assert lab in prompt
    # Applicator still keys off neighbour classes
    assert "stall_issue = issue in ('empty_grasp', 'tip_push', 'blocked_by_neighbour', 'object_pushed_by_descent')" in src
    print('OK Codex monitor issue enum')



if __name__ == '__main__':
    main()
