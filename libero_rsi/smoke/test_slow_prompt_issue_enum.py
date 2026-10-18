"""Smoke: SLOW_PROMPT issue enum aligned with applicator neighbour-block classes."""
import ast
import re
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py'

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
    # Locate SLOW_PROMPT string via assignment
    m = re.search(r'SLOW_PROMPT\s*=\s*"""(.*?)"""', src, re.S)
    assert m, 'SLOW_PROMPT not found'
    slow = m.group(1)
    # Issue enum line
    im = re.search(r'"issue":\s*one of ([^\n]+)', slow)
    assert im, 'issue enum line missing in SLOW_PROMPT'
    enum_line = im.group(1)
    for lab in REQUIRED:
        assert lab in enum_line, f'missing issue label {lab!r} in {enum_line}'
    # Applicator still keys off neighbour classes
    assert "stall_issue = issue in ('empty_grasp', 'tip_push', 'blocked_by_neighbour', 'object_pushed_by_descent')" in src
    # Fallback already had them
    fm = re.search(r'FALLBACK_PROMPT\s*=\s*"""(.*?)"""', src, re.S) or re.search(
        r'"failure_class":\s*one of ([^\n]+)', src)
    assert 'blocked_by_neighbour' in src and 'object_pushed_by_descent' in src
    print('OK slow_prompt_issue_enum', enum_line.strip()[:120])


if __name__ == '__main__':
    main()
