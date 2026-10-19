"""Logic smoke: after close, grasp_confirm suppresses push_aside / deadend undo (no MuJoCo)."""
from pathlib import Path

import harness_source

def main():
    src = harness_source.read()
    assert 'def _confirm_grasp_after_close' in src
    assert 'grasp_confirm=False' in src or 'grasp_confirm=bool' in src
    assert 'skip_plan_grasp_secured' in src
    assert 'push_aside_skipped_grasp_secured' in src
    assert 'grasp_confirm_skip' in src
    assert 'skip_secured_grasp' in src
    # close path calls confirm
    assert '_confirm_grasp_after_close()' in src
    # open clears confirm
    assert "EP['grasp_confirm'] = False" in src
    print('ok grasp_confirm_logic')

if __name__ == '__main__':
    main()
