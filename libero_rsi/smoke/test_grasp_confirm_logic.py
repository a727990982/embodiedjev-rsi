"""Logic smoke: after close, grasp_confirm suppresses push_aside / deadend undo (no MuJoCo)."""
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py'
CAS = Path(__file__).resolve().parents[1] / 'coding_agent_slow.py'

def main():
    src = HARNESS.read_text()
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
    cas = CAS.read_text()
    assert 'grasp_secured' in cas and 'grasp confirmed' in cas.lower()
    print('ok grasp_confirm_logic')

if __name__ == '__main__':
    main()
