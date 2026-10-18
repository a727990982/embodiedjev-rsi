"""Logic smoke: relook target change → block close until re-approach (B03). No MuJoCo / Astra."""
import ast
from pathlib import Path
import numpy as np

HARNESS = Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py'
REAPPROACH_XY_M = 0.03


def xy(v):
    return np.asarray(v, float).reshape(-1)[:2]


def tol(obj_w=0.05):
    return max(REAPPROACH_XY_M, 0.5 * obj_w)


def tcp_near_grasp(tcp, grasp, obj_w=0.05):
    return float(np.linalg.norm(xy(tcp) - xy(grasp))) <= tol(obj_w)


def mark_reapproach(ep, old_grasp, new_grasp, tcp, obj_w=0.05, reason='relook'):
    """Mirror of harness _mark_reapproach_if_needed (open gripper)."""
    if ep.get('closed'):
        return False
    near = tcp_near_grasp(tcp, new_grasp, obj_w)
    if near:
        return False
    ep['need_reapproach'] = True
    ep['reapproach_reason'] = reason
    if ep.get('force_next_intent') == 'close_gripper':
        ep['force_next_intent'] = 'move_above_object'
    return True


def close_allowed(ep, tcp, grasp, goal_id_ok=True, obj_w=0.05):
    """Mirror of execute close gates relevant to B03."""
    if not goal_id_ok:
        return False, 'identity'
    if ep.get('need_reapproach') and not tcp_near_grasp(tcp, grasp, obj_w):
        ep['force_next_intent'] = 'move_above_object'
        return False, 'need_reapproach'
    if ep.get('need_reapproach') and tcp_near_grasp(tcp, grasp, obj_w):
        ep['need_reapproach'] = False
    return True, 'closed'


def main():
    src = HARNESS.read_text()
    ast.parse(src)
    for needle in (
        'need_reapproach', '_mark_reapproach_if_needed', '_tcp_near_grasp',
        'close_blocked_need_reapproach', 'close_blocked_reapproach',
        '_clear_reapproach_if_aligned', 'REAPPROACH_XY_M',
        "both_contact_but_need_reapproach",
    ):
        assert needle in src, f'missing: {needle}'

    # Scenario: TCP over A, relook → B elsewhere ⇒ close blocked until reapproach
    A = np.array([0.10, 0.00, 0.05])
    B = np.array([0.25, 0.10, 0.05])  # ~0.18 m XY away
    tcp_over_A = np.array([0.10, 0.00, 0.08])
    ep = {'closed': False, 'need_reapproach': False, 'force_next_intent': None}

    assert mark_reapproach(ep, A, B, tcp_over_A, reason='keypoints:relook') is True
    assert ep['need_reapproach'] is True

    ok, why = close_allowed(ep, tcp_over_A, B, goal_id_ok=True)
    assert ok is False and why == 'need_reapproach', (ok, why)
    assert ep['force_next_intent'] == 'move_above_object'

    # After re-approach to B, close allowed
    tcp_over_B = np.array([0.25, 0.10, 0.06])
    ok2, why2 = close_allowed(ep, tcp_over_B, B, goal_id_ok=True)
    assert ok2 is True and why2 == 'closed', (ok2, why2)
    assert ep['need_reapproach'] is False

    # Small nudge while already near: no block
    ep2 = {'closed': False, 'need_reapproach': False, 'force_next_intent': None}
    A2 = np.array([0.10, 0.00, 0.05])
    A2b = np.array([0.11, 0.005, 0.05])  # ~1cm
    tcp = np.array([0.10, 0.00, 0.06])
    assert mark_reapproach(ep2, A2, A2b, tcp) is False  # still near new grasp
    ok3, _ = close_allowed(ep2, tcp, A2b, goal_id_ok=True)
    assert ok3 is True

    # Pending close invalidated on mark
    ep3 = {'closed': False, 'need_reapproach': False, 'force_next_intent': 'close_gripper'}
    assert mark_reapproach(ep3, A, B, tcp_over_A) is True
    assert ep3['force_next_intent'] == 'move_above_object'

    # Holding: never force reapproach-for-close
    ep4 = {'closed': True, 'need_reapproach': False, 'force_next_intent': None}
    assert mark_reapproach(ep4, A, B, tcp_over_A) is False

    print('test_reapproach_before_close: OK')


if __name__ == '__main__':
    main()
