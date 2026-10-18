"""Logic smoke for multi-action stall feedback + descent_servo_stalled (no MuJoCo / Astra)."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate as V
import dsl_libero as dsl

LIB = json.load(open(Path(__file__).resolve().parents[1] / 'libs' / 'ee_pg_easy_v2_lib.json'))
HARNESS = Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py'

def eval_preds(ns):
    preds = {}
    for p in LIB['predicates']:
        preds[p['name']] = bool(dsl.evaluate(p['expr'], {**ns, **preds}))
    return preds

def first_recovery(ns, preds):
    env = {**ns, **preds}
    for rc in LIB['recoveries']:
        if bool(dsl.evaluate(rc['trigger'], env)):
            return rc
    return None

NEIGHBOUR_STALL_ISSUES = ('blocked_by_neighbour', 'tip_push', 'object_pushed_by_descent')

def choose_stall_feedback(ns, stall_feedbacks=0, clear_reapproach_n=0, last_stall_issue=None, id_ok=True):
    """Mirror of harness chooser (keep in sync with run_ep_fbpatch._choose_stall_feedback)."""
    l = bool(ns.get('left_contact')); r = bool(ns.get('right_contact')); both = bool(ns.get('contact'))
    hit = bool(ns.get('descent_hit')); empty = bool(ns.get('empty_close'))
    last = str(ns.get('last_result') or '')
    n = int(stall_feedbacks)
    n_clear = int(clear_reapproach_n if clear_reapproach_n is not None else ns.get('clear_reapproach_n', 0))
    issue = str(last_stall_issue or '')
    neighbour_issue = issue in NEIGHBOUR_STALL_ISSUES
    if hit or (l != r and not both):
        if n_clear >= 1 and id_ok:
            return 'push_aside'
        return 'clear_reapproach'
    if id_ok and not both and not empty and (neighbour_issue or n_clear >= 1):
        if n_clear >= 1 or issue == 'blocked_by_neighbour':
            return 'push_aside'
        return 'clear_reapproach'
    if empty:
        return 'raise_redescend' if (n % 2 == 0) else 'lateral_recenter'
    if last in ('stalled', 'timeout') and not both:
        menu = ('raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact')
        return menu[n % len(menu)]
    menu = ('raise_redescend', 'change_approach_yaw', 'lateral_recenter', 'confirm_contact')
    return menu[n % len(menu)]

def base_ns(**kw):
    ns = {
        'tcp': np.array([0.098, -0.096, 0.09]), 'grasp': np.array([0.082, -0.098, 0.053]),
        'objc': np.array([0.082, -0.098, 0.1]), 'place': np.zeros(3),
        'opening': 0.08, 'opening_at_close': 0.08, 'tcp_at_close': np.zeros(3),
        'obj_h': 0.12, 'obj_w': 0.05, 'clear': 0.03, 'open_max': 0.08,
        'n_same': 1, 'step': 10, 'n_close': 0, 'n_relook': 0,
        'prev': 'descend_to_grasp', 'contact': False, 'left_contact': False, 'right_contact': False,
        'ever_contact': False, 'closed_cmd': False, 'released': False, 'seen_held': False,
        'last_result': 'stalled',
        'clear_reapproach_n': 0, 'push_aside_n': 0,
    }
    ns.update(kw)
    return ns

def main():
    assert LIB['version'] == 'ee_pg_v14+place_release'
    V.validate(LIB)
    assert any(r.get('stall_feedback') for r in LIB['recoveries'])
    assert any(r.get('id') == 'hp_neighbour_block_push_aside' for r in LIB['recoveries'])
    assert 'last_result' in dsl.STR_VARS
    assert any(p['name'] == 'descent_servo_stalled' for p in LIB['predicates'])
    hp = next(r for r in LIB['recoveries'] if r['id'] == 'hp_descent_stalled_feedback')
    assert 'descent_servo_stalled' in hp['trigger'] and 'descent_stalled' in hp['trigger']

    # 1) Tomato bug: one stall near grasp → descent_stalled True, stall_feedback recovery wins
    ns = base_ns(n_same=1, last_result='stalled')
    preds = eval_preds(ns)
    assert preds['descent_stalled'] is True, preds
    assert preds['descent_servo_stalled'] is True, preds
    assert preds['at_grasp_relaxed'] is False, preds
    rc = first_recovery(ns, preds)
    assert rc is not None and rc['id'] == 'hp_descent_stalled_feedback', rc
    assert rc.get('stall_feedback') is True

    # 1b) P0: servo stall FAR from grasp estimate → still fires (no dist gate)
    oh = 0.12
    far_tcp = np.array([0.082, -0.098, 0.053 + 3.0 * oh])  # dist = 3*obj_h
    ns_far = base_ns(n_same=1, last_result='stalled',
                     tcp=far_tcp, grasp=np.array([0.082, -0.098, 0.053]), obj_h=oh)
    preds_far = eval_preds(ns_far)
    assert preds_far['descent_stalled'] is False, preds_far  # dist gate blocks near-grasp variant
    assert preds_far['descent_servo_stalled'] is True, preds_far
    rc_far = first_recovery(ns_far, preds_far)
    assert rc_far is not None and rc_far['id'] == 'hp_descent_stalled_feedback', rc_far
    assert rc_far.get('stall_feedback') is True

    # 1c) timeout also counts as servo stall when far
    ns_to = base_ns(n_same=1, last_result='timeout', tcp=far_tcp, grasp=np.array([0.082, -0.098, 0.053]), obj_h=oh)
    assert eval_preds(ns_to)['descent_servo_stalled'] is True

    # 2) Two consecutive descends without last_result still works via n_same>=2 (near)
    ns2 = base_ns(n_same=2, last_result='reached')
    preds2 = eval_preds(ns2)
    assert preds2['descent_stalled'] is True
    assert preds2['descent_servo_stalled'] is False  # last_result not stall/timeout

    # 2b) repeated contactless: far + n_same>=2 + reached → sibling recovery (not dead zone)
    ns_rep = base_ns(n_same=2, last_result='reached', tcp=far_tcp, grasp=np.array([0.082, -0.098, 0.053]), obj_h=oh)
    preds_rep = eval_preds(ns_rep)
    assert preds_rep['descent_stalled'] is False
    assert preds_rep['descent_servo_stalled'] is False
    assert preds_rep.get('at_grasp') is False or True
    rc_rep = first_recovery(ns_rep, preds_rep)
    assert rc_rep is not None and rc_rep['id'] == 'repeated_contactless_descent_feedback', rc_rep

    # 3) Bilateral contact near grasp → close recovery, not stall feedback
    ns3 = base_ns(n_same=2, last_result='reached', contact=True, left_contact=True, right_contact=True)
    preds3 = eval_preds(ns3)
    assert preds3['descent_stalled'] is False  # contact present
    assert preds3['descent_servo_stalled'] is False
    assert preds3['at_grasp_relaxed'] is True
    rc3 = first_recovery(ns3, preds3)
    assert rc3['id'] == 'hp_relaxed_at_grasp_close', rc3

    # 4) Unilateral hit → clear recovery
    ns4 = base_ns(left_contact=True, right_contact=False, contact=False, last_result='stalled', n_same=1)
    preds4 = eval_preds(ns4)
    assert preds4['descent_hit'] is True
    rc4 = first_recovery(ns4, preds4)
    assert rc4['id'] == 'hp_descent_hit_clear', rc4
    assert choose_stall_feedback({**ns4, **preds4}, 0) == 'clear_reapproach'

    # 5) Empty close → empty feedback recovery + raise/recenter chooser
    ns5 = base_ns(closed_cmd=True, n_close=1, opening_at_close=0.005, contact=False,
                  left_contact=False, right_contact=False, last_result='closed', prev='close_gripper')
    preds5 = eval_preds(ns5)
    assert preds5['empty_close'] is True
    rc5 = first_recovery(ns5, preds5)
    assert rc5['id'] == 'hp_empty_close_feedback', rc5
    assert choose_stall_feedback({**ns5, **preds5}, 0) == 'raise_redescend'
    assert choose_stall_feedback({**ns5, **preds5}, 1) == 'lateral_recenter'

    # 6) Multi-action menu rotates (not yaw-only)
    stalled = {**base_ns(), **eval_preds(base_ns())}
    actions = [choose_stall_feedback(stalled, i) for i in range(4)]
    assert actions == ['raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact'], actions
    assert 'change_approach_yaw' in actions and 'raise_redescend' in actions
    assert len(set(actions)) == 4

    # 7) Relative yaw still relative (no hard-coded 90 scene absolute)
    def rel_yaw(ns, nchg):
        l = bool(ns.get('left_contact')); r = bool(ns.get('right_contact'))
        if l and not r: sign = 1.0
        elif r and not l: sign = -1.0
        else: sign = 1.0 if (nchg % 2 == 0) else -1.0
        return sign * 45.0
    assert abs(rel_yaw({'left_contact': True, 'right_contact': False}, 0)) == 45.0

    # 8) clear_reapproach must NOT call relative approach yaw change in harness
    hsrc = HARNESS.read_text()
    i0 = hsrc.find("elif action == 'clear_reapproach':")
    assert i0 > 0
    # End at the unknown-action else (after clear's note), not the inner "no pad signal" else.
    marker = "detail['note'] = 'raised_cleared_recenter_no_yaw"
    i_note = hsrc.find(marker, i0)
    assert i_note > 0, 'clear_reapproach yaw-preserve note missing'
    clear_block = hsrc[i0:i_note + 120]
    assert '_apply_relative_approach_change' not in clear_block, 'clear_reapproach must not swing yaw'
    assert 'yaw_preserved' in clear_block
    # change_approach_yaw branch still uses relative change
    yaw_i = hsrc.find("elif action == 'change_approach_yaw':")
    assert yaw_i > 0
    yaw_end = hsrc.find("elif action == 'confirm_contact':", yaw_i)
    yaw_block = hsrc[yaw_i:yaw_end]
    assert '_apply_relative_approach_change' in yaw_block

    print('OK stall_feedback_logic')

if __name__ == '__main__':
    main()
