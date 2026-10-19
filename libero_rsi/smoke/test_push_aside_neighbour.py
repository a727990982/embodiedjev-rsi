"""Logic smoke: neighbour-block → push_aside chooser / identity gate / reapproach-after-push. No MuJoCo / Astra."""
import ast
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate as V
import dsl_libero as dsl

LIB = json.load(open(Path(__file__).resolve().parents[1] / 'libs' / 'ee_pg_easy_v2_lib.json'))
import harness_source

NEIGHBOUR_STALL_ISSUES = ('blocked_by_neighbour', 'tip_push', 'object_pushed_by_descent')
STALL_FB_ACTIONS = (
    'raise_redescend', 'lateral_recenter', 'change_approach_yaw',
    'confirm_contact', 'clear_reapproach', 'push_aside',
)


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


def choose_stall_feedback(ns, *, stall_feedbacks=0, clear_reapproach_n=0, last_stall_issue=None, id_ok=True):
    """Mirror of harness _choose_stall_feedback (keep in sync)."""
    l = bool(ns.get('left_contact')); r = bool(ns.get('right_contact')); both = bool(ns.get('contact'))
    hit = bool(ns.get('descent_hit')); empty = bool(ns.get('empty_close'))
    last = str(ns.get('last_result') or '')
    n = int(stall_feedbacks)
    n_clear = int(clear_reapproach_n)
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
    V.validate(LIB)
    src = harness_source.read()
    ast.parse(src)
    for needle in (
        "STALL_FB_ACTIONS", "_execute_push_aside", "_neighbour_block_hint",
        "_mark_need_reapproach_after_push", "push_aside_refused_identity",
        "clear_reapproach_n",
        "'push_aside'",
    ):
        assert needle in src, f'missing harness needle: {needle}'

    assert 'clear_reapproach_n' in dsl.COUNT_VARS
    assert 'push_aside_n' in dsl.COUNT_VARS
    assert any(r['id'] == 'hp_neighbour_block_push_aside' for r in LIB['recoveries'])
    push_rc = next(r for r in LIB['recoveries'] if r['id'] == 'hp_neighbour_block_push_aside')
    assert push_rc.get('stall_feedback') == 'push_aside'

    # 1) Blocker signal (unilateral) first time → clear_reapproach (not push yet)
    ns1 = base_ns(left_contact=True, right_contact=False, contact=False, last_result='stalled')
    preds1 = eval_preds(ns1)
    assert preds1['descent_hit'] is True
    assert choose_stall_feedback({**ns1, **preds1}, clear_reapproach_n=0, id_ok=True) == 'clear_reapproach'
    rc1 = first_recovery(ns1, preds1)
    assert rc1 is not None and rc1['id'] == 'hp_descent_hit_clear', rc1

    # 2) After clear exhausted + still unilateral / stalled → push_aside chosen
    ns2 = base_ns(left_contact=True, right_contact=False, contact=False, last_result='stalled',
                  clear_reapproach_n=1)
    preds2 = eval_preds(ns2)
    assert choose_stall_feedback({**ns2, **preds2}, clear_reapproach_n=1, id_ok=True) == 'push_aside'
    rc2 = first_recovery(ns2, preds2)
    assert rc2 is not None and rc2['id'] == 'hp_neighbour_block_push_aside', rc2
    assert rc2.get('stall_feedback') == 'push_aside'

    # 2b) blocked_by_neighbour issue (no pad hit) + id ok → push
    ns2b = base_ns(last_result='stalled', clear_reapproach_n=0)
    preds2b = eval_preds(ns2b)
    assert choose_stall_feedback(
        {**ns2b, **preds2b}, clear_reapproach_n=0, last_stall_issue='blocked_by_neighbour', id_ok=True
    ) == 'push_aside'

    # 2c) tip_push with no prior clear → clear first (not immediate push)
    assert choose_stall_feedback(
        {**ns2b, **preds2b}, clear_reapproach_n=0, last_stall_issue='tip_push', id_ok=True
    ) == 'clear_reapproach'

    # 3) Identity-bad → chooser must NOT pick push_aside (stay on clear / other)
    assert choose_stall_feedback({**ns2, **preds2}, clear_reapproach_n=1, id_ok=False) == 'clear_reapproach'
    assert choose_stall_feedback(
        {**ns2b, **preds2b}, clear_reapproach_n=0, last_stall_issue='blocked_by_neighbour', id_ok=False
    ) != 'push_aside'

    # Harness source: refuse path present; recovery wiring passes string action
    assert "push_aside_refused_identity" in src
    assert "isinstance(sf, str) and sf in STALL_FB_ACTIONS" in src
    assert "plan_step=st, dep=dep" in src

    # 4) After push: close gated until reapproach (reuse B03 gate semantics)
    REAPPROACH_XY_M = 0.03

    def tcp_near(tcp, grasp, obj_w=0.05):
        tol = max(REAPPROACH_XY_M, 0.5 * obj_w)
        return float(np.linalg.norm(np.asarray(tcp, float)[:2] - np.asarray(grasp, float)[:2])) <= tol

    ep = {'closed': False, 'need_reapproach': False, 'force_next_intent': None}
    # simulate mark after push (TCP still far from grasp)
    grasp = np.array([0.08, -0.10, 0.05])
    tcp_far = np.array([0.20, 0.00, 0.12])
    ep['need_reapproach'] = True
    ep['reapproach_reason'] = 'after_push_aside'
    assert not tcp_near(tcp_far, grasp)
    # close blocked
    if ep.get('need_reapproach') and not tcp_near(tcp_far, grasp):
        ep['force_next_intent'] = 'move_above_object'
        blocked = True
    else:
        blocked = False
    assert blocked is True
    assert ep['force_next_intent'] == 'move_above_object'
    # after reapproach
    tcp_near_g = grasp + np.array([0.01, 0.0, 0.02])
    assert tcp_near(tcp_near_g, grasp)
    if ep.get('need_reapproach') and tcp_near(tcp_near_g, grasp):
        ep['need_reapproach'] = False
    assert ep['need_reapproach'] is False

    # 5) Non-neighbour empty stall menu still rotates (no forced push)
    stalled = base_ns(last_result='stalled', clear_reapproach_n=0)
    preds_s = eval_preds(stalled)
    actions = [choose_stall_feedback({**stalled, **preds_s}, stall_feedbacks=i, clear_reapproach_n=0, id_ok=True)
               for i in range(4)]
    assert 'push_aside' not in actions, actions
    assert actions[0] == 'raise_redescend'

    print('OK push_aside_neighbour')


if __name__ == '__main__':
    main()
