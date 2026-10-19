"""Logic smoke: spectator/gate Jev (当局者迷旁观者清) → Astra defer.

No MuJoCo / Jev / Astra. Mirrors harness rules in run_ep_fbpatch.py.
"""
import ast
import sys
from pathlib import Path

import harness_source
ASK_ASTRA_NAMES = frozenset({'ask_astra', 'consult_astra'})
FAIL_RESULTS = frozenset({'timeout', 'stalled'})
MAX_SLOW = 4
GATE_JEV_COOLDOWN = 3


def spectator_distress_signals(
    intent, result, *, goal_id_ok, n_same, hist, result_hist,
    close_refusals, stall_feedbacks,
):
    """Mirror of _spectator_distress_signals (explicit args)."""
    sig = []
    rs = str(result or '')
    h = hist or []
    rh = result_hist or []
    if not goal_id_ok:
        sig.append('identity_unconfirmed')
    if close_refusals > 0:
        sig.append('close_refusals')
    if rs.startswith('close_refused') or rs.startswith('identity_escape') or rs in (
            'relooked_unmatched', 'close_refused_identity'):
        sig.append('identity_event')
    if result in FAIL_RESULTS:
        sig.append('stall_or_timeout')
    if rs in ('empty_close',) or 'empty' in rs:
        sig.append('empty_close')
    if n_same >= 2 and intent not in ('finish', 'ask_astra', 'consult_astra'):
        sig.append('repeat_intent')
    if stall_feedbacks > 0 and (result in FAIL_RESULTS or n_same >= 2 or 'empty' in rs):
        sig.append('post_stall_feedback')
    if len(h) >= 6 and len(set(h[-6:])) <= 2:
        flips = sum(a != b for a, b in zip(h[-6:], h[-5:]))
        if flips >= 4:
            sig.append('intent_oscillation')
    if len(rh) >= 3 and sum(1 for x in rh[-3:] if x in FAIL_RESULTS or str(x).startswith('close_refused')) >= 2:
        sig.append('recent_fail_cluster')
    seen = set(); out = []
    for s in sig:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def spectator_gate_eligible(
    intent, dec, *, fallback=True, gate_on=True, slow_used=0,
    max_slow=MAX_SLOW, last_gate_dec=None, cooldown=GATE_JEV_COOLDOWN,
):
    if not fallback or not gate_on:
        return False
    if slow_used >= max_slow:
        return False
    if intent in ASK_ASTRA_NAMES:
        return False
    if last_gate_dec is not None and (dec - last_gate_dec) < cooldown:
        return False
    return True


def hybrid_astra_path(
    intent, result, *, hard_reason=None, gate_says_yes=False,
    slow_used=0, signals=None, eligible=True,
):
    """Orchestration mirror: acting ask → hard → spectator → normal.

    Returns (action, source) where action is 'ask_done'|'sync_hard'|'sync_spectator'|'slow_think'|'skip'.
    """
    if intent in ASK_ASTRA_NAMES:
        return 'ask_done', 'ask_astra'
    if hard_reason:
        return 'sync_hard', hard_reason
    if eligible and signals and gate_says_yes and slow_used < MAX_SLOW:
        return 'sync_spectator', 'spectator_jev'
    if eligible and signals and gate_says_yes and slow_used >= MAX_SLOW:
        return 'skip', 'spectator_cap'
    return 'slow_think', 'async_or_critical'


def main():
    src = harness_source.read()
    ast.parse(src)
    for needle in (
        'GATE_JEV', 'GATE_JEV_COOLDOWN',
        'def _spectator_distress_signals(',
        'def _spectator_gate_eligible(',
        'def _spectator_jev_should_ask_astra(',
        'def do_spectator_astra(',
        "source='spectator_jev'",
        'defer_to_astra',
        'continue_acting',
        'assist._spectator_gate_eligible(it, rt.dec)',
        '旁观者清',
        'gate_jev_last_dec',
        'gate_jev_calls',
    ):
        assert needle in src, needle

    # --- distress signals ---
    assert 'stall_or_timeout' in spectator_distress_signals(
        'descend_to_grasp', 'stalled', goal_id_ok=True, n_same=1, hist=[], result_hist=[],
        close_refusals=0, stall_feedbacks=0)
    assert 'identity_unconfirmed' in spectator_distress_signals(
        'move_above_object', 'reached', goal_id_ok=False, n_same=1, hist=[], result_hist=[],
        close_refusals=0, stall_feedbacks=0)
    assert 'repeat_intent' in spectator_distress_signals(
        'descend_to_grasp', 'reached', goal_id_ok=True, n_same=2, hist=[], result_hist=[],
        close_refusals=0, stall_feedbacks=0)
    # quiet progress → no signals
    assert spectator_distress_signals(
        'move_above_object', 'reached', goal_id_ok=True, n_same=1, hist=['a'], result_hist=['reached'],
        close_refusals=0, stall_feedbacks=0) == []
    # oscillation
    osc = ['x', 'y', 'x', 'y', 'x', 'y']
    assert 'intent_oscillation' in spectator_distress_signals(
        'x', 'reached', goal_id_ok=True, n_same=1, hist=osc, result_hist=['reached'] * 6,
        close_refusals=0, stall_feedbacks=0)

    # --- eligibility ---
    assert spectator_gate_eligible('descend_to_grasp', 5) is True
    assert spectator_gate_eligible('ask_astra', 5) is False  # acting already chose
    assert spectator_gate_eligible('consult_astra', 5) is False
    assert spectator_gate_eligible('descend_to_grasp', 5, slow_used=MAX_SLOW) is False
    assert spectator_gate_eligible('descend_to_grasp', 5, fallback=False) is False
    assert spectator_gate_eligible('descend_to_grasp', 5, gate_on=False) is False
    assert spectator_gate_eligible('descend_to_grasp', 5, last_gate_dec=3, cooldown=3) is False  # 5-3=2 < 3
    assert spectator_gate_eligible('descend_to_grasp', 6, last_gate_dec=3, cooldown=3) is True

    # --- orchestration ---
    # gate yes → Astra called
    act, src_ = hybrid_astra_path(
        'descend_to_grasp', 'reached', hard_reason=None, gate_says_yes=True,
        slow_used=0, signals=['repeat_intent'], eligible=True)
    assert (act, src_) == ('sync_spectator', 'spectator_jev'), (act, src_)

    # gate no → no spectator Astra (falls to slow_think)
    act, src_ = hybrid_astra_path(
        'descend_to_grasp', 'reached', hard_reason=None, gate_says_yes=False,
        slow_used=0, signals=['repeat_intent'], eligible=True)
    assert (act, src_) == ('slow_think', 'async_or_critical'), (act, src_)

    # budget exhausted → no call (eligible false or slow_used high)
    act, src_ = hybrid_astra_path(
        'descend_to_grasp', 'reached', hard_reason=None, gate_says_yes=True,
        slow_used=MAX_SLOW, signals=['stall_or_timeout'], eligible=True)
    assert (act, src_) == ('skip', 'spectator_cap'), (act, src_)
    assert spectator_gate_eligible('descend_to_grasp', 10, slow_used=MAX_SLOW) is False

    # acting ask_astra already chosen → skip spectator force
    act, src_ = hybrid_astra_path(
        'ask_astra', 'ask_astra_ok', hard_reason=None, gate_says_yes=True,
        slow_used=0, signals=['stall_or_timeout'], eligible=True)
    assert (act, src_) == ('ask_done', 'ask_astra'), (act, src_)
    act, src_ = hybrid_astra_path(
        'consult_astra', 'ask_astra_ok', hard_reason=None, gate_says_yes=True,
        slow_used=0, signals=['identity_unconfirmed'], eligible=True)
    assert (act, src_) == ('ask_done', 'ask_astra'), (act, src_)

    # hard rules still win over spectator (additive, not replacement)
    act, src_ = hybrid_astra_path(
        'descend_to_grasp', 'stalled', hard_reason='hard_stall', gate_says_yes=True,
        slow_used=0, signals=['stall_or_timeout'], eligible=True)
    assert (act, src_) == ('sync_hard', 'hard_stall'), (act, src_)

    # no signals → spectator not invoked even if "yes" would have been
    act, src_ = hybrid_astra_path(
        'move_above_object', 'reached', hard_reason=None, gate_says_yes=True,
        slow_used=0, signals=[], eligible=True)
    assert (act, src_) == ('slow_think', 'async_or_critical'), (act, src_)

    print('OK spectator_jev_gate')


if __name__ == '__main__':
    main()
