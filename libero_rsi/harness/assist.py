"""Help requests: ask_astra, hard triggers, spectator Jev gate and early fallback."""

import numpy as np

import jev_choice as JC

from . import observation, planning, recovery, vision
from .config import ASK_ASTRA_NAMES, FAIL_RESULTS, GATE_JEV, GATE_JEV_COOLDOWN, MAX_FB, MAX_RELOOK, MAX_SLOW
from .state import EP, GOAL_ID, P, PATCH, T, log, rt


_SPECTATOR_CRITERIA = {
    'defer_to_astra': (
        'Yes — request Codex planning: acting looks stuck, identity doubtful, oscillating, '
        'or current images contradict task_todo progress. Shares slow-think budget.'
    ),
    'continue_acting': (
        'No — let acting Jev keep going; progress looks plausible or distress is mild / already handled.'
    ),
}


def _consume_help_budget(reason):
    """Shared MAX_SLOW slot for ask_astra / hard-force consults (even if Codex says intervene=false)."""
    if P['slow'] < MAX_SLOW:
        P['slow'] += 1
        log(t=T[0], astra_help_budget=True, reason=reason, slow_used=P['slow'], max_slow=MAX_SLOW)
        return True
    return False


def do_ask_astra(reason):
    """Jev-chosen Codex consult (existing ask_astra intent name): sync slow-think / same plan path as critical fallback. Shares MAX_SLOW."""
    if not rt.FALLBACK:
        return 'ask_astra_disabled'
    if P['slow'] >= MAX_SLOW:
        return 'ask_astra_cap'
    prev = EP.get('last_result') or 'consult'
    sr, so = planning._sync_slow('ask_astra', prev, crit=f'jev_ask:{reason}')
    # Count consult toward shared budget if _finish_slow did not already increment on intervene.
    if sr in ('slow_cap', 'slow_cap_drop'):
        return 'ask_astra_cap'
    if sr == 'intervened':
        pass  # P['slow'] already bumped in _finish_slow
    elif sr == 'ok':
        _consume_help_budget(f'ask_astra_ok:{reason}')
    elif isinstance(sr, str) and sr.startswith('sync_error'):
        return sr
    elif so is not None and P['slow'] < MAX_SLOW:
        _consume_help_budget(f'ask_astra:{sr}:{reason}')
    log(t=T[0], ask_astra=True, ask_result=sr, reason=reason, slow_used=P['slow'])
    if so is not None:
        log(t=T[0], ask_astra_detail=True, issue=(so or {}).get('issue'), intervene=(so or {}).get('intervene'),
            plan_done=(so or {}).get('plan_done'), crit=(so or {}).get('crit'))
    if sr == 'intervened':
        return 'ask_astra_intervened'
    if sr == 'ok':
        return 'ask_astra_ok'
    return f'ask_astra:{sr}'


def _hard_astra_trigger(intent, result):
    """Critical physical stalls retain one bounded diagnosis after general help is exhausted."""
    if not rt.FALLBACK:
        return None
    if observation._recovery_context()['no_progress'] and MAX_SLOW > 0:
        if P['slow'] < MAX_SLOW or not EP.get('motion_help_used'):
            return 'recovery_no_progress'
    if result in ('stalled', 'timeout') and MAX_SLOW > 0:
        if P['slow'] < MAX_SLOW or not EP.get('motion_help_used'):
            return 'motion_obstruction'
    if P['slow'] >= MAX_SLOW:
        return None
    if intent in ASK_ASTRA_NAMES:
        return None  # already consulted via execute
    if result in FAIL_RESULTS:
        return 'hard_stall'
    if not vision.goal_id_confirmed():
        rs = str(result or '')
        if (rs.startswith('close_refused') or rs.startswith('identity_escape')
                or rs in ('relooked_unmatched', 'close_refused_identity')
                or (intent not in ('relook', 'finish') and EP.get('n_same', 0) >= 2)):
            return 'hard_identity_unstable'
    return None


def _spectator_distress_signals(intent, result):
    """Soft distress cues for spectator gate (hard rules already cover stall/identity force).
    Empty list => skip spectator this step (no spam)."""
    sig = []
    rs = str(result or '')
    h = EP.get('hist') or []
    rh = EP.get('result_hist') or []
    if not vision.goal_id_confirmed():
        sig.append('identity_unconfirmed')
    if GOAL_ID.get('close_refusals', 0) > 0:
        sig.append('close_refusals')
    if rs.startswith('close_refused') or rs.startswith('identity_escape') or rs in (
            'relooked_unmatched', 'close_refused_identity'):
        sig.append('identity_event')
    if result in FAIL_RESULTS:
        sig.append('stall_or_timeout')
    if rs in ('empty_close',) or 'empty' in rs:
        sig.append('empty_close')
    if EP.get('n_same', 0) >= 2 and intent not in ('finish', 'ask_astra', 'consult_astra'):
        sig.append('repeat_intent')
    if EP.get('stall_feedbacks', 0) > 0 and (
            result in FAIL_RESULTS or EP.get('n_same', 0) >= 2 or 'empty' in rs):
        sig.append('post_stall_feedback')
    if len(h) >= 6 and len(set(h[-6:])) <= 2:
        flips = sum(a != b for a, b in zip(h[-6:], h[-5:]))
        if flips >= 4:
            sig.append('intent_oscillation')
    if len(rh) >= 3 and sum(1 for x in rh[-3:] if x in FAIL_RESULTS or str(x).startswith('close_refused')) >= 2:
        sig.append('recent_fail_cluster')
    # de-dupe preserve order
    seen = set(); out = []
    for s in sig:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def _spectator_gate_eligible(intent, dec):
    """Budget / cooldown / enable checks before spending a spectator Jev call."""
    if not rt.FALLBACK or not GATE_JEV:
        return False
    if P['slow'] >= MAX_SLOW:
        return False
    if intent in ASK_ASTRA_NAMES:
        return False  # acting Jev already chose ask_astra
    last = EP.get('gate_jev_last_dec')
    if last is not None and (dec - last) < GATE_JEV_COOLDOWN:
        return False
    return True


def _spectator_jev_state(intent, result, signals, ns, facts):
    """Current image/todo/proprioception plus actual execution events."""
    state, assets = observation._jev_input()
    state['role'] = 'spectator_gate'
    state['execution_events'] = {
        'just_finished_intent': intent, 'intent_result': result,
        'recent_intents': (EP.get('hist') or [])[-8:],
        'recent_intent_results': (EP.get('result_hist') or [])[-8:],
        'slow_used': P['slow'], 'max_slow': MAX_SLOW,
    }
    return state, assets


def _spectator_jev_should_ask_astra(intent, result, signals, ns, facts, dec):
    """Light spectator Jev call → (defer: bool, detail dict). Updates cooldown on every call."""
    EP['gate_jev_last_dec'] = dec
    EP['gate_jev_calls'] = int(EP.get('gate_jev_calls') or 0) + 1
    state, assets = _spectator_jev_state(intent, result, signals, ns, facts)
    instr = (
        'Spectator gate only: choose defer_to_astra OR continue_acting. '
        'Prefer defer_to_astra when execution_events and current images show failure/repetition without recovery, '
        'or current images contradict task_todo progress. Prefer continue_acting when the last result '
        'shows progress, relook just matched, or budget is nearly spent. One short decisive choice.'
    )
    meta = {**rt.META, 'phase': 'spectator_jev_gate', 'signals': list(signals)}
    try:
        choice, probs, lat = JC.choose(state, instr, _SPECTATOR_CRITERIA, assets=assets, meta=meta)
        defer = choice == 'defer_to_astra'
        detail = {'choice': choice, 'probabilities': probs, 'latency_s': lat, 'signals': list(signals)}
        log(dec=dec, t=T[0], spectator_jev=True, defer=defer, **{k: detail[k] for k in ('choice', 'signals', 'latency_s')})
        return defer, detail
    except Exception as e:
        detail = {'choice': None, 'error': str(e)[:200], 'signals': list(signals), 'defer': False}
        log(dec=dec, t=T[0], spectator_jev=True, defer=False, error=detail['error'], signals=list(signals))
        return False, detail


def do_spectator_astra(intent, result, reason):
    """Spectator said yes → sync Codex path (same budget as ask_astra / hard force)."""
    if not rt.FALLBACK:
        return 'spectator_disabled', None
    if P['slow'] >= MAX_SLOW:
        return 'spectator_cap', None
    sr, so = planning._sync_slow(intent, result, crit=f'spectator_jev:{reason}')
    if sr in ('slow_cap', 'slow_cap_drop'):
        return 'spectator_cap', so
    if sr == 'intervened':
        pass
    elif sr == 'ok':
        _consume_help_budget(f'spectator_ok:{reason}')
    elif isinstance(sr, str) and sr.startswith('sync_error'):
        return sr, so
    elif so is not None and P['slow'] < MAX_SLOW:
        _consume_help_budget(f'spectator:{sr}:{reason}')
    log(t=T[0], spectator_astra=True, ask_result=sr, reason=reason, slow_used=P['slow'])
    return sr, so


def do_fallback(trig):
    """Refresh visual evidence through Astra, then ask Codex for the recovery plan."""
    if P['relook'] >= MAX_RELOOK:
        esc = recovery._escape_identity_deadlock(f'fallback_relook_exhausted:{trig}')
        return f'escape_after_budget:{esc}', {**PATCH, 'escape': True, 'trigger': trig, 'escape_result': esc}
    # Revoke pending advice before changing the target it was based on.
    planning._abandon_async_slow(reason=f'fallback:{trig}')
    perception = vision.do_relook(f'fallback:{trig}')
    if perception == 'relook_error':
        return 'fallback_perception_error', None
    result, detail = planning._sync_slow('fallback', perception, f'fallback:{trig}')
    return result, {**PATCH, 'trigger': trig, 'perception': perception, 'slow_detail': detail}


def fb_trigger():
    if not rt.FALLBACK or P['fb'] >= MAX_FB: return None
    # general stage-fail (shared debounce with slow-think via stage_fail_mark)
    sf = planning._stage_fail_grip_unchanged(consume=True)
    if sf: return sf
    h = EP['hist']; th = EP['tcp_hist']
    if len(h) >= 4 and len(set(h[-4:])) == 1 and h[-1] not in ('finish',) and np.linalg.norm(th[-1] - th[-4]) < .01: return 'no_progress_same_intent'
    if len(h) >= 8 and len(set(h[-8:])) <= 2 and sum(a != b for a, b in zip(h[-8:], h[-7:])) >= 6: return 'intent_oscillation'
    if EP['n_close'] >= 3 and EP['n_close'] % 3 == 0 and EP.get('fb_close_mark') != EP['n_close']:
        EP['fb_close_mark'] = EP['n_close']; return 'repeated_grasp_attempts'
    return None
