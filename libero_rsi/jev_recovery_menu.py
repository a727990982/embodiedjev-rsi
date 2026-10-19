"""Opt-in discrete menu for stall recoveries that the harness otherwise auto-picks.

Default mode is ``rules``: ``JEV_RECOVERY_MENU`` unset. Jev's criteria and observed
facts stay as they are, and library ``stall_feedback`` / ``approach_change`` rules
still call ``_choose_stall_feedback``.

``jev`` puts these actions on the menu and skips those auto-rules so a rule cannot
move the arm on the same decision Jev was about to choose. Explicit Codex plan
steps still execute. Identity-deadlock escape and the local-deadend cap stay on
in every mode; they are freeze/horizon guards, not the per-stall menu.

``both`` offers the menu and still runs the auto-rules. A matching rule runs
before ``JC.choose`` and can force the next intent, so Jev does not see that
decision.

``confirm_contact`` stays off this menu. The rule chooser and Codex plans can
still request it.
"""
from __future__ import annotations

RECOVERY_ACTIONS = (
    'raise_redescend',
    'lateral_recenter',
    'change_approach_yaw',
    'clear_reapproach',
    'push_aside',
)

_RULES = {
    '', '0', 'off', 'false', 'rules', 'rule', 'auto',
}
_JEV = {
    '1', 'jev', 'on', 'true', 'yes', 'menu',
}
_BOTH = {
    'both', 'jev+rules', 'rules+jev',
}

INSTRUCTION_TIP = (
    ' When descent stalls, times out, hits one finger, or the fingers close on nothing, '
    'select raise_redescend, lateral_recenter, change_approach_yaw, or clear_reapproach '
    'instead of repeating the failed motion. When a neighbour blocks a confirmed grasp, '
    'select push_aside or clear_reapproach, and do not push the language-goal object. '
    'Read last_recovery_outcome and recent_recovery_outcomes before repeating a recovery.'
)


def parse_mode(raw):
    """Return ``(mode, warning)`` with mode in ``rules``, ``jev``, or ``both``.

    Unknown text warns and stays on ``rules`` so a typo cannot enable the menu.
    ``None`` and ``''`` are the default rules arm.
    """
    if raw is None:
        raw = ''
    key = str(raw).strip().lower()
    if key in _RULES:
        return 'rules', None
    if key in _JEV:
        return 'jev', None
    if key in _BOTH:
        return 'both', None
    return 'rules', f'unknown JEV_RECOVERY_MENU={raw!r}; using rules'


def menu_enabled(mode):
    return mode in ('jev', 'both')


def rules_auto_choose_enabled(mode):
    """Hand-written stall chooser may fill an empty plan or a ``stall_feedback: true`` rule.

    ``jev`` disables that chooser path. Explicit plan steps still name an action.
    """
    return mode != 'jev'


def recovery_rule_skipped(rc, mode):
    """Skip a library recovery that would auto-run a menu action.

    Place-release and other ``force_intent`` / ``shape`` rules still run.
    ``both`` and ``rules`` do not skip.
    """
    if mode != 'jev':
        return False
    return bool((rc or {}).get('stall_feedback') or (rc or {}).get('approach_change'))


def is_recovery_intent(intent):
    intent = intent or {}
    return intent.get('name') in RECOVERY_ACTIONS or intent.get('action') in RECOVERY_ACTIONS


def acting_intents(intents, mode):
    """Intents Jev may select. Rules mode hides the recovery actions."""
    intents = list(intents or [])
    if menu_enabled(mode):
        return intents
    return [i for i in intents if not is_recovery_intent(i)]


def recovery_intent_specs():
    """Intent records inserted when the menu is on and the library lacks them.

    Definitions avoid numeric lengths so ``validate.py`` accepts them.
    ``action`` is not ``move``, so the harness dispatches to the stall routine
    instead of servoing a refpoint.
    """
    common = {
        'fingers': 'keep',
        'refpoint': 'tcp',
        'status': 'hypothesis',
        'evidence': 'jev recovery menu',
        'tilt_deg': '0',
        'tilt_dir_deg': '0',
    }
    rows = (
        ('raise_redescend',
         'Descent stalled, timed out, or the fingers closed on nothing: raise clear of the object, shallow the grasp, and re-approach from above.'),
        ('lateral_recenter',
         'The gripper is offset from the object centre after a miss or stall: shift the grasp estimate toward the centre, raise, and re-approach.'),
        ('change_approach_yaw',
         'The current approach angle is blocked or keeps missing: raise and rotate the approach about vertical, then re-approach.'),
        ('clear_reapproach',
         'One finger hit a neighbour or the descent is blocked: raise clear, recenter in the plane, and re-approach without changing yaw.'),
        ('push_aside',
         'A neighbour blocks the grasp and identity is confirmed: push the blocker aside with open fingers, then re-approach the language-goal object. Do not push the goal itself. Refused when identity is unconfirmed, a grasp is already secured, or the push cap is reached.'),
    )
    out = []
    for name, definition in rows:
        out.append({
            'id': f'{name}@v0',
            'name': name,
            'action': name,
            'definition': definition,
            **common,
        })
    return out


def ensure_recovery_intents(lib):
    """Insert any missing recovery intents before ``finish``. Idempotent."""
    intents = lib.setdefault('intents', [])
    names = {i.get('name') for i in intents}
    fin_i = next((i for i, it in enumerate(intents) if it.get('name') == 'finish'), len(intents))
    for spec in recovery_intent_specs():
        if spec['name'] in names:
            continue
        intents.insert(fin_i, dict(spec))
        fin_i += 1
        names.add(spec['name'])
    instr = lib.get('intent_instructions') or ''
    if 'raise_redescend' not in instr:
        lib['intent_instructions'] = instr + INSTRUCTION_TIP
    return lib


def instruction_extra(mode):
    """Mode-specific sentence appended to Jev's instructions. Empty in rules mode."""
    if mode == 'jev':
        return (
            'Recovery menu is on and automatic stall-feedback rules are off. '
            'When descent stalls, times out, hits one finger, closes on nothing, or a neighbour blocks a confirmed grasp, '
            'select raise_redescend, lateral_recenter, change_approach_yaw, clear_reapproach, or push_aside. '
            'Read last_recovery_outcome and recent_recovery_outcomes before repeating an action. '
            'push_aside is refused when identity is unconfirmed, a grasp is secured, or the push cap is reached.'
        )
    if mode == 'both':
        return (
            'Recovery menu is on and automatic stall-feedback rules still run. '
            'A matching rule executes before your choice and can force the next motion, so you will not see every stalled state. '
            'When you do choose, raise_redescend, lateral_recenter, change_approach_yaw, clear_reapproach, and push_aside are available. '
            'Read last_recovery_outcome and recent_recovery_outcomes before repeating an action.'
        )
    return ''


def join_extra(*parts):
    return ' '.join(p.strip() for p in parts if p and str(p).strip())


def remember(ep, text, cap=8):
    hist = ep.setdefault('recovery_outcomes', [])
    hist.append(str(text))
    if len(hist) > cap:
        del hist[:-cap]
    return hist


def format_recovery_outcome(requested, detail, force_next=None):
    """One colon-separated outcome line for Jev's text state and ``last_intent_result``."""
    detail = detail or {}
    requested = str(requested or detail.get('action') or 'recovery')
    actual = str(detail.get('action') or requested)
    via = str(detail.get('reason') or '').replace(':', '/').replace(' ', '_')
    via_bit = f'via_{via[:60]}' if via else ''

    def finish(parts):
        if force_next:
            parts.append('next_' + str(force_next))
        if via_bit:
            parts.append(via_bit)
        return ':'.join(str(p) for p in parts if p is not None and str(p) != '')

    if actual == 'skip_secured_grasp':
        return finish([requested, 'skipped', 'grasp_secured'])
    if actual == 'local_deadend_global_relook':
        why = str(detail.get('deadend_reason') or 'local_deadend').replace(':', '/').replace(' ', '_')
        return finish([requested, 'intercepted', 'local_deadend', why])

    err = detail.get('error')
    if err:
        err_s = str(err).replace(':', '/').replace(' ', '_')
        return finish([requested, 'refused', err_s])

    if requested == 'push_aside' or actual == 'push_aside':
        if detail.get('pushed'):
            parts = ['push_aside', 'ok', 'pushed', str(detail.get('blocker_mode') or 'heuristic')]
            servo = detail.get('servo')
            if servo:
                parts.append('servo_' + str(servo))
            if detail.get('need_reapproach'):
                parts.append('need_reapproach')
            if detail.get('local_deadend_global_relook'):
                why = str(detail.get('deadend_reason') or 'local_deadend').replace(':', '/').replace(' ', '_')
                parts.append('then_local_deadend')
                parts.append(why)
            return finish(parts)
        return finish(['push_aside', 'not_pushed'])

    parts = [requested, 'ok']
    if requested == 'raise_redescend':
        parts.append('raised_and_shallowed')
    elif requested == 'lateral_recenter':
        parts.append('shifted_toward_centre')
    elif requested == 'change_approach_yaw':
        parts.append('yaw_rotated' if detail.get('yaw_to') is not None else 'yaw_unchanged')
    elif requested == 'clear_reapproach':
        parts.append('raised_recenter_yaw_kept')
    else:
        parts.append(str(detail.get('confirm') or actual))
    if detail.get('need_reapproach'):
        parts.append('need_reapproach')
    if detail.get('local_deadend_global_relook'):
        why = str(detail.get('deadend_reason') or 'local_deadend').replace(':', '/').replace(' ', '_')
        parts.append('then_local_deadend')
        parts.append(why)
    return finish(parts)


def recovery_fact_fields(ep, mode, push_cap=2, push_approach_cap=1):
    """Extra observed_facts. Caller must pass these only when the menu is enabled."""
    ep = ep or {}
    hist = [str(x) for x in (ep.get('recovery_outcomes') or [])]
    push_n = int(ep.get('push_aside_n') or 0)
    push_ap = int(ep.get('push_aside_approach_n') or 0)
    issue = ep.get('last_stall_issue') or 'none'
    return {
        'recovery_menu': mode,
        'last_recovery_outcome': hist[-1] if hist else 'none',
        'recent_recovery_outcomes': hist[-5:],
        'clear_reapproach_count': int(ep.get('clear_reapproach_n') or 0),
        'push_aside_count': push_n,
        'push_aside_approach_count': push_ap,
        'push_aside_capped': bool(push_n >= int(push_cap) or push_ap >= int(push_approach_cap)),
        'need_reapproach': bool(ep.get('need_reapproach')),
        'stall_feedback_count': int(ep.get('stall_feedbacks') or 0),
        'last_stall_issue': issue,
    }


def facts_extra(mode, ep, push_cap=2, push_approach_cap=1):
    """Rules mode adds nothing, so the default Jev state keeps its existing keys."""
    if not menu_enabled(mode):
        return {}
    return recovery_fact_fields(ep, mode, push_cap=push_cap, push_approach_cap=push_approach_cap)
