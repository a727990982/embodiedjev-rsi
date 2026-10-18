"""Short actionable experience brief for Jev intent choice.

Recoveries + lessons are summarized for the decision-maker — not a full prose dump.
Triggers stay general (predicate / failure-class level); no scene-specific angles.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Predicate / signal tokens that make a lesson "relevant" when present in facts.
_SIGNAL_HINTS = {
    'descent_stalled': ('stall', 'stalled', 'descent', 'align', 'approach', 'grasp'),
    'descent_servo_stalled': ('stall', 'stalled', 'descent', 'servo', 'timeout', 'align', 'approach', 'grasp', 'clearance'),
    'descent_hit': ('contact', 'hit', 'blocked', 'descent', 'push', 'blocker', 'neighbour', 'neighbor'),
    'empty_close': ('empty', 'close', 'capture', 'grasp', 'relook', 'align'),
    'at_grasp': ('grasp', 'close', 'contact'),
    'at_grasp_relaxed': ('grasp', 'close', 'contact'),
    'held': ('held', 'lift', 'carry', 'release'),
    'above_grasp': ('above', 'approach', 'descend'),
    'above_place': ('place', 'release', 'lower'),
    'at_place': ('place', 'release', 'lower'),
}


def _clip(s: Any, n: int = 140) -> str:
    t = ' '.join(str(s or '').split())
    return t if len(t) <= n else t[: n - 1] + '…'


def _true_preds(facts: dict) -> set:
    out = set()
    for k, v in (facts or {}).items():
        if v is True and isinstance(k, str):
            out.add(k)
    return out


def _recovery_hint(rc: dict) -> str:
    bits = []
    if rc.get('stall_feedback'):
        bits.append('stall feedback (raise/recenter/change approach/confirm contact/clear)')
    if rc.get('approach_change'):
        bits.append('relative approach change')
    sh = rc.get('shape') or {}
    if sh:
        keys = ','.join(sorted(sh.keys()))
        bits.append(f'shape override ({keys})')
    return '; '.join(bits) if bits else ''


def _recovery_line(rc: dict, uses: dict) -> dict:
    uid = rc.get('id', '')
    max_u = int(rc.get('max_uses', 3 if 'force_intent' in rc else 1))
    used = int((uses or {}).get(uid, 0))
    line = {
        'id': uid,
        'when': _clip(rc.get('trigger'), 100),
        'prefer': rc.get('force_intent') or None,
        'uses_left': max(0, max_u - used),
        'status': rc.get('status'),
    }
    h = _recovery_hint(rc)
    if h:
        line['hint'] = h
    return line


def _lesson_score(lesson: dict, true_preds: set, facts: dict) -> float:
    blob = ' '.join(
        str(lesson.get(k) or '') for k in ('name', 'signature', 'instead')
    ).lower()
    score = 0.0
    tokens = set()
    for p in true_preds:
        tokens.add(p.lower())
        tokens.update(_SIGNAL_HINTS.get(p, ()))
        tokens.add(p.lower().replace('_', ' '))
    last = str(facts.get('last_intent_result') or facts.get('last_result') or '').lower()
    if last in ('stalled', 'timeout'):
        tokens.update(('stall', 'stalled', 'descent', 'align', 'approach'))
    if facts.get('both_fingers_contact') is False and last == 'stalled':
        tokens.update(('empty', 'contactless', 'miss'))
    prev = str(facts.get('previous_intent') or '').lower()
    if prev:
        tokens.add(prev.replace('_', ' '))
    for t in tokens:
        if t and t in blob:
            score += 1.0
    st = lesson.get('status')
    if st == 'verified':
        score += 0.5
    elif st == 'hypothesis':
        score += 0.1
    return score


def _lesson_line(lesson: dict) -> dict:
    return {
        'id': lesson.get('id'),
        'when': _clip(lesson.get('signature') or lesson.get('name'), 110),
        'instead': _clip(lesson.get('instead'), 110),
        'status': lesson.get('status'),
    }


def _episode_lesson_line(item: dict, i: int) -> dict:
    return {
        'id': f'ep_fb_{i}',
        'when': _clip(item.get('failure_class') or item.get('diagnosis'), 90),
        'instead': _clip(item.get('lesson'), 110),
        'status': 'episode',
    }


def build_jev_experience_brief(
    lib: dict,
    facts: Optional[dict] = None,
    ns: Optional[dict] = None,
    uses: Optional[dict] = None,
    episode_lessons: Optional[List[dict]] = None,
    episode_start_plan: Optional[dict] = None,
    dsl=None,
    max_recoveries: int = 6,
    max_lessons: int = 5,
    max_active: int = 3,
) -> Dict[str, Any]:
    """Build a compact experience_brief for JC.choose state.

    - active_recoveries: triggers true right now (if ns+dsl given), still have uses left
    - recoveries: short catalog of when→prefer (+hint)
    - lessons: relevant when→instead (scored against current facts), not full library prose
    - episode_reflections: in-episode fallback lessons, if any (short)
    - episode_start_plan: Great Wall start prior (step1_must / phase_order / risks); follow unless contradicted
    """
    facts = facts or {}
    uses = uses or {}
    true_preds = _true_preds(facts)

    active: List[dict] = []
    catalog: List[dict] = []
    for rc in lib.get('recoveries', []) or []:
        line = _recovery_line(rc, uses)
        catalog.append(line)
        if ns is not None and dsl is not None and line['uses_left'] > 0:
            try:
                if bool(dsl.evaluate(rc['trigger'], ns)):
                    active.append(line)
            except Exception:
                pass
    catalog = catalog[:max_recoveries]
    active = active[:max_active]

    scored = []
    for les in lib.get('lessons', []) or []:
        scored.append((_lesson_score(les, true_preds, facts), les))
    scored.sort(key=lambda x: (-x[0], 0 if x[1].get('status') == 'verified' else 1, x[1].get('id') or ''))
    # Prefer positive scores; if none match, still surface a few verified / early priors (capped).
    picked = [L for s, L in scored if s > 0][:max_lessons]
    if not picked:
        picked = [L for s, L in scored[:max_lessons]]
    lessons = [_lesson_line(L) for L in picked]

    ep_lines = []
    for i, item in enumerate(episode_lessons or []):
        if not item:
            continue
        ep_lines.append(_episode_lesson_line(item, i))
        if len(ep_lines) >= 3:
            break

    brief = {
        'note': (
            'Actionable library advice for this decision. Prefer intents consistent with '
            'active_recoveries and matching lessons. Triggers are general; do not invent '
            'scene-specific angles or object brands.'
        ),
        'active_recoveries': active,
        'recoveries': catalog,
        'lessons': lessons,
    }
    if ep_lines:
        brief['episode_reflections'] = ep_lines
    if episode_start_plan and isinstance(episode_start_plan, dict):
        # Compact prior for Jev: follow unless evidence contradicts; mid slow-think fine-tunes only.
        brief['episode_start_plan'] = {
            'note': (
                'Episode-start Great Wall prior. Prefer intents consistent with step1_must / '
                'phase_order unless current contact/identity evidence contradicts. Mid-episode '
                'slow-think only fine-tunes; do not replan from zero.'
            ),
            'step1_must': episode_start_plan.get('step1_must'),
            'phase_order': list(episode_start_plan.get('phase_order') or [])[:8],
            'predicted_risks': list(episode_start_plan.get('predicted_risks') or [])[:6],
            'issue': episode_start_plan.get('issue'),
            'plan': list(episode_start_plan.get('plan') or [])[:3],
        }
    return brief


def jev_choose_state(
    task: str,
    facts: dict,
    lib: dict,
    ns=None,
    uses=None,
    episode_lessons=None,
    episode_start_plan=None,
    dsl=None,
) -> dict:
    """Full JC.choose state dict including experience_brief."""
    return {
        'task': task,
        'observed_facts': facts,
        'predicate_meanings': {
            p['name']: p.get('meaning', '') for p in lib.get('predicates', [])
        },
        'experience_brief': build_jev_experience_brief(
            lib, facts=facts, ns=ns, uses=uses,
            episode_lessons=episode_lessons,
            episode_start_plan=episode_start_plan, dsl=dsl,
        ),
    }


def jev_choose_instructions(lib: dict, extra: str = '') -> str:
    base = lib.get(
        'intent_instructions',
        'Select the next intent whose definition matches observed_facts.',
    )
    tip = (
        ' Also weigh experience_brief: active recoveries (when→prefer) and relevant '
        'lessons (when→instead). Stay consistent with them when facts match. '
        'If experience_brief.episode_start_plan is present, honour step1_must / phase_order '
        'unless current evidence clearly contradicts (then recover via stall menu / ask_astra).'
    )
    return base + tip + ((' ' + extra) if extra else '')
