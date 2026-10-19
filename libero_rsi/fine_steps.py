"""Bounded single-axis corrections for the Jev menu.

Default off. ``run_ep_fbpatch`` merges ``libs/ee_pg_fine_steps.json`` only when
``FINE_STEPS`` is not ``0``. Each option servos the current gripper along one
world axis by five millimetres, one centimetre, or three centimetres and keeps
the current gripper command.

``fine_unit`` (one centimetre) is a controller spec, like ``open_max``, so the
menu can write those steps as dimensionless factors. The harness still clamps
every fine-step target to three centimetres per axis before ``servo``. The
servo tolerance is tighter than the six-millimetre move early-out, otherwise
the five-millimetre option would be reported as already at the target.
"""
import json
import os

import numpy as np

FINE_UNIT_M = 0.01
FINE_STEP_MAX_M = 0.03
FINE_STEP_TOL_M = 0.002
FINE_STEP_MAX_STEPS = 40
MERGE_TIP = (
    'Fine-step menu: intents named fine_* keep the current gripper state and servo '
    'along one world axis by five millimetres, one centimetre, or three centimetres. '
    'Use one when the grasp or place estimate is only slightly off. '
    'After it runs, read observed_facts.last_fine_step for the servo result and the new end-effector position.'
)
_AXES = (
    ('x', 0, 'positive world x', 'negative world x'),
    ('y', 1, 'positive world y', 'negative world y'),
    ('z', 2, 'upward world z', 'downward world z'),
)
_SIZES = (
    ('5mm', '0.5*fine_unit', 'five millimetres'),
    ('1cm', 'fine_unit', 'one centimetre'),
    ('3cm', '3*fine_unit', 'three centimetres'),
)


def enabled_flag(value=None):
    """True only when FINE_STEPS is set to something other than 0. Unset is off."""
    if value is None:
        value = os.environ.get('FINE_STEPS', '0')
    return str(value) != '0'


def fine_library_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs', 'ee_pg_fine_steps.json')


def load_fine_library(path=None):
    with open(path or fine_library_path()) as f:
        return json.load(f)


def build_overlay():
    """Valid standalone library. Merge skips finish; do not run this file as the episode menu."""
    intents = []
    for axis, idx, plus_words, minus_words in _AXES:
        for sign, sign_name, words in ((1, 'plus', plus_words), (-1, 'minus', minus_words)):
            for size_name, factor, size_words in _SIZES:
                name = f'fine_{axis}_{sign_name}_{size_name}'
                op = '+' if sign > 0 else '-'
                comps = ['tcp[0]', 'tcp[1]', 'tcp[2]']
                comps[idx] = f'tcp[{idx}] {op} {factor}'
                intents.append({
                    'id': f'{name}@v0',
                    'name': name,
                    'action': 'fine_step',
                    'fingers': 'keep',
                    'definition': (
                        f'Keep the current gripper state and servo along {words} by {size_words}.'
                    ),
                    'refpoint': f'vec({comps[0]}, {comps[1]}, {comps[2]})',
                    'gain': '1',
                    'status': 'hypothesis',
                    'evidence': 'bounded world-axis correction; harness clamps to three centimetres per axis',
                    'tilt_deg': '0',
                    'tilt_dir_deg': '0',
                })
    intents.append({
        'id': 'finish@v0',
        'name': 'finish',
        'action': 'finish',
        'fingers': 'keep',
        'definition': 'Stop. Present so this overlay validates; merge does not replace the episode finish intent.',
        'refpoint': 'tcp',
        'status': 'hypothesis',
        'evidence': 'overlay sentinel',
    })
    return {
        'version': 'ee_pg_fine_steps_v1',
        'note': (
            'Overlay menu of bounded world-axis translations. Not a pick-and-place policy. '
            'Enable with FINE_STEPS=1 so the episode library gains these intents. '
            'Do not point an experiment at this file alone. '
            'Sizes are five millimetres (0.5*fine_unit), one centimetre (fine_unit), and three centimetres '
            '(3*fine_unit), with fine_unit = 0.01 m. Servo tolerance is 2 mm so the smallest step is not '
            'swallowed by the 6 mm move early-out.'
        ),
        'intent_instructions': MERGE_TIP,
        'merge_instructions': MERGE_TIP,
        'intents': intents,
        'predicates': [],
        'recoveries': [],
        'lessons': [],
    }


def merge_fine_steps(lib, overlay):
    """Insert overlay fine_step intents just before finish. Idempotent. Does not replace finish."""
    names = {i.get('name') for i in lib.get('intents', [])}
    extra = []
    for it in overlay.get('intents', []):
        if it.get('action') != 'fine_step':
            continue
        if it.get('fingers', 'keep') != 'keep':
            raise ValueError(f"fine step {it.get('name')} must keep the current gripper")
        if it.get('name') in names:
            continue
        extra.append(json.loads(json.dumps(it)))
        names.add(it.get('name'))
    intents = lib.setdefault('intents', [])
    if extra:
        fin_i = next((i for i, it in enumerate(intents) if it.get('name') == 'finish'), len(intents))
        intents[fin_i:fin_i] = extra
    tip = str(overlay.get('merge_instructions') or MERGE_TIP).strip()
    instr = lib.get('intent_instructions') or ''
    if tip and tip not in instr:
        lib['intent_instructions'] = (instr.rstrip() + ' ' + tip).strip()
    return lib


def apply_fine_steps(lib, enabled, overlay=None):
    """Merge when enabled. When disabled, leave the loaded library untouched and do not read the overlay."""
    if not enabled:
        return lib
    if overlay is None:
        overlay = load_fine_library()
    merge_fine_steps(lib, overlay)
    if not any(i.get('action') == 'fine_step' for i in lib.get('intents', [])):
        raise RuntimeError('FINE_STEPS is on, but no fine_step intents were added')
    return lib


def clamp_fine_target(tcp, target, max_m=FINE_STEP_MAX_M):
    """Clamp a world-frame target to ±max_m on each axis relative to the current TCP.

    Returns (target, delta, status). status is 'ok' or 'rejected'. A rejected
    command must not be servoed.
    """
    try:
        origin = np.asarray(tcp, dtype=float).reshape(3)
        goal = np.asarray(target, dtype=float).reshape(3)
    except Exception:
        return np.zeros(3), np.zeros(3), 'rejected'
    delta = goal - origin
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(delta)):
        safe = origin.copy() if np.all(np.isfinite(origin)) else np.zeros(3)
        return safe, np.zeros(3), 'rejected'
    cap = abs(float(max_m))
    limited = np.clip(delta, -cap, cap)
    return origin + limited, limited, 'ok'


def fine_step_report(intent, result, tcp, delta, closed):
    """Text Jev can read: what was commanded, whether it landed, and where the gripper is now."""
    ee = [round(float(x), 4) for x in np.asarray(tcp, float).reshape(3)]
    dlt = [round(float(x), 4) for x in np.asarray(delta, float).reshape(3)]
    grip = 'closed' if closed else 'open'
    text = (
        f'{intent} {result}; end-effector now at [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}] m; '
        f'commanded delta [{dlt[0]:.4f}, {dlt[1]:.4f}, {dlt[2]:.4f}] m; gripper {grip}'
    )
    return {
        'intent': intent,
        'result': result,
        'ee_xyz_m': ee,
        'commanded_delta_m': dlt,
        'gripper_closed': bool(closed),
        'text': text,
    }


def fine_step_fact(enabled, last):
    """Fields for observed_facts. Empty when the feature is off and no fine step has run."""
    if not enabled and not last:
        return {}
    out = {'last_fine_step': last}
    if isinstance(last, dict) and last.get('text'):
        out['last_fine_step_text'] = last['text']
    return out
