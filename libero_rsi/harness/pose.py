"""Optional pose menu (POSE_MENU): library intents and their geometric execution.

Pose intents are installed into the library at start-up when POSE_MENU is set, and are executed
as bounded servo programs relative to the current visual estimates and the start pose."""

import os

import numpy as np

from . import robot, vision
from .state import EP, PATCH, PER, RSHAPE, SHAPE, T, log, nonfatal, rt


def _pose_menu_enabled():
    return os.environ.get('POSE_MENU', '0').strip().lower() not in ('', '0', 'false', 'off', 'no')


def _pose_menu_kinds():
    """None means do not filter. A set limits which pose families stay in the menu."""
    raw = os.environ.get('POSE_MENU_KINDS', '').strip()
    if not raw:
        return None
    return frozenset(part.strip() for part in raw.split(',') if part.strip())


def _pose_kind_of(it):
    pose = (it or {}).get('pose')
    if pose in ('rotate', 'rotate_toward_astra'):
        return 'rotate'
    if pose == 'align':
        return 'align'
    if pose == 'home':
        return 'home'
    return None


def _pose_menu_intent_specs():
    """Discrete pose-menu intents. Degrees in the text are dimensionless; clearances stay in the harness."""
    specs = []

    def add(name, pose, definition, **extra):
        specs.append({
            'id': f'{name}@v1', 'name': name, 'action': 'pose_menu', 'pose': pose, 'fingers': 'keep',
            'definition': definition, 'status': 'hypothesis', 'evidence': 'pose menu', **extra,
        })

    add('rotate_yaw_plus_15', 'rotate',
        'Hold position and rotate gripper yaw by 15 degrees positive about vertical.', yaw_delta_deg=15)
    add('rotate_yaw_minus_15', 'rotate',
        'Hold position and rotate gripper yaw by 15 degrees negative about vertical.', yaw_delta_deg=-15)
    add('rotate_yaw_plus_45', 'rotate',
        'Hold position and rotate gripper yaw by 45 degrees positive about vertical.', yaw_delta_deg=45)
    add('rotate_yaw_minus_45', 'rotate',
        'Hold position and rotate gripper yaw by 45 degrees negative about vertical.', yaw_delta_deg=-45)
    add('rotate_yaw_toward_astra', 'rotate_toward_astra',
        'Hold position and rotate gripper yaw by 15 degrees toward the estimated grasp yaw.', yaw_step_deg=15)
    for side in ('top', 'left', 'right', 'front', 'back'):
        for clearance in ('small', 'medium', 'large'):
            if side == 'top':
                definition = (
                    f'Move to the top of the target object at {clearance} clearance, '
                    'using the grasp point, object top centre, and object height estimates.'
                )
            else:
                definition = (
                    f'Move to the {side} of the target object at {clearance} clearance. '
                    'Keep the current height and face the gripper toward the object. '
                    'Uses the grasp point, object top centre, and object height estimates.'
                )
            add(f'align_{side}_{clearance}', 'align', definition, side=side, clearance=clearance)
    add('return_home', 'home',
        'Return to the initial end-effector pose for recovery. Finger command stays as it is.')
    return specs


def _pose_menu_tip(lib):
    kinds = {_pose_kind_of(i) for i in lib.get('intents', []) if i.get('action') == 'pose_menu'}
    parts = []
    if 'rotate' in kinds:
        parts.append('rotate the gripper in place by 15 or 45 degrees, or by 15 degrees toward the estimated grasp yaw')
    if 'align' in kinds:
        parts.append('align to the top, left, right, front, or back of the target at a small, medium, or large clearance')
    if 'home' in kinds:
        parts.append('return to the home pose')
    if not parts:
        return ''
    return (
        ' Pose menu: you may ' + '; or '.join(parts) +
        '. In-place rotation holds position. Side alignments keep the current height and turn the gripper toward the object. '
        'Read pose_action_outcome before repeating one.'
    )


def _pose_menu_visible():
    if _pose_menu_enabled():
        return True
    return any(i.get('action') == 'pose_menu' for i in (rt.LIB or {}).get('intents', []))


def _install_pose_menu(lib):
    """Inject pose intents when POSE_MENU=1, then apply POSE_MENU_KINDS. Default env: no change."""
    changed = False
    if _pose_menu_enabled():
        names = {i['name'] for i in lib.get('intents', [])}
        fin_i = next((i for i, it in enumerate(lib['intents']) if it['name'] == 'finish'), len(lib['intents']))
        for spec in _pose_menu_intent_specs():
            if spec['name'] in names:
                continue
            lib['intents'].insert(fin_i, dict(spec))
            fin_i += 1
            names.add(spec['name'])
            changed = True
    kinds = _pose_menu_kinds()
    if kinds is not None:
        kept = []
        for it in lib.get('intents', []):
            if it.get('action') == 'pose_menu' and _pose_kind_of(it) not in kinds:
                changed = True
                continue
            kept.append(it)
        lib['intents'] = kept
    instr = lib.get('intent_instructions', '') or ''
    tip = _pose_menu_tip(lib)
    if tip and 'Pose menu:' not in instr:
        lib['intent_instructions'] = instr + tip
        changed = True
    return changed


def _wrap_deg(angle):
    return float((float(angle) + 180.0) % 360.0 - 180.0)


def _yaw_step(current, delta):
    current = _wrap_deg(current)
    step = float(delta)
    if abs(step) < 0.5:
        return current, 0.0
    return _wrap_deg(current + step), step


def _yaw_toward(current, target, step_deg):
    """Step at most step_deg along the short arc from current toward target."""
    current = _wrap_deg(current)
    delta = _wrap_deg(float(target) - current)
    step = float(np.clip(delta, -abs(float(step_deg)), abs(float(step_deg))))
    if abs(step) < 0.5:
        return current, 0.0
    return _wrap_deg(current + step), step


def _unit_xy(v):
    v = np.asarray(v, float).reshape(-1)[:2]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return v / n


def _object_side_dirs(objc_xy, home_xy, cam_xy=None):
    """Front is from the object toward the initial gripper (robot side). Left is 90 deg CCW of front."""
    front = _unit_xy(np.asarray(home_xy, float) - np.asarray(objc_xy, float))
    if front is None and cam_xy is not None:
        front = _unit_xy(np.asarray(cam_xy, float) - np.asarray(objc_xy, float))
    if front is None:
        front = np.array([0.0, -1.0])
    left = np.array([-front[1], front[0]])
    return {'front': front, 'back': -front, 'left': left, 'right': -left}


def _pose_body_radius(grasp, objc, obj_h):
    """Horizontal body proxy from the Astra grasp point, object top centre, and object height."""
    grasp = np.asarray(grasp, float).reshape(3)
    objc = np.asarray(objc, float).reshape(3)
    radial = float(np.linalg.norm(grasp[:2] - objc[:2]))
    return float(max(radial, 0.5 * abs(float(obj_h))))


def _yaw_face_point(from_xy, to_xy):
    """Yaw whose forward axis (cos yaw, sin yaw) points at to_xy. Open axis stays perpendicular."""
    d = _unit_xy(np.asarray(to_xy, float) - np.asarray(from_xy, float))
    if d is None:
        return None
    return _wrap_deg(float(np.degrees(np.arctan2(d[1], d[0]))))


def _pose_clearance_m(name):
    """Guava align clearances: small 0.05 m, medium 0.08 m, large 0.15 m."""
    defaults = {'small': 0.05, 'medium': 0.08, 'large': 0.15}
    env_key = {'small': 'POSE_CLEAR_SMALL_M', 'medium': 'POSE_CLEAR_MEDIUM_M', 'large': 'POSE_CLEAR_LARGE_M'}[name]
    try:
        return float(os.environ.get(env_key, defaults[name]))
    except ValueError:
        return float(defaults[name])


def _align_target(side, clearance_m, tcp, grasp, objc, obj_h, home_xy, cam_xy=None):
    tcp = np.asarray(tcp, float).reshape(3).copy()
    objc = np.asarray(objc, float).reshape(3)
    if side == 'top':
        tgt = objc.copy()
        tgt[2] = float(objc[2]) + float(clearance_m)
        return tgt, None
    dirs = _object_side_dirs(objc[:2], home_xy, cam_xy)
    if side not in dirs:
        raise ValueError(f'unknown side {side}')
    radius = _pose_body_radius(grasp, objc, obj_h)
    tgt = tcp.copy()
    tgt[:2] = objc[:2] + dirs[side] * (radius + float(clearance_m))
    tgt[2] = float(tcp[2])
    return tgt, _yaw_face_point(tgt[:2], objc[:2])


def _pose_grip():
    return 1.0 if EP.get('closed') else -1.0


def _finite3(v):
    try:
        a = np.asarray(v, float).reshape(-1)
        return a.size >= 3 and bool(np.all(np.isfinite(a[:3])))
    except (TypeError, ValueError):
        return False


def _pose_note(kind, text):
    EP['pose_menu_note'] = text
    counts = EP.setdefault('pose_menu_counts', {})
    counts[kind] = int(counts.get(kind, 0)) + 1
    with nonfatal('pose_menu_log'):
        log(t=T[0], pose_menu=True, kind=kind, outcome=text)
    return text


def _execute_pose_menu(it, ns):
    """Run one pose-menu intent through servo. Does not call a detector."""
    pose = it.get('pose')
    kind = _pose_kind_of(it) or 'pose'
    grip = _pose_grip()
    if pose in ('rotate', 'rotate_toward_astra'):
        hold = np.asarray(robot.eef(), float).copy()
        cur = float(vision._cmd_yaw_deg())
        if pose == 'rotate_toward_astra':
            astra = float(PER.get('yaw', 0.0) or 0.0)
            new_yaw, step = _yaw_toward(cur, astra, float(it.get('yaw_step_deg', 15)))
            if abs(step) < 0.5:
                _pose_note(kind, f'in-place rotate toward Astra yaw {astra:.1f}: already aligned; position held')
                return 'at_target'
            label = f'in-place rotate {abs(step):.1f} deg toward Astra yaw {astra:.1f}'
        else:
            new_yaw, step = _yaw_step(cur, float(it.get('yaw_delta_deg', 0)))
            label = f'in-place rotate {step:+.1f} deg'
        PATCH['yaw_deg'] = float(new_yaw)
        result = robot.servo(hold, grip, 0.6, max_steps=80, tol=0.01, ori_tol=0.08)
        _pose_note(kind, f'{label}: commanded yaw {cur:.1f} -> {new_yaw:.1f}; position held; servo {result}')
        return result
    if pose == 'align':
        if not (_finite3(PER.get('grasp')) and _finite3(PER.get('objc')) and np.isfinite(float(PER.get('obj_h', np.nan)))):
            _pose_note(kind, 'align refused: missing grasp point, object top centre, or object height estimate')
            return 'pose_menu_no_estimate'
        side = str(it.get('side') or '')
        clearance_name = str(it.get('clearance') or 'medium')
        try:
            clearance = _pose_clearance_m(clearance_name)
        except KeyError:
            _pose_note(kind, f'align refused: unknown clearance {clearance_name}')
            return 'pose_menu_bad_clearance'
        tcp = np.asarray(ns.get('tcp', robot.eef()), float)
        cam = None if rt.CP is None else np.asarray(rt.CP, float).reshape(-1)[:2]
        home_xy = tcp[:2] if rt.HOME_EEF is None else np.asarray(rt.HOME_EEF, float).reshape(-1)[:2]
        try:
            tgt, yaw = _align_target(side, clearance, tcp, PER['grasp'], PER['objc'], PER['obj_h'], home_xy, cam)
        except ValueError as ex:
            _pose_note(kind, f'align refused: {ex}')
            return 'pose_menu_bad_side'
        if yaw is not None:
            PATCH['yaw_deg'] = float(yaw)
        result = robot.servo(tgt, grip, 0.8, max_steps=80, tol=0.01, ori_tol=0.08 if yaw is not None else None)
        where = 'above object top centre' if side == 'top' else 'height held and gripper yaw faces the object'
        _pose_note(kind, (
            f'align {side} clearance {clearance_name} ({clearance:.3f} m): '
            f'target {[round(float(x), 3) for x in tgt]}; {where}; servo {result}'))
        return result
    if pose == 'home':
        if rt.HOME_EEF is None:
            _pose_note(kind, 'return home refused: initial end-effector pose was not captured')
            return 'pose_menu_no_home'
        home = np.asarray(rt.HOME_EEF, float).reshape(3).copy()
        PATCH['yaw_deg'] = 0.0
        PATCH['tilt_deg'] = 0.0
        PATCH['tilt_dir_deg'] = 0.0
        result = robot.servo(home, grip, 1.0, max_steps=80, tol=0.01, ori_tol=0.08)
        if result == 'reached':
            for key in ('yaw_deg','tilt_deg','tilt_dir_deg'): PATCH[key] = None
            for key in ('yaw','tilt','tdir'):
                RSHAPE[key] = None
                SHAPE[key] = None
        _pose_note(kind, f'return home: target {home.tolist()}; servo {result}; '
                   f'orientation overrides cleared={result == "reached"}; fingers unchanged')
        return result
    _pose_note(kind, f'pose menu ignored unknown pose {pose}')
    return 'pose_menu_unknown'
