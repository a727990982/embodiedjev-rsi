"""Astra perception, goal-identity gate and grasp-target ownership.

One PER keypoint base plus one PATCH offset layer own the grasp target. Relook / keypoint
refresh overwrites PER and clears position offsets in PATCH; confirmed approach yaw/tilt in
PATCH survive unless new keypoints explicitly change yaw."""

import json
import os
import time

import numpy as np
from PIL import Image

import task_todo as TODO
import robot_geometry as RG

from . import observation, robot
from .config import ASTRA_EFFORT, MAX_ESCAPE_LOOK, MAX_RELOOK, PLACE_CLEARANCE_DEFAULT, REAPPROACH_XY_M, S, YAW_EXPLICIT_EPS_DEG
from .state import (_empty_patch, EP, GOAL_ID, GOAL_TARGET, ID_EVENTS, P, PATCH, PER, RSHAPE, SHAPE, START_ID, T, log,
                    nonfatal, rt, warn)


def identity_fields(p):
    m = p.get('matched_language_goal')
    if m is True or str(m).lower() == 'true': matched = True
    elif m is False or str(m).lower() == 'false': matched = False
    else: matched = None
    return {'goal_object_named': p.get('goal_object_named'), 'goal_object_where': p.get('goal_object_where'),
            'place_region_named': p.get('place_region_named'), 'place_region_where': p.get('place_region_where'),
            'matched_language_goal': matched, 'confidence': p.get('confidence')}


def identity_ok(p):
    return identity_fields(p)['matched_language_goal'] is True


def set_goal_id(confirmed, source, detail=None):
    ok = confirmed is True
    GOAL_ID['confirmed'] = ok
    GOAL_ID['source'] = source
    GOAL_ID['detail'] = detail
    GOAL_ID['events'].append({'confirmed': ok, 'source': source, 'detail': detail, 't': T[0]})
    return ok


def goal_id_confirmed():
    return GOAL_ID.get('confirmed') is True


def _active_language_goal():
    item = TODO.active_item(EP.get('task_todo'))
    if item is None:
        return rt.LANG
    return f"{rt.LANG}\nCurrent TODO target: {item['target']}. Current objective: {item['goal']}. Localize and verify ONLY this target."


def perceive(context=None, phase=None):
    """Use Codex CLI to ask Astra for visual keypoints at start and on relook."""
    import coding_agent_slow as CAS
    from PIL import Image as _PILImage
    rgb, dep = robot.rgbd()
    ph = phase or ('perceive' if not context else 'relook')
    frame = os.path.join(rt.ep_dir, f'perc_{ph}_{P["astra_calls"]}.jpg')
    _PILImage.fromarray(rgb).save(frame)
    images = [frame]
    wrist = rt.obs.get('robot0_eye_in_hand_image')
    if wrist is not None:
        wrist_frame = os.path.join(rt.ep_dir, f'perc_{ph}_{P["astra_calls"]}_wrist.jpg')
        _PILImage.fromarray(wrist[::-1]).save(wrist_frame)
        images.append(wrist_frame)
    geometry, annotated = observation.astra_geometry(f'perceive_{ph}_{P["astra_calls"]}')
    images.extend(annotated)
    visual_context = json.dumps({'request_context': context, 'robot_geometry': geometry})
    parsed, rec = CAS.harness_call_perceive(
        lang=_active_language_goal(), phase=ph, image_paths=images, context=visual_context,
        img_size=S, mode='codex', effort=ASTRA_EFFORT)
    observation.remember_geometry_advice(parsed.get('robot_geometry_advice'), geometry, ph)
    P['astra_cost'] += rec['cost_usd']; P['astra_calls'] += 1; P['last_astra_wall'] = time.time()
    log(t=T[0], perception=True, phase=ph, backend=rec.get('backend'),
        model=rec.get('model'), effort=rec.get('effort'), wall_s=rec.get('wall_s'))
    return parsed, rgb, dep


def _reset_patch(reason='relook_overwrite'):
    """Clear episode offsets so stale position patches cannot fight a newly locked keypoint base.
    Approach yaw/tilt may be restored by _commit_keypoints after a relook overwrite."""
    old = {k: (list(v) if isinstance(v, list) else v) for k, v in PATCH.items()}
    PATCH.clear(); PATCH.update(_empty_patch())
    GOAL_TARGET['rev'] = int(GOAL_TARGET.get('rev', 0)) + 1
    with nonfatal('patch_reset_log'):
        log(t=T[0], patch_reset=True, reason=reason, old_patch=old, goal_target=dict(GOAL_TARGET))
    return old


def _new_keypoints_explicit_yaw(p, old_per_yaw, cmd_yaw=None, eps=YAW_EXPLICIT_EPS_DEG):
    """True when fresh keypoints clearly propose a NEW approach yaw vs the current command AND prior PER.
    Echoing PER (e.g. yaw=0) must NOT wipe a confirmed PATCH yaw. Pass cmd_yaw from BEFORE any PATCH reset."""
    if not isinstance(p, dict) or p.get('grasp_yaw_deg') is None:
        return False
    try:
        new_y = float(p['grasp_yaw_deg'])
        if cmd_yaw is not None:
            cmd = float(cmd_yaw)
        else:
            cmd = float(_cmd_yaw_deg())
        if abs(new_y - cmd) < float(eps):
            return False  # agrees with current command — keep PATCH
        if old_per_yaw is not None and abs(new_y - float(old_per_yaw)) < float(eps):
            return False  # relook echoed PER/default — do not reset confirmed PATCH to 0
        return True
    except (TypeError, ValueError):
        return False


def _restore_approach_pose(old_patch, *, explicit_new_yaw):
    """After relook clears PATCH: keep confirmed yaw/tilt unless new keypoints explicitly changed yaw."""
    if explicit_new_yaw or not old_patch:
        return {}
    restored = {}
    for k in ('yaw_deg', 'tilt_deg', 'tilt_dir_deg'):
        v = old_patch.get(k)
        if v is not None:
            try:
                PATCH[k] = float(v)
                restored[k] = PATCH[k]
            except (TypeError, ValueError):
                pass
    return restored


def _commit_keypoints(p, dep, source, held_now=False):
    """Write the shared keypoint BASE (PER). Relook / keypoint-refresh overwrites PER and clears position offsets.
    Confirmed approach yaw/tilt in PATCH survive unless new keypoints explicitly change yaw.
    Episode-start only seeds PER. Slow-think adjust/stall must NOT go through here — use _note_patch_mutation.
    A material target XY change while open marks need_reapproach so close cannot fire at the old TCP."""
    old_patch = {k: (list(v) if isinstance(v, list) else v) for k, v in PATCH.items()}
    old_per_yaw = PER.get('yaw')
    old_grasp = np.asarray(PER['grasp'], float).copy() if PER.get('grasp') is not None else None
    # Capture commanded yaw BEFORE reset clears PATCH (needed for explicit-yaw gate).
    try:
        old_cmd_yaw = float(old_patch['yaw_deg']) if old_patch.get('yaw_deg') is not None else (
            float(RSHAPE['yaw']) if RSHAPE.get('yaw') is not None else float(PER.get('yaw', 0.0) or 0.0)
        )
    except (TypeError, ValueError):
        old_cmd_yaw = old_per_yaw
    set_perc(p, dep, held_now=held_now)
    src = str(source or 'unknown')
    # Anything that refreshes pixels owns the base at relook priority (except pure episode_start seed).
    is_start = src in ('episode_start', 'perceive_start')
    preserved = {}
    if not is_start:
        _reset_patch(reason=f'overwrite:{src}')
        # Escape looks must not reset confirmed PATCH yaw to a default/PER echo.
        if src == 'escape_look':
            explicit = False
        else:
            explicit = _new_keypoints_explicit_yaw(p, old_per_yaw, cmd_yaw=old_cmd_yaw)
        preserved = _restore_approach_pose(old_patch, explicit_new_yaw=explicit)
        GOAL_TARGET['source'] = 'relook'
        with nonfatal('approach_pose_log'):
            log(t=T[0], approach_pose_preserve=True, source=src, explicit_new_yaw=explicit,
                old_per_yaw=old_per_yaw, new_per_yaw=PER.get('yaw'),
                old_patch_yaw=old_patch.get('yaw_deg'), preserved=preserved or None,
                goal_target=dict(GOAL_TARGET))
    else:
        GOAL_TARGET['source'] = 'episode_start'
    GOAL_TARGET['base_from'] = src
    GOAL_TARGET['rev'] = int(GOAL_TARGET.get('rev', 0)) + 1
    # Skip start-seed commits (EP is still empty); only mid-episode refreshes can unlock a stale close.
    if (not is_start and not held_now and src not in ('perceive_start_relook',)
            and 'need_reapproach' in EP):
        _mark_reapproach_if_needed(old_grasp, f'keypoints:{src}')
    with nonfatal('commit_keypoints_log'):
        log(t=T[0], goal_commit_keypoints=True, source=src, goal_target=dict(GOAL_TARGET),
            approach_preserved=preserved or None,
            need_reapproach=bool(EP.get('need_reapproach')),
            grasp=list(np.asarray(PER.get('grasp'), float).round(4)) if 'grasp' in PER else None)
    return dict(GOAL_TARGET)


def _note_patch_mutation(source, detail=None):
    """Slow-think / stall feedback mutates the SAME shared PATCH on the current PER base (does not create a second goal)."""
    GOAL_TARGET['source'] = 'slow_patch'
    GOAL_TARGET['last_patch_from'] = str(source or 'patch')
    GOAL_TARGET['rev'] = int(GOAL_TARGET.get('rev', 0)) + 1
    with nonfatal('patch_mutation_log'):
        log(t=T[0], goal_patch_mutation=True, source=source, detail=detail, patch=dict(PATCH), goal_target=dict(GOAL_TARGET))
    return dict(GOAL_TARGET)


def set_perc(p, dep, held_now=False):
    PER['seen_held'] = bool(p.get('object_held_by_gripper', False))
    PER['place'] = robot.bp(p['place_px'], dep)
    if not held_now or 'grasp' not in PER:
        PER['grasp'] = robot.bp(p['grasp_px'], dep) - [0, 0, float(np.clip(p.get('grasp_depth_m', .02), 0, .06))]
        PER['objc'] = robot.bp(p.get('object_center_px', p['grasp_px']), dep)
        PER['obj_h'] = float(np.clip(p.get('object_height_m', .05), .015, .2))
        PER['yaw'] = float(p.get('grasp_yaw_deg', 0)); PER['obj_w'] = float(np.clip(p.get('object_width_m') or PER['obj_h'] * .6, .01, .2))
    PER['clear'] = float(np.clip(p.get('place_clearance_m', PLACE_CLEARANCE_DEFAULT), 0, .1))
    PER['identity'] = identity_fields(p)


def verify_grasp_identity(p, rgb, phase='perceive_start_verify'):
    """Mark grasp_px and ask Astra through Codex CLI whether THAT marked body is the language-goal object.
    Catches over-confident wrong locks (claimed match but pixel on a neighbour). No appearance hard-coding."""
    from PIL import ImageDraw as _ID
    im = Image.fromarray(rgb.copy()); d = _ID.Draw(im)
    try: u, v = int(round(p['grasp_px'][0])), int(round(p['grasp_px'][1]))
    except (KeyError, TypeError, ValueError, IndexError): return {'confirmed': None, 'reason': 'no_grasp_px'}, rgb
    r = 14; d.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0), width=3)
    d.line([u - 18, v, u + 18, v], fill=(255, 0, 0), width=2); d.line([u, v - 18, u, v + 18], fill=(255, 0, 0), width=2)
    d.text((6, 6), 'RED MARK = proposed grasp', fill=(255, 0, 0))
    prompt = (f'Task instruction (language goal): "{_active_language_goal()}".\n'
              f'The RED circle+cross marks the proposed grasp pixel from a prior look.\n'
              f'Using ONLY the language goal and THIS image (do not rely on memorized brand looks):\n'
              f'{{"marked_object_what": short phrase naming what body the mark sits on,\n'
              f' "is_language_goal_object": true if that marked body IS the object the instruction asks to move, false if it is a different object, null if unsure,\n'
              f' "language_goal_where_if_different": if false/null, where the true language-goal object is in this image (or "not visible"), else null,\n'
              f' "reason": one short sentence}}\nReply ONLY JSON.')
    import coding_agent_slow as CAS
    frame = os.path.join(rt.ep_dir, f'identity_{phase}_{P["astra_calls"]}.jpg')
    im.save(frame)
    geometry, annotated = observation.astra_geometry(f'identity_{phase}_{P["astra_calls"]}')
    prompt += RG.INSTRUCTIONS + '\nrobot_geometry: ' + json.dumps(geometry)
    images = [frame]
    wrist = rt.obs.get('robot0_eye_in_hand_image')
    if wrist is not None:
        wrist_frame = os.path.join(rt.ep_dir, f'identity_{phase}_{P["astra_calls"]}_wrist.jpg')
        Image.fromarray(wrist[::-1]).save(wrist_frame)
        images.append(wrist_frame)
    parsed, rec = CAS.call_identity_check(prompt, images + annotated, effort=ASTRA_EFFORT)
    observation.remember_geometry_advice(parsed.get('robot_geometry_advice'), geometry, phase)
    log(t=T[0], identity_verification=True, phase=phase, backend=rec.get('backend'),
        model=rec.get('model'), effort=rec.get('effort'), wall_s=rec.get('wall_s'))
    P['astra_cost'] += rec['cost_usd']; P['astra_calls'] += 1; P['last_astra_wall'] = time.time()
    return parsed, np.asarray(im)


def _xy(v):
    a = np.asarray(v, float).reshape(-1)
    return a[:2]


def _reapproach_tol():
    return max(REAPPROACH_XY_M, 0.5 * float(PER.get('obj_w', 0.03) or 0.03))


def _tcp_near_grasp(tcp=None, grasp=None, tol=None):
    """True when TCP XY is close enough to the effective grasp to allow close."""
    if grasp is None and any(PER.get(k) is None for k in ('grasp', 'obj_w', 'obj_h')):
        return False
    tcp = robot.eef() if tcp is None else np.asarray(tcp, float)
    if grasp is None:
        dxy = PATCH.get('grasp_dxy_objw') or [0.0, 0.0]
        grasp = PER['grasp'] + np.array([float(dxy[0]) * PER['obj_w'], float(dxy[1]) * PER['obj_w'],
                                        (PATCH['grasp_dz_objh'] + RSHAPE['dz']) * PER['obj_h']])
    tol = float(tol if tol is not None else _reapproach_tol())
    return float(np.linalg.norm(_xy(tcp) - _xy(grasp))) <= tol


def _clear_reapproach_if_aligned(reason='aligned'):
    if not EP.get('need_reapproach'):
        return False
    if _tcp_near_grasp():
        EP['need_reapproach'] = False
        EP['reapproach_reason'] = None
        with nonfatal('reapproach_cleared_log'):
            log(t=T[0], reapproach_cleared=True, reason=reason, goal_target=dict(GOAL_TARGET))
        return True
    return False


def _mark_reapproach_if_needed(old_grasp, reason):
    """After keypoint refresh: if TCP is not near the NEW grasp (material XY shift or new target), block close until re-approach.
    General geometry only — no object names. Does not force reapproach while already holding."""
    if EP.get('closed'):
        return False
    new_g = PER.get('grasp')
    if new_g is None:
        return False
    near = _tcp_near_grasp(grasp=new_g)
    shifted = False
    if old_grasp is not None:
        shifted = float(np.linalg.norm(_xy(new_g) - _xy(old_grasp))) >= _reapproach_tol()
    # Block close whenever TCP is not aligned to the refreshed target (covers identity change and large shift).
    if near and not shifted:
        return False
    if near:
        return False  # already over the new target
    EP['need_reapproach'] = True
    EP['reapproach_reason'] = str(reason or 'target_refresh')
    EP['reapproach_rev'] = int(GOAL_TARGET.get('rev', 0))
    if EP.get('force_next_intent') == 'close_gripper':
        EP['force_next_intent'] = 'move_above_object'
    with nonfatal('need_reapproach_log'):
        log(t=T[0], need_reapproach=True, reason=EP['reapproach_reason'], rev=EP['reapproach_rev'],
            old_grasp=list(np.asarray(old_grasp, float).round(4)) if old_grasp is not None else None,
            new_grasp=list(np.asarray(new_g, float).round(4)),
            tcp=list(np.asarray(robot.eef(), float).round(4)), goal_target=dict(GOAL_TARGET))
    return True


def _motion_orientation(shape=None):
    shape = SHAPE if shape is None else shape
    def pick(pk, sk, default):
        return next((v for v in (PATCH.get(pk), RSHAPE.get(sk), shape.get(sk)) if v is not None), default)
    return [float(pick('yaw_deg', 'yaw', PER['yaw'])),
            float(pick('tilt_deg', 'tilt', 0)), float(pick('tilt_dir_deg', 'tdir', 0))]


def _cmd_yaw_deg():
    """Current commanded grasp yaw (episode patch overrides perception)."""
    if PATCH.get('yaw_deg') is not None:
        return float(PATCH['yaw_deg'])
    if RSHAPE.get('yaw') is not None:
        return float(RSHAPE['yaw'])
    return float(PER.get('yaw', 0.0))


def _global_front_relook(reason):
    """Force keypoint refresh from current agentview (front) after retreat — not wrist-only."""
    detail = {'reason': reason, 'matched': None, 'error': None}
    ns, _ = observation.namespace()
    ap = _relook_approach_context(reason)
    ctx = json.dumps({
        'reason': reason,
        'language_goal': rt.LANG,
        'local_deadend_global_relook': True,
        'gripper_xyz_m': [round(float(x), 3) for x in ns['tcp']],
        'last_result': ap.get('last_result'),
        'stall_hint': ap.get('stall_hint'),
        'current_approach_yaw_deg': ap.get('current_approach_yaw_deg'),
        'current_tilt_deg': ap.get('current_tilt_deg'),
        'current_tilt_dir_deg': ap.get('current_tilt_dir_deg'),
        'instruction': (
            'GLOBAL FRONT relook after local dead-end: arm retreated for a clear agentview of the table. '
            'Re-identify the language-goal object on the FRONT/left camera; do NOT lock a neighbour. '
            'Local wrist/near poses already had no useful solution — prefer a fresh global grasp lock. '
            + str(ap.get('instruction_extra') or '')
        ),
    })
    try:
        # Prefer consuming a normal relook slot when available; else escape_look budget.
        if P['relook'] < MAX_RELOOK:
            P['relook'] += 1
            phase = 'global_relook'
            p, rgb, dep = perceive(ctx, phase=phase)
            tag = f'global_relook{int(EP.get("local_deadend_global_relooks", 0)) + 1}'
        else:
            phase = 'escape_look'
            p, rgb, dep = perceive(ctx, phase=phase)
            EP['escape_looks'] = int(EP.get('escape_looks', 0)) + 1
            tag = f'global_escape{EP["escape_looks"]}'
        idf = identity_fields(p)
        ID_EVENTS.append({'where': 'local_deadend_global_relook', 'reason': reason, **idf})
        _commit_keypoints(p, dep, 'global_relook', held_now=bool(p.get('object_held_by_gripper')) and EP['closed'])
        matched = identity_ok(p)
        set_goal_id(matched, 'global_relook', idf)
        Image.fromarray(rgb).save(os.path.join(rt.ep_dir, f'{tag}.jpg'))
        # Also save combo (front|wrist) for slow-think stills
        with nonfatal('global_relook_combo_save'):
            both, _, _ = robot._combo_rgb()
            Image.fromarray(both).save(os.path.join(rt.ep_dir, f'{tag}_combo.jpg'))
        detail.update(matched=matched, phase=phase, tag=tag, identity=idf)
        return detail, dep
    except Exception as ex:
        warn('global_front_relook', ex, reason=reason)
        detail['error'] = str(ex)[:300]
        return detail, None


def _escape_look(reason):
    """One extra identity look that does NOT consume the normal relook budget (separate escape budget)."""
    if EP.get('escape_looks', 0) >= MAX_ESCAPE_LOOK:
        return 'escape_look_budget_exhausted'
    EP['escape_looks'] = int(EP.get('escape_looks', 0)) + 1
    ns, _ = observation.namespace()
    ap = _relook_approach_context(reason)
    ctx = json.dumps({'reason': reason, 'language_goal': rt.LANG, 'escape_look': EP['escape_looks'],
                      'gripper_xyz_m': [round(float(x), 3) for x in ns['tcp']],
                      **{k: ap[k] for k in ('last_result', 'stall_hint', 'current_approach_yaw_deg', 'current_tilt_deg', 'current_tilt_dir_deg')},
                      'instruction': 'ESCAPE VIEW after raise/reapproach: re-identify the language-goal object from a clearer height; set matched_language_goal honestly. '
                                     + ap['instruction_extra']})
    try:
        p, rgb, dep = perceive(ctx, phase='escape_look')
        idf = identity_fields(p)
        ID_EVENTS.append({'where': 'escape_look', 'reason': reason, **idf})
        _commit_keypoints(p, dep, 'escape_look', held_now=bool(p.get('object_held_by_gripper')) and EP['closed'])
        matched = identity_ok(p)
        set_goal_id(matched, 'escape_look', idf)
        Image.fromarray(rgb).save(os.path.join(rt.ep_dir, f'escape{EP["escape_looks"]}.jpg'))
        return 'escape_looked_matched' if matched else 'escape_looked_unmatched'
    except Exception as e:
        warn('escape_look', e, reason=reason)
        return f'escape_look_error:{e}'


def _relook_approach_context(reason):
    """Stall + current commanded approach for relook/escape prompts. Do not silently revert a working yaw."""
    try:
        yaw = float(_cmd_yaw_deg())
    except (TypeError, ValueError):
        yaw = float(PER.get('yaw') or 0.0)
    tilt = PATCH.get('tilt_deg')
    if tilt is None:
        tilt = RSHAPE.get('tilt')
    tdir = PATCH.get('tilt_dir_deg')
    if tdir is None:
        tdir = RSHAPE.get('tdir')
    last = str(EP.get('last_result') or 'none')
    stalled = last in ('stalled', 'timeout')
    return {
        'last_result': last,
        'stall_hint': (
            f'Descent/approach recently {last}; refresh grasp on the language-goal object and keep a workable approach.'
            if stalled else
            'Re-identify / refresh keypoints on the language-goal object.'
        ),
        'current_approach_yaw_deg': round(yaw, 1),
        'current_tilt_deg': None if tilt is None else round(float(tilt), 1),
        'current_tilt_dir_deg': None if tdir is None else round(float(tdir), 1),
        'why_asked': reason,
        'instruction_extra': (
            'Current commanded approach yaw/tilt is given above. '
            'Do NOT revert grasp_yaw_deg to an old/default approach (e.g. 0°) unless THIS image gives clear new evidence '
            'that a different approach is better (neighbour clearance, object long axis, finger alignment). '
            'If unsure, keep grasp_yaw_deg near current_approach_yaw_deg. '
            'Relook refreshes grasp/place pixels; a confirmed effective approach pose is retained by the controller '
            'unless you explicitly propose a different yaw with evidence.'
        ),
    }


def do_relook(reason):
    if P['relook'] >= MAX_RELOOK: return 'relook_budget_exhausted'
    P['relook'] += 1
    ns, _ = observation.namespace()
    ap = _relook_approach_context(reason)
    ctx = json.dumps({'reason': reason, 'language_goal': rt.LANG, 'gripper_closed_command': ns['closed_cmd'], 'fingers_opening_m': round(ns['opening'], 4),
                      'both_fingers_contact': ns['contact'], 'gripper_xyz_m': [round(float(x), 3) for x in ns['tcp']],
                      'last_result': ap['last_result'], 'stall_hint': ap['stall_hint'],
                      'current_approach_yaw_deg': ap['current_approach_yaw_deg'],
                      'current_tilt_deg': ap['current_tilt_deg'], 'current_tilt_dir_deg': ap['current_tilt_dir_deg'],
                      'instruction': 'Re-identify the language-goal object in THIS image; lock grasp/object_center ONLY on that object. '
                                     + ap['instruction_extra']})
    try:
        p, rgb, dep = perceive(ctx, phase='relook'); idf = identity_fields(p)
        ID_EVENTS.append({'where': 'library_relook', 'reason': reason, **idf})
        # if identity still fails/unsure and budget remains, one immediate identity-focused retry (counts as another relook)
        if not identity_ok(p) and P['relook'] < MAX_RELOOK:
            P['relook'] += 1
            ctx2 = json.dumps({'reason': 'library relook identity failed or unsure', 'language_goal': rt.LANG, 'previous': idf,
                               'last_result': ap['last_result'], 'current_approach_yaw_deg': ap['current_approach_yaw_deg'],
                               'current_tilt_deg': ap['current_tilt_deg'],
                               'instruction': 'Name the language-goal object from the instruction + image; do NOT lock a neighbour. '
                                              'Set matched_language_goal honestly. ' + ap['instruction_extra']})
            p, rgb, dep = perceive(ctx2, phase='relook_identity'); idf = identity_fields(p)
            ID_EVENTS.append({'where': 'library_relook_identity', 'reason': reason, **idf})
            Image.fromarray(rgb).save(os.path.join(rt.ep_dir, f'relook{P["relook"]}_id.jpg'))
        _commit_keypoints(p, dep, 'relook', held_now=bool(p.get('object_held_by_gripper')) and EP['closed'])
        matched = identity_ok(p)
        set_goal_id(matched, 'relook', idf)
        Image.fromarray(rgb).save(os.path.join(rt.ep_dir, f'relook{P["relook"]}.jpg'))
        return 'relooked_matched' if matched else 'relooked_unmatched'
    except Exception as e:
        warn('relook', e, reason=reason)
        return 'relook_error'


def start_identity_check():
    """Start-of-episode identity-checked perception (general library knowledge; not fallback-only)."""
    rt.per0, rt.RGB0, rt.DEP0 = perceive(phase='perceive_start')
    Image.fromarray(rt.RGB0).save(os.path.join(rt.ep_dir, 'start.jpg'))
    with nonfatal('start_wrist_save'):
        _wr0 = rt.obs.get('robot0_eye_in_hand_image')
        if _wr0 is not None:
            Image.fromarray(_wr0[::-1]).save(os.path.join(rt.ep_dir, 'start_wrist.jpg'))
    START_ID['attempts'].append({**identity_fields(rt.per0), 'phase': 'perceive_start'})
    ID_EVENTS.append({'where': 'start', **identity_fields(rt.per0)})
    # Always verify the locked grasp pixel against the language goal (overconfident wrong locks are the soup failure mode)
    try:
        vrec, vimg = verify_grasp_identity(rt.per0, rt.RGB0)
        Image.fromarray(vimg).save(os.path.join(rt.ep_dir, 'start_id_verify.jpg'))
        conf = vrec.get('is_language_goal_object')
        if conf is True or str(conf).lower() == 'true': confirmed = True
        elif conf is False or str(conf).lower() == 'false': confirmed = False
        else: confirmed = None
        START_ID['verify'] = {'confirmed': confirmed, 'marked_object_what': vrec.get('marked_object_what'),
                              'language_goal_where_if_different': vrec.get('language_goal_where_if_different'),
                              'reason': vrec.get('reason')}
        ID_EVENTS.append({'where': 'start_verify', **START_ID['verify']})
    except Exception as ex:
        warn('start_verify', ex)
        confirmed = None; START_ID['verify'] = {'confirmed': None, 'error': str(ex)[:200]}
    need_relook = (not identity_ok(rt.per0)) or (confirmed is not True)
    if need_relook:
        # one forced relook before any motion; still no hard-coded appearance — language goal + current image only
        START_ID['forced_relook'] = True
        ctx = json.dumps({'reason': 'start identity check failed, unsure, or grasp-mark verify rejected',
                          'language_goal': rt.LANG,
                          'previous': identity_fields(rt.per0),
                          'verify': START_ID.get('verify'),
                          'instruction': 'Name the language-goal object in THIS image and lock grasp/object_center ONLY on that object; set matched_language_goal true only if sure. Do not lock a neighbour can/bottle/box.'})
        rt.per0, rt.RGB0, rt.DEP0 = perceive(ctx, phase='perceive_start_relook')
        Image.fromarray(rt.RGB0).save(os.path.join(rt.ep_dir, 'start_id_relook.jpg'))
        START_ID['attempts'].append({**identity_fields(rt.per0), 'phase': 'perceive_start_relook'})
        ID_EVENTS.append({'where': 'start_relook', **identity_fields(rt.per0)})
        # one more mark-verify after relook (still before motion)
        try:
            vrec2, vimg2 = verify_grasp_identity(rt.per0, rt.RGB0, phase='perceive_start_verify2')
            Image.fromarray(vimg2).save(os.path.join(rt.ep_dir, 'start_id_verify2.jpg'))
            conf2 = vrec2.get('is_language_goal_object')
            confirmed2 = True if conf2 is True or str(conf2).lower() == 'true' else (False if conf2 is False or str(conf2).lower() == 'false' else None)
            START_ID['verify_after_relook'] = {'confirmed': confirmed2, 'marked_object_what': vrec2.get('marked_object_what'),
                                              'language_goal_where_if_different': vrec2.get('language_goal_where_if_different'),
                                              'reason': vrec2.get('reason')}
            ID_EVENTS.append({'where': 'start_verify_after_relook', **START_ID['verify_after_relook']})
        except Exception as ex:
            warn('start_verify_after_relook', ex)
            START_ID['verify_after_relook'] = {'confirmed': None, 'error': str(ex)[:200]}
    START_ID['final'] = identity_fields(rt.per0)
    _commit_keypoints(rt.per0, rt.DEP0, 'perceive_start_relook' if START_ID.get('forced_relook') else 'episode_start')
    # Gate closes on start identity: matched_language_goal True AND mark-verify confirmed True. Unsure/false => not confirmed.
    _vfinal = START_ID.get('verify_after_relook') if START_ID.get('forced_relook') else START_ID.get('verify')
    _vconf = (_vfinal or {}).get('confirmed')
    _start_ok = identity_ok(rt.per0) and (_vconf is True)
    set_goal_id(_start_ok, 'start', {'matched': START_ID['final'].get('matched_language_goal'), 'verify_confirmed': _vconf,
                                      'forced_relook': START_ID.get('forced_relook')})
    START_ID['goal_id_confirmed'] = _start_ok
    ID_EVENTS.append({'where': 'start_goal_id_gate', 'confirmed': _start_ok, 'matched': START_ID['final'].get('matched_language_goal'),
                      'verify_confirmed': _vconf})
