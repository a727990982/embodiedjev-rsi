"""Policy namespace, Jev inputs, motion context and candidate gating."""

import os

import numpy as np
from PIL import Image

import dsl_libero as dsl
import fine_steps as FS
import jev_observation as JO
import motion_feedback as MF

from . import pose, recovery, robot, vision
from .config import (ASK_ASTRA_NAMES, CLOSE_NOT_READY, FINE_STEPS, MAX_RELOOK, MAX_SLOW, OPEN_MAX, RECOVERY_MODE, S,
                     TRANSPORT_INTENTS)
from .state import DONE, EP, P, PATCH, PER, RSHAPE, T, log, rt, warn


def namespace():
    l, r = robot.contacts(); c = l and r; EP['ever_contact'] |= c
    tcp = robot.eef()
    dxy = PATCH.get('grasp_dxy_objw') or [0.0, 0.0]
    g = PER['grasp'] + np.array([float(dxy[0]) * PER['obj_w'], float(dxy[1]) * PER['obj_w'],
                                 (PATCH['grasp_dz_objh'] + RSHAPE['dz']) * PER['obj_h']])
    ns = {'fine_unit': FS.FINE_UNIT_M, 'tcp': tcp, 'opening': robot.opening(), 'left_contact': l, 'right_contact': r, 'contact': c, 'ever_contact': EP['ever_contact'],
          'closed_cmd': EP['closed'], 'released': EP['released'], 'seen_held': PER['seen_held'],
          'grasp': g, 'objc': PER['objc'], 'place': PER['place'], 'obj_h': PER['obj_h'], 'obj_w': PER['obj_w'], 'clear': PER['clear'], 'open_max': OPEN_MAX,
          'tcp_at_close': EP['tcp_at_close'] if EP['tcp_at_close'] is not None else tcp,
          'opening_at_close': EP['opening_at_close'] if EP['opening_at_close'] is not None else robot.opening(),
          'n_same': EP['n_same'], 'step': EP['step'], 'n_close': EP['n_close'], 'n_relook': P['relook'], 'prev': EP['prev'],
          'last_result': EP.get('last_result', 'none'),
          'clear_reapproach_n': int(EP.get('clear_reapproach_n', 0)),
          'push_aside_n': int(EP.get('push_aside_n', 0)),
          'place_lower_stall_n': int(EP.get('place_lower_stall_n', 0)),
          'place_lift_lower_cycles': int(EP.get('place_lift_lower_cycles', 0)),
          'place_carry_reached_once': bool(EP.get('place_carry_reached_once')),
          'place_force_release': bool(EP.get('place_force_release'))}
    preds = {}
    for p in rt.LIB.get('predicates', []):
        try: preds[p['name']] = bool(dsl.evaluate(p['expr'], {**ns, **preds}))
        except Exception as ex:
            warn('predicate', ex, once=True, name=p['name'])
            preds[p['name']] = False
    ns.update(preds); return ns, preds


def facts(ns, preds):
    r3 = lambda v: [round(float(x), 3) for x in v]
    f = {'gripper_xyz_m': r3(ns['tcp']), 'fingers_opening_m': round(ns['opening'], 4), 'gripper_closed_command': ns['closed_cmd'],
         'left_finger_contact': ns['left_contact'], 'right_finger_contact': ns['right_contact'], 'both_fingers_contact': ns['contact'],
         'opening_right_after_last_close_m': round(ns['opening_at_close'], 4), 'gripper_xyz_at_last_close_m': r3(ns['tcp_at_close']),
         'times_closed': ns['n_close'], 'released_since_close': ns['released'], 'astra_says_object_held': ns['seen_held'],
         'grasp_point_estimate_m': r3(ns['grasp']), 'object_top_centre_estimate_m': r3(ns['objc']), 'place_point_estimate_m': r3(ns['place']),
         'object_height_estimate_m': round(ns['obj_h'], 3), 'previous_intent': ns['prev'], 'decisions_in_previous_intent': ns['n_same'], 'last_intent_result': ns.get('last_result', 'none')}
    f.update(preds); return f


def _current_observation():
    """Capture images and measured proprioception without stepping the simulator."""
    left, right = robot.contacts()
    proprio = {
        'gripper_xyz_m': rt.obs['robot0_eef_pos'].tolist(),
        'gripper_orientation_xyzw': rt.obs['robot0_eef_quat'].tolist(),
        'finger_joint_positions_m': rt.obs['robot0_gripper_qpos'].tolist(),
        'fingers_opening_m': robot.opening(), 'gripper_closed_command': bool(EP['closed']),
        'left_finger_contact': bool(left), 'right_finger_contact': bool(right),
    }
    frames = {'front': rt.obs['agentview_image'][::-1].copy()}
    if rt.obs.get('robot0_eye_in_hand_image') is not None:
        frames['wrist'] = rt.obs['robot0_eye_in_hand_image'][::-1].copy()
    return proprio, frames


def _jev_input(*, include_execution_history=False):
    proprio, frames = _current_observation()
    state, assets = JO.build_state(rt.LANG, EP['task_todo'], proprio, frames, T[0], DONE[0])
    state['motion_context'] = _motion_context()
    if pose._pose_menu_visible():
        state['motion_context']['pose_action_outcome'] = EP.get('pose_menu_note')
    if FINE_STEPS:
        state['motion_context'].update(FS.fine_step_fact(True, EP.get('last_fine_step')))
    state['startup_obstacle_advice'] = _startup_obstacle_advice()
    failed_motion = EP.get('last_failed_motion') or EP.get('last_motion', {})
    failed_image = EP.get('failed_motion_before_path') or EP.get('motion_before_path')
    if failed_image and failed_motion.get('result') in ('stalled', 'timeout'):
        import base64, uuid
        path = failed_image
        ref = str(uuid.uuid4())
        with open(path, 'rb') as before_file:
            encoded = base64.b64encode(before_file.read()).decode()
        assets[ref] = {'type': 'image', 'format': 'png', 'data_base64': encoded}
        state['motion_before_image'] = {'asset': ref, 'step': failed_motion['start_step'],
                                       'note': 'Historical front+wrist image before the failed motion.'}
    if include_execution_history:
        state['execution_history'] = JO.execution_history(EP['hist'], EP['result_hist'])
        start = EP.get('recovery_history_start', 0)
        target_history = JO.execution_history(EP['hist'][start:], EP['result_hist'][start:])
        state['grasp_experience'] = JO.grasp_experience(state, target_history)
    # Store the exact observation sent to Jev; no privileged diagnostic fields.
    EP['vision_serial'] += 1
    paths = {}
    for name, pixels in frames.items():
        path = os.path.join(rt.ep_dir, f'jev_{EP["vision_serial"]}_{name}.png')
        Image.fromarray(pixels).save(path)
        paths[name] = path
    log(t=T[0], jev_observation=True, state=state, frames=paths)
    return state, assets


def _motion_context():
    ns, _ = namespace()
    return {'current_robot_contacts': robot.robot_contacts(), 'last_motion': EP.get('last_motion'),
            'last_failed_motion': EP.get('last_failed_motion'),
            'recovery_validation': _recovery_context(),
            'recovery_control': RECOVERY_MODE,
            'close_position_check': MF.grasp_position(robot.eef().tolist(), ns['grasp'].tolist(), ns['obj_w'], ns['obj_h']),
            'blocked_intents': list(EP.get('blocked_motions', {})),
            'slow_help_remaining': max(0, MAX_SLOW - P['slow']),
            'reserved_motion_diagnosis_remaining': int(rt.FALLBACK and MAX_SLOW > 0 and not EP.get('motion_help_used'))}


def _save_approach_overlay(geometry):
    from PIL import ImageDraw
    im = Image.fromarray(rt.obs['agentview_image'][::-1].copy())
    draw = ImageDraw.Draw(im)
    previous = None
    for label, key, color in [('TCP', 'tcp_xyz_m', 'cyan'),
                               ('approach', 'approach_xyz_m', 'yellow'),
                               ('grasp', 'descent_xyz_m', 'red')]:
        pixel = MF.project(geometry[key], rt.CP, rt.RM, rt.F, S)
        if pixel is None: previous = None; continue
        x, y = pixel
        if previous is not None: draw.line([previous, (x, y)], fill=color, width=2)
        draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
        draw.text((x+5, y+5), label, fill=color)
        previous = (x, y)
    draw.text((5, 5), 'Proposed TCP route only; NOT full hand/arm clearance', fill='yellow')
    path = os.path.join(rt.ep_dir, 'start_approach_overlay.png')
    im.save(path)
    return path


def _startup_obstacle_advice():
    plan = EP.get('start_plan') or {}
    return {'obstacle_assessment': plan.get('obstacle_assessment'),
            'suggested_recovery': plan.get('plan', []), 'diagnosis': plan.get('diagnosis'),
            'note': 'Historical startup prediction, not current collision truth or a forced sequence. '
                    'Use current motion/contact feedback if the scene has changed.'}


def _approach_geometry():
    ns, _ = namespace()
    grasp = ns['grasp'].tolist()
    return {'coordinate_frame': 'world meters; vision-derived estimates, not object ground truth',
            'tcp_xyz_m': robot.eef().tolist(), 'tcp_orientation_xyzw': rt.obs['robot0_eef_quat'].tolist(),
            'approach_xyz_m': [grasp[0], grasp[1], grasp[2] + 2 * ns['obj_h']],
            'descent_xyz_m': grasp, 'place_xyz_m': ns['place'].tolist(),
            'commanded_yaw_tilt_direction_deg': vision._motion_orientation(),
            'maximum_finger_opening_m': OPEN_MAX, 'current_finger_opening_m': robot.opening(),
            'object_width_estimate_m': ns['obj_w'], 'object_height_estimate_m': ns['obj_h'],
            'current_robot_contacts': robot.robot_contacts(),
            'limitations': 'Finger opening is NOT total hand width. Palm/arm swept volume is not '
                          'collision-checked; occluded clearance must be reported as uncertain.'}


def _transport_grasp_ready():
    """Confirmed target grasp with current sensor support, not a historical latch."""
    if not (EP.get('closed') and EP.get('grasp_confirm') and vision.goal_id_confirmed()):
        return False
    left, right = robot.contacts()
    return bool((left and right) or ((left or right) and robot.opening() >= 0.10 * OPEN_MAX))


def _action_block_reason(name, it, ns):
    held = _transport_grasp_ready()
    if name == 'close_gripper' and not EP['closed']:
        if not vision.goal_id_confirmed(): return 'close_refused_identity'
        if not MF.grasp_position(robot.eef().tolist(), ns['grasp'].tolist(), ns['obj_w'], ns['obj_h'])['ready']:
            return CLOSE_NOT_READY
    if held and (it.get('fingers') == 'open' or name in ('clear_reapproach', 'change_approach_yaw')):
        if name != 'open_gripper': return 'motion_not_executed:preserve_confirmed_grasp'
        if not MF.release_position(robot.eef().tolist(), ns['place'].tolist(), ns['clear']):
            return 'release_not_executed:not_at_destination'
    if name in EP.get('blocked_motions', {}):
        if name == 'back_off':
            target = (robot.eef() + [0, 0, max(.04, float(PER['obj_h']))]).tolist()
        else:
            target = np.asarray(dsl.evaluate(it['refpoint'], ns), float).tolist()
        shape = {key: float(dsl.evaluate(str(it[field]), ns)) if field in it else None
                 for key, field in [('yaw', 'yaw_deg'), ('tilt', 'tilt_deg'), ('tdir', 'tilt_dir_deg')]}
        orientation = vision._motion_orientation() if name == 'back_off' else vision._motion_orientation(shape)
        if MF.same_motion(EP['blocked_motions'][name], target, orientation):
            return 'motion_not_executed:unchanged_failed_waypoint'
    return None


def _acting_menu(intents, ns=None):
    ns = ns if ns is not None else namespace()[0]
    ready = _transport_grasp_ready()
    return {name: intent for name, intent in intents.items()
            if (name != 'close_gripper' or vision.goal_id_confirmed())
            and (name not in TRANSPORT_INTENTS or ready)
            and not _action_block_reason(name, intent, ns)
            and (name not in ASK_ASTRA_NAMES or (rt.FALLBACK and P['slow'] < MAX_SLOW))
            and (name != 'relook' or P['relook'] < MAX_RELOOK)}


def _recovery_context():
    ns, _ = namespace()
    start = EP.get('recovery_history_start', 0)
    progress = MF.recovery_progress(
        EP.get('hist', [])[start:], EP.get('result_hist', [])[start:],
        EP.get('tcp_hist', [])[start:], ns['grasp'].tolist(),
        EP.get('recovery_contact_hist', [])[start:])
    preparation = EP.get('last_recovery_preparation')
    if preparation:
        progress['last_preparation'] = preparation
        progress['descent_tested_since_preparation'] = EP.get('last_descent_step', -1) >= preparation['step']
    return progress


def _candidate_execution(menu, ns, state):
    """Describe allowed candidates without stepping, mutating state or predicting physics."""
    descriptions = {}
    tcp = np.asarray(ns['tcp'], float)
    for name in menu:
        if name not in ('move_above_object', 'descend_to_grasp', 'clear_reapproach', 'change_approach_yaw'):
            continue
        it = menu[name]
        d = {'observation_step': state['observation_step'],
             'source': 'executor rules, current proprioception and visual estimates; no physics rollout',
             'program_steps': [], 'conditional_effects': [],
             'unverified_physics': ['Whether the motion reaches its target without obstruction or unintended contact.',
                                    'Target graspability and eventual task success.']}
        if name in ('move_above_object', 'descend_to_grasp'):
            # Custom same-name actions must not inherit an invented move description.
            if it.get('action', 'move') != 'move' or it.get('fingers') == 'close':
                continue
            finger = it.get('fingers', 'keep')
            d['program_steps'].append({'gripper_command': finger,
                'command_transition_needed': (finger == 'open' and EP['closed']) or (finger == 'close' and not EP['closed'])})
            try:
                target = np.asarray(dsl.evaluate(it['refpoint'], ns), float).reshape(3)
            except Exception as ex:
                warn('candidate_refpoint', ex, once=True, intent=name)
                d['program_steps'].append({'return': 'refpoint_error', 'servo_motion': False})
                d['conditional_effects'] = ['No movement target can be evaluated; this call cannot test descent.']
                descriptions[name] = d
                continue
            else:
                def ev(key):
                    if key not in it: return None
                    try: return float(dsl.evaluate(str(it[key]), ns))
                    except Exception as ex:
                        warn('candidate_shape', ex, once=True, intent=name, field=key)
                        return None
                shape = {'tilt': ev('tilt_deg') or 0.0, 'tdir': ev('tilt_dir_deg') or 0.0, 'yaw': ev('yaw_deg')}
                within = float(np.linalg.norm(target - tcp)) < .006
                d['program_steps'].append({'operation': 'position_check_then_optional_servo',
                    'commanded_tcp_target_m': target.tolist(), 'requested_delta_m': (target-tcp).tolist(),
                    'commanded_orientation_deg': vision._motion_orientation(shape), 'position_tolerance_m': .006,
                    'within_tolerance_at_observation': within,
                    'branch': 'At execution, return at_target if within tolerance; otherwise attempt servo. Opening fingers first may change the pose.'})
            if name == 'move_above_object':
                d['conditional_effects'] = ['Reaching the approach waypoint only prepares the next decision; no descent or grasp is tested.']
            else:
                d['conditional_effects'] = ['If motion is needed, this attempts a descent test; at_target performs no new motion. Closing is a separate choice, offered only if the existing identity and position guards pass afterward.']
        else:
            # execute clears stale grasp latches before entering this recovery path.
            # _note_grasp_progress may then recognize a closed bilateral contact.
            if EP['closed'] and bool(ns.get('contact')):
                followup = 'lift' if any(i['name'] == 'lift' for i in rt.LIB['intents']) else 'move_above_object'
                d['program_steps'] = [{'operation': 'existing_grasp_preservation_guard',
                                       'queued_followup': followup, 'local_preparation_skipped': True}]
                d['conditional_effects'] = ['The existing closed/bilateral-contact guard preserves the grip; it does not newly verify target identity.']
                descriptions[name] = d
                continue
            fire, reason = recovery._local_deadend_should_trigger(pending_action=name, check_grasp=False)
            if not EP.get('in_local_deadend') and fire:
                d['program_steps'] = [{'operation': 'existing_global_deadend_guard', 'reason': reason,
                    'steps': 'Retreat for a global view, relocalize, and request bounded diagnosis; the nominal local recovery is replaced.'}]
                d['conditional_effects'] = ['Subsequent recovery depends on the new observation and available help budget; no descent result is known.']
            else:
                scale = 3.0 if name == 'clear_reapproach' else 2.0
                raise_m = max(scale * float(PER.get('obj_h', .05)), .06)
                patch_z = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + .15, -.5, .5))
                if name == 'clear_reapproach':
                    delta = recovery._clear_reapproach_delta(ns)
                    old = PATCH.get('grasp_dxy_objw') or [0.0, 0.0]
                    offset = [float(np.clip(a+b, -1.5, 1.5)) for a,b in zip(old,delta)]
                    d['program_steps'] = [
                        {'operation': 'attempt_upward_servo', 'requested_raise_m': raise_m, 'gripper': 'keep current command'},
                        {'operation': 'update_visual_grasp_offsets', 'xy_object_width_units': offset,
                         'z_object_height_units': patch_z, 'commanded_yaw_deg_unchanged': vision._cmd_yaw_deg()}]
                else:
                    current, _, yaw = recovery._approach_yaw_plan(ns, int(EP.get('approach_changes', 0)) + 1)
                    d['program_steps'] = [
                        {'operation': 'set_approach_command', 'yaw_from_deg': current, 'yaw_to_deg': yaw,
                         'z_object_height_units': patch_z},
                        {'operation': 'attempt_upward_servo_with_new_orientation', 'requested_raise_m': raise_m,
                         'gripper': 'keep current command'}]
                d['program_steps'].append({'operation': 'queue_existing_followup', 'intent': 'move_above_object'})
                d['conditional_effects'] = ['If recovery is not interrupted, the executor queues approach; neither recovery nor approach tests descent.',
                                            'The existing post-recovery deadend guard can still request global relook/diagnosis.']
                d['unverified_physics'].append('Whether the hand actually reaches the commanded yaw; upward servo has no orientation-convergence requirement.')
            recovery_validation = state.get('motion_context', {}).get('recovery_validation', {})
            d['current_recovery_evidence'] = {k: recovery_validation[k] for k in
                ('preparations_without_descent', 'no_progress', 'descent_tested_since_preparation') if k in recovery_validation}
        descriptions[name] = d
    return descriptions
