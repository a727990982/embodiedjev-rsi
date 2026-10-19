"""Executes one library intent against the simulator, with safety gates."""

import numpy as np

import dsl_libero as dsl
import fine_steps as FS
import motion_feedback as MF

from . import assist, grasping, observation, pose, recovery, robot, vision
from .config import ASK_ASTRA_NAMES, CLOSE_NOT_READY, GRASP_FAILED, MAX_RELOOK, MOTION_RECOVERIES, TRANSPORT_INTENTS
from .state import DONE, EP, GOAL_ID, ID_EVENTS, P, PER, SHAPE, T, log, nonfatal, rt, warn


def execute(name, ns):
    it = {i['name']: i for i in rt.LIB['intents']}[name]
    act = it.get('action', 'move'); fing = it.get('fingers', 'keep')
    if act in ASK_ASTRA_NAMES or name in ASK_ASTRA_NAMES:
        return assist.do_ask_astra(f'jev_intent:{name}')
    if act == 'relook': return vision.do_relook('library rule chose relook')
    if act == 'finish':
        EP['todo_review_due'] = 'finish_requested'
        return 'finish' if DONE[0] else 'skip_finish'
    if name in TRANSPORT_INTENTS and not observation._transport_grasp_ready():
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        return f'{name}_not_executed:grasp_unconfirmed'
    blocked = observation._action_block_reason(name, it, ns)
    if blocked and blocked != 'close_refused_identity':
        log(t=T[0], action_not_executed=name, reason=blocked)
        return blocked
    if act == 'pose_menu': return pose._execute_pose_menu(it, ns)
    if act == 'fine_step': return _execute_fine_step(it, ns)
    if name in MOTION_RECOVERIES:
        if name == 'back_off':
            target = robot.eef().copy(); target[2] += max(.04, float(PER['obj_h']))
            r = robot.servo(target, 1 if EP['closed'] else -1, .5, max_steps=40)
            if r == 'reached': EP['blocked_motions'] = {}
            elif r in ('stalled', 'timeout'):
                EP.setdefault('blocked_motions', {})[name] = {
                    'target_xyz_m': target.tolist(), 'orientation_deg': vision._motion_orientation()}
            return r
        # Expired hold latches must not suppress recovery after actual contact/identity loss.
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        before_step = T[0]
        detail = recovery._apply_stall_feedback(ns, 'acting_motion_recovery', action=name)
        motion = EP.get('last_motion') or {}
        if motion.get('start_step', -1) >= before_step and motion.get('result') in ('stalled', 'timeout'):
            return motion['result']
        return 'recovery_prepared:' + str(detail.get('action')) + ':descent_unverified'
    if name == 'close_gripper' and EP['closed']:
        if observation._transport_grasp_ready():
            return 'already_closed'
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        return GRASP_FAILED
    if fing == 'close' and not EP['closed']:
        # HARD GATE: null/false/unsure goal identity ⇒ force relook, refuse close until matched. Uncertain ≠ proceed.
        # When relook budget is exhausted, DO NOT freeze: raise + relative approach change + escape look.
        if not vision.goal_id_confirmed():
            GOAL_ID['close_refusals'] = GOAL_ID.get('close_refusals', 0) + 1
            if P['relook'] >= MAX_RELOOK:
                rr = 'relook_budget_exhausted'
                ID_EVENTS.append({'where': 'close_refused', 'relook_result': rr, 'goal_id': dict(GOAL_ID),
                                  'identity': PER.get('identity'), 'escaped': True})
                log(t=T[0], close_refused=True, relook_result=rr, goal_id_confirmed=False,
                    identity=PER.get('identity'), escaped=True)
                return recovery._escape_identity_deadlock(f'close_refused:{rr}')
            rr = vision.do_relook('close refused: goal identity not confirmed (null/false/unsure ≠ proceed)')
            ID_EVENTS.append({'where': 'close_refused', 'relook_result': rr, 'goal_id': dict(GOAL_ID),
                              'identity': PER.get('identity')})
            log(t=T[0], close_refused=True, relook_result=rr, goal_id_confirmed=vision.goal_id_confirmed(),
                identity=PER.get('identity'))
            if not vision.goal_id_confirmed():
                # Budget just hit during this relook, or repeated unmatched closes → escape instead of silent refuse.
                if rr == 'relook_budget_exhausted' or P['relook'] >= MAX_RELOOK or GOAL_ID['close_refusals'] >= 3:
                    return recovery._escape_identity_deadlock(f'close_refused:{rr}')
                return f'close_refused_identity:{rr}'
            # relook matched — but a NEW/shifted target must be re-approached before close (B03)
        # HARD GATE: after target refresh, do not close while TCP is still over the old place.
        if EP.get('need_reapproach') and not vision._tcp_near_grasp():
            EP['force_next_intent'] = 'move_above_object'
            with nonfatal('close_blocked_log'):
                log(t=T[0], close_blocked_reapproach=True, reason=EP.get('reapproach_reason'),
                    rev=EP.get('reapproach_rev'), goal_id_confirmed=vision.goal_id_confirmed(),
                    tcp=list(np.asarray(robot.eef(), float).round(4)),
                    grasp=list(np.asarray(PER.get('grasp'), float).round(4)) if 'grasp' in PER else None)
            ID_EVENTS.append({'where': 'close_blocked_reapproach', 'reason': EP.get('reapproach_reason'),
                              'rev': EP.get('reapproach_rev'), 'goal_id': dict(GOAL_ID)})
            return 'close_blocked_need_reapproach'
        # Identity/relook may have moved the grasp estimate since the menu was built.
        current_ns, _ = observation.namespace()
        if not MF.grasp_position(robot.eef().tolist(), current_ns['grasp'].tolist(),
                                 current_ns['obj_w'], current_ns['obj_h'])['ready']:
            return CLOSE_NOT_READY
        vision._clear_reapproach_if_aligned('pre_close')
        EP['closed'] = True; robot.grip(1, 20); EP['n_close'] += 1; EP['released'] = False
        EP['tcp_at_close'] = robot.eef(); EP['opening_at_close'] = robot.opening()
        EP['closes'].append({'xyz': [round(float(x), 3) for x in robot.eef()], 'opening_after_close': round(robot.opening(), 4),
                             'both_contact': all(robot.contacts()), 'goal_id_confirmed': True})
        secured, _ = grasping._confirm_grasp_after_close()
        if not secured:
            EP['grasp_confirm'] = False
            EP['grasp_secured_once'] = False
            PER['seen_held'] = False
            return GRASP_FAILED
        return 'closed'
    if fing == 'open' and EP['closed']:
        EP['closed'] = False; robot.grip(-1, 15); EP['released'] = True; EP['just_released'] = True
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        EP['todo_review_due'] = 'gripper_released'
    if name == 'reopen_and_retry':
        return 'reopened'  # Opening is separate from approaching a potentially stale point.
    try: tgt = np.asarray(dsl.evaluate(it['refpoint'], ns), float).reshape(3)
    except Exception as e:
        warn('refpoint', e, intent=name)
        return 'refpoint_error'
    def ev(k, default):
        if k not in it: return default
        try: return float(dsl.evaluate(str(it[k]), ns))
        except Exception as e:
            warn('intent_shape', e, once=True, intent=name, field=k)
            return default
    SHAPE.update(tilt=ev('tilt_deg', None) or 0.0, tdir=ev('tilt_dir_deg', None) or 0.0, yaw=ev('yaw_deg', None))
    gain = ev('gain', 1.0)
    if np.linalg.norm(tgt - robot.eef()) < .006:
        vision._clear_reapproach_if_aligned('at_target')
        return 'at_target'
    r = robot.servo(tgt, 1 if EP['closed'] else -1, gain)
    if r in ('stalled', 'timeout'):
        EP.setdefault('blocked_motions', {})[name] = {
            'target_xyz_m': tgt.tolist(), 'orientation_deg': vision._motion_orientation()}
    elif r == 'reached':
        EP.get('blocked_motions', {}).pop(name, None)
        vision._clear_reapproach_if_aligned('reached')
    return r


def _remember_fine_step(rep):
    EP['last_fine_step'] = rep
    EP.setdefault('fine_step_events', []).append(dict(rep))


def _execute_fine_step(it, ns):
    """Servo one clamped world-axis correction. Does not change the gripper command."""
    name = it.get('name', 'fine_step')
    closed_before = bool(EP.get('closed'))
    tcp_now = np.asarray(robot.eef(), float).reshape(3)
    try:
        raw = np.asarray(dsl.evaluate(it['refpoint'], ns), float).reshape(3)
    except Exception as e:
        warn('refpoint', e, intent=name)
        rep = FS.fine_step_report(name, 'refpoint_error', tcp_now, np.zeros(3), closed_before)
        _remember_fine_step(rep)
        return 'refpoint_error'
    tgt, delta, status = FS.clamp_fine_target(tcp_now, raw)
    if status != 'ok':
        rep = FS.fine_step_report(name, 'rejected', tcp_now, np.zeros(3), closed_before)
        _remember_fine_step(rep)
        return 'rejected'
    if float(np.linalg.norm(delta)) < FS.FINE_STEP_TOL_M:
        r = 'at_target'
    else:
        # Keep the current approach yaw/tilt (PATCH / RSHAPE / perception). Do not invent a new orientation.
        try:
            gain = float(dsl.evaluate(str(it.get('gain', '1')), ns))
        except Exception as e:
            warn('intent_shape', e, once=True, intent=name, field='gain')
            gain = 1.0
        grip = 1 if closed_before else -1
        r = robot.servo(tgt, grip, gain, max_steps=FS.FINE_STEP_MAX_STEPS, tol=FS.FINE_STEP_TOL_M)
    if r in ('reached', 'at_target'):
        vision._clear_reapproach_if_aligned('fine_step')
    rep = FS.fine_step_report(name, r, robot.eef(), delta, bool(EP.get('closed')))
    _remember_fine_step(rep)
    return r
