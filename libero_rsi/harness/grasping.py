"""Grasp confirmation after close and place-phase forced release."""

import numpy as np

from . import robot, vision
from .config import FAIL_RESULTS, MAX_PLACE_LIFT_LOWER_CYCLES, MAX_PLACE_LOWER_STALL, OPEN_MAX, PLACE_XY_M
from .state import EP, PER, T, log, rt


def _place_xy_tol():
    return max(PLACE_XY_M, 0.5 * float(PER.get('obj_w', 0.03) or 0.03))


def _tcp_near_place_region(tcp=None, tol=None):
    """True when held-object XY (tcp - grasp + objc) is over the place region — same geometry as above_place."""
    try:
        tcp = robot.eef() if tcp is None else np.asarray(tcp, float)
        held_xy = vision._xy(tcp) - vision._xy(PER['grasp']) + vision._xy(PER['objc'])
        tol = float(tol if tol is not None else _place_xy_tol())
        return float(np.linalg.norm(held_xy - vision._xy(PER['place']))) <= tol
    except Exception:
        return False


def _note_place_phase(intent, result):
    """Track place-phase thrash: consecutive lower stalls + lift↔lower cycles."""
    if intent == 'carry_to_place' and result == 'reached':
        EP['place_carry_reached_once'] = True
        if int(EP.get('place_lower_stall_n', 0)) > 0:
            EP['place_cycle_armed'] = True
    if intent == 'lift' and int(EP.get('place_lower_stall_n', 0)) > 0 and EP.get('closed'):
        EP['place_cycle_armed'] = True
    if intent == 'lower_to_place':
        if EP.get('place_cycle_armed'):
            EP['place_lift_lower_cycles'] = int(EP.get('place_lift_lower_cycles', 0)) + 1
            EP['place_cycle_armed'] = False
        if result in FAIL_RESULTS:
            EP['place_lower_stall_n'] = int(EP.get('place_lower_stall_n', 0)) + 1
        elif result in ('reached', 'at_target'):
            EP['place_lower_stall_n'] = 0
    if intent == 'open_gripper' or (not EP.get('closed') and EP.get('released')):
        # Keep stall/cycle counters for result.json; only clear the armed cycle latch.
        EP['place_cycle_armed'] = False


def _should_force_place_release():
    """Force open when lower thrash is stuck over the place region (general; no object names)."""
    if EP.get('place_force_release') or not EP.get('closed') or EP.get('released'):
        return False, None
    stall_n = int(EP.get('place_lower_stall_n', 0))
    cycles = int(EP.get('place_lift_lower_cycles', 0))
    stall_hit = stall_n >= MAX_PLACE_LOWER_STALL
    cycle_hit = cycles >= MAX_PLACE_LIFT_LOWER_CYCLES
    if not (stall_hit or cycle_hit):
        return False, None
    # Must be over the place region (carried there / currently above).
    if not _tcp_near_place_region():
        return False, None
    if stall_hit and cycle_hit:
        reason = f'lower_stall_x{stall_n}+lift_lower_cycles_x{cycles}'
    elif stall_hit:
        reason = f'lower_stall_x{stall_n}'
    else:
        reason = f'lift_lower_cycles_x{cycles}'
    return True, reason


def _maybe_queue_place_force_release():
    ok, reason = _should_force_place_release()
    if not ok:
        return False
    EP['place_force_release'] = True
    EP['place_force_release_reason'] = reason
    EP['force_next_intent'] = 'open_gripper'
    try:
        log(t=T[0], place_force_release=True, reason=reason,
            place_lower_stall_n=int(EP.get('place_lower_stall_n', 0)),
            place_lift_lower_cycles=int(EP.get('place_lift_lower_cycles', 0)),
            place_carry_reached_once=bool(EP.get('place_carry_reached_once')),
            near_place=True,
            tcp=list(np.asarray(robot.eef(), float).round(4)),
            place=list(np.asarray(PER.get('place'), float).round(4)) if 'place' in PER else None)
    except Exception:
        pass
    return True


def _grasp_secured_now(ns=None):
    """Policy-visible: confirmed grasp, or bilateral contact + closed (+ optional seen_held)."""
    if EP.get('grasp_confirm') and EP.get('closed'):
        return True
    if EP.get('grasp_secured_once') and EP.get('closed'):
        return True
    ns = ns or {}
    both = bool(ns.get('contact'))
    if EP.get('closed') and both:
        return True
    if EP.get('closed') and bool(PER.get('seen_held')) and both:
        return True
    return False


def _confirm_grasp_after_close(ns=None):
    """After close_gripper: do not trust contact alone.
    Proprio (opening width + pads) + short closed lift probe (+ optional seen_held).
    Logs grasp_confirm=true/false. Secured → lift path; else reopen for recovery."""
    detail = {
        'method': 'proprio_lift',
        'opening_m': None,
        'opening_frac': None,
        'both_contact_pre': False,
        'both_contact_post': False,
        'lift_probe_m': 0.0,
        'seen_held_pre': bool(PER.get('seen_held')),
    }
    try:
        op = float(robot.opening())
        open_max = float(OPEN_MAX) if OPEN_MAX else 0.08
        l, r = robot.contacts()
        both = bool(l and r)
        detail['opening_m'] = round(op, 4)
        detail['opening_frac'] = round(op / max(open_max, 1e-6), 4)
        detail['both_contact_pre'] = both
        detail['left_contact'] = bool(l)
        detail['right_contact'] = bool(r)
        empty_like = (op < 0.15 * open_max) and not both
        detail['empty_like'] = empty_like
        if (not EP.get('closed')) or empty_like:
            detail['fail'] = 'empty_or_not_closed'
            EP['grasp_confirm'] = False
            EP['grasp_confirm_detail'] = detail
            try:
                log(t=T[0], grasp_confirm=False, **detail)
            except Exception:
                pass
            # empty close → reopen so recovery can run
            try:
                EP['closed'] = False
                robot.grip(-1, 12)
                EP['released'] = True
            except Exception:
                EP['closed'] = False
            if not EP.get('force_next_intent'):
                EP['force_next_intent'] = 'move_above_object'
            return False, detail
        width_ok = op >= 0.12 * open_max
        if not (width_ok or both):
            detail['fail'] = 'no_width_no_bilateral'
            EP['grasp_confirm'] = False
            EP['grasp_confirm_detail'] = detail
            try:
                log(t=T[0], grasp_confirm=False, **detail)
            except Exception:
                pass
            try:
                EP['closed'] = False
                robot.grip(-1, 12)
                EP['released'] = True
            except Exception:
                EP['closed'] = False
            EP['force_next_intent'] = 'move_above_object'
            return False, detail
        # Short lift probe while commanding closed — drop/collapse ⇒ not secured
        oh = float(PER.get('obj_h', 0.05) or 0.05)
        lift_m = float(max(0.5 * oh, 0.025))
        detail['lift_probe_m'] = round(lift_m, 4)
        tcp0 = robot.eef().copy()
        try:
            robot.servo(tcp0 + np.array([0.0, 0.0, lift_m]), 1.0, 1.0, max_steps=30, tol=0.008)
        except Exception as ex:
            detail['lift_err'] = str(ex)[:120]
        op2 = float(robot.opening())
        l2, r2 = robot.contacts()
        both2 = bool(l2 and r2)
        detail['opening_m_post'] = round(op2, 4)
        detail['both_contact_post'] = both2
        detail['left_contact_post'] = bool(l2)
        detail['right_contact_post'] = bool(r2)
        still_width = op2 >= 0.10 * open_max
        collapsed = (op2 < 0.08 * open_max) and not both2
        visual_ok = bool(PER.get('seen_held'))
        still_hold = both2 or (still_width and (l2 or r2 or both))
        secured = (not collapsed) and still_hold and (both2 or still_width or visual_ok)
        if both2 and still_width:
            secured = True
        if collapsed:
            secured = False
            detail['fail'] = 'collapsed_after_lift'
        detail['visual_bonus'] = visual_ok
        EP['grasp_confirm'] = bool(secured)
        EP['grasp_confirm_detail'] = detail
        if secured:
            EP['grasp_secured_once'] = True
            PER['seen_held'] = True
            EP['local_fb_family_hist'] = []
            EP['push_aside_approach_n'] = 0
            lift_name = 'lift' if 'lift' in {i['name'] for i in rt.LIB.get('intents', [])} else 'move_above_object'
            EP['force_next_intent'] = lift_name
            detail['force_next'] = lift_name
        else:
            try:
                EP['closed'] = False
                robot.grip(-1, 15)
                EP['released'] = True
            except Exception:
                EP['closed'] = False
            EP['force_next_intent'] = 'move_above_object'
            detail['reopened'] = True
            detail['force_next'] = 'move_above_object'
            if 'fail' not in detail:
                detail['fail'] = 'lift_probe_unconfirmed'
        try:
            log(t=T[0], grasp_confirm=bool(secured), **{k: v for k, v in detail.items()})
        except Exception:
            pass
        return bool(secured), detail
    except Exception as ex:
        detail['error'] = str(ex)[:200]
        EP['grasp_confirm'] = False
        EP['grasp_confirm_detail'] = detail
        try:
            log(t=T[0], grasp_confirm=False, **detail)
        except Exception:
            pass
        return False, detail


def _note_grasp_progress(ns=None):
    if _grasp_secured_now(ns):
        EP['grasp_secured_once'] = True
        EP['local_fb_family_hist'] = []
        EP['push_aside_approach_n'] = 0
        return True
    return False
