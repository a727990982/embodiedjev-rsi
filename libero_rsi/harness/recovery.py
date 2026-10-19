"""Stall feedback, push-aside, local dead-end escape and identity-deadlock escape."""

import numpy as np

from . import assist, grasping, observation, planning, robot, vision
from .config import LOCAL_DEADEND_N, LOCAL_LOOP_FAMILY, MAX_ESCAPE, MAX_LOCAL_DEADEND_GLOBAL, MAX_PUSH_ASIDE, MAX_PUSH_ASIDE_PER_APPROACH, NEIGHBOUR_STALL_ISSUES, STALL_FB_ACTIONS
from .state import CUR, EP, GOAL_ID, GOAL_TARGET, ID_EVENTS, P, PATCH, PER, T, log, rt


def _relative_yaw_delta_deg(ns, approach_changes=None):
    """Sign from observed finger contact asymmetry; else alternate. Magnitude is a relative step, not a scene angle."""
    l = bool(ns.get('left_contact')) if ns else False
    r = bool(ns.get('right_contact')) if ns else False
    if l and not r:
        sign = 1.0
    elif r and not l:
        sign = -1.0
    else:
        count = EP.get('approach_changes', 0) if approach_changes is None else approach_changes
        sign = 1.0 if (count % 2 == 0) else -1.0
    # Relative quarter-turn/2 step about vertical; not the lucky scene-specific 90° from one success.
    return sign * 45.0


def _approach_yaw_plan(ns, approach_changes):
    """Read-only commanded yaw calculation shared by description and executor."""
    delta = _relative_yaw_delta_deg(ns, approach_changes)
    current = vision._cmd_yaw_deg()
    target = float(np.clip(current + delta, -90.0, 90.0))
    if abs(target - current) < 1.0:
        target = float(np.clip(current - delta, -90.0, 90.0))
    return current, delta, target


def _clear_reapproach_delta(ns):
    """Read-only offset rule; these values adjust an estimate, not an object."""
    left, right = bool(ns.get('left_contact')), bool(ns.get('right_contact'))
    if left and not right: return .4, 0.0
    if right and not left: return -.4, 0.0
    tcp = np.asarray(ns.get('tcp', robot.eef()), float)
    objc = np.asarray(ns.get('objc', PER.get('objc', tcp)), float)
    delta = objc[:2] - tcp[:2]
    norm = float(np.linalg.norm(delta)) + 1e-9
    return .25 * delta[0] / norm, .25 * delta[1] / norm


def _raise_tcp(scale=2.0, max_steps=40, ori_tol=None):
    """Lift gripper in place (open or closed as currently commanded)."""
    up_z = max(scale * float(PER.get('obj_h', 0.05)), 0.06)
    robot.servo(robot.eef() + np.array([0.0, 0.0, up_z]), -1.0 if not EP['closed'] else 1.0, 1.0, max_steps=max_steps, tol=0.01, ori_tol=ori_tol)
    return up_z


def _nudge_grasp_xy(dx_objw, dy_objw):
    """Episode-only lateral grasp offset in object-width units (no hard-coded metres)."""
    cur = list(PATCH.get('grasp_dxy_objw') or [0.0, 0.0])
    cur[0] = float(np.clip(cur[0] + float(dx_objw), -1.5, 1.5))
    cur[1] = float(np.clip(cur[1] + float(dy_objw), -1.5, 1.5))
    PATCH['grasp_dxy_objw'] = cur
    return cur


def _record_local_stall_family(action):
    """Append local-loop stall action; wipe hist on non-loop actions."""
    act = str(action or '')
    hist = list(EP.get('local_fb_family_hist') or [])
    if act in LOCAL_LOOP_FAMILY:
        hist.append(act)
        # keep a short window
        EP['local_fb_family_hist'] = hist[-12:]
    else:
        EP['local_fb_family_hist'] = []
    return list(EP.get('local_fb_family_hist') or [])


def _local_deadend_counts():
    hist = list(EP.get('local_fb_family_hist') or [])
    n_push = int(EP.get('push_aside_n', 0))
    n_push_ap = int(EP.get('push_aside_approach_n', 0))
    n_nb_issue = 0
    for a in hist:
        if a == 'push_aside':
            n_nb_issue += 1
    # consecutive trailing local-loop actions
    consec = 0
    for a in reversed(hist):
        if a in LOCAL_LOOP_FAMILY:
            consec += 1
        else:
            break
    return {
        'local_fb_family_hist': hist,
        'consec_local_loop': consec,
        'push_aside_n': n_push,
        'push_aside_approach_n': n_push_ap,
        'push_aside_in_hist': sum(1 for a in hist if a == 'push_aside'),
        'last_stall_issue': EP.get('last_stall_issue'),
        'grasp_secured_once': bool(EP.get('grasp_secured_once')),
        'local_deadend_n': int(EP.get('local_deadend_n', 0)),
        'local_deadend_global_relooks': int(EP.get('local_deadend_global_relooks', 0)),
    }


def _local_deadend_should_trigger(pending_action=None, check_grasp=True):
    """True when local view/actions have no useful solution — force global relook + sync slow.
    General (no object-name hacks). Skips if grasp already secured or global budget exhausted."""
    if check_grasp and (EP.get('grasp_secured_once') or grasping._grasp_secured_now()):
        return False, None
    if int(EP.get('local_deadend_global_relooks', 0)) >= MAX_LOCAL_DEADEND_GLOBAL:
        return False, 'global_relook_budget_exhausted'
    c = _local_deadend_counts()
    pending = str(pending_action or '')
    # Cap: another push would exceed episode or per-approach limit
    if pending == 'push_aside' or c['push_aside_n'] >= MAX_PUSH_ASIDE or c['push_aside_approach_n'] >= MAX_PUSH_ASIDE_PER_APPROACH:
        if c['push_aside_n'] >= MAX_PUSH_ASIDE:
            return True, f'push_aside_episode_cap:{c["push_aside_n"]}>={MAX_PUSH_ASIDE}'
        if c['push_aside_approach_n'] >= MAX_PUSH_ASIDE_PER_APPROACH and pending == 'push_aside':
            return True, f'push_aside_approach_cap:{c["push_aside_approach_n"]}>={MAX_PUSH_ASIDE_PER_APPROACH}'
        if pending == 'push_aside' and c['push_aside_n'] >= MAX_PUSH_ASIDE:
            return True, f'push_aside_episode_cap:{c["push_aside_n"]}'
    # N consecutive local-loop stall interventions without progress
    if c['consec_local_loop'] >= LOCAL_DEADEND_N:
        return True, f'consec_local_loop:{c["consec_local_loop"]}>={LOCAL_DEADEND_N}'
    # Same neighbour issue + multiple push_aside already this episode while still looping
    if (str(EP.get('last_stall_issue') or '') == 'blocked_by_neighbour'
            and c['push_aside_n'] >= max(1, MAX_PUSH_ASIDE) and c['consec_local_loop'] >= 2):
        return True, f'neighbour_push_exhausted:push={c["push_aside_n"]},loop={c["consec_local_loop"]}'
    # Slow/recovery keeps proposing push while already pushed enough
    if pending == 'push_aside' and c['push_aside_n'] >= max(1, MAX_PUSH_ASIDE - 1) and c['consec_local_loop'] >= 2:
        return True, f'repeated_push_proposal:push={c["push_aside_n"]},loop={c["consec_local_loop"]}'
    return False, None


def _retreat_for_global_view(scale=4.5):
    """Back away upward so the fixed front camera sees the scene (not wrist-stuck pose)."""
    if EP.get('closed') and not (bool(PER.get('seen_held')) or EP.get('grasp_confirm') or EP.get('grasp_secured_once')):
        EP['closed'] = False
        robot.grip(-1, 12)
    raised = _raise_tcp(scale, max_steps=50)
    # Extra lift toward a safer clearance above current grasp estimate
    tcp = robot.eef()
    oh = float(PER.get('obj_h', 0.05) or 0.05)
    grasp = PER.get('grasp')
    tgt = tcp.copy()
    tgt[2] = max(float(tcp[2]), float((tcp if grasp is None else grasp)[2]) + max(3.0 * oh, 0.12))
    robot.servo(tgt, -1.0 if not EP['closed'] else 1.0, 1.0, max_steps=40, tol=0.015)
    return raised


def _clear_stale_local_stall_state():
    """After global relook, drop stale neighbour/local-loop priors so we do not immediately re-push."""
    EP['last_stall_issue'] = None
    EP['local_fb_family_hist'] = []
    EP['push_aside_approach_n'] = 0
    # Keep episode push_aside_n (cap still applies). clear_reapproach_n kept as evidence but not required.

def _local_deadend_global_relook_sync_slow(reason, ns=None):
    """Hard rule: abort local loop — retreat, global front relook, SYNCHRONOUS slow-think replan, then resume.
    Uses the Codex _sync_slow path. Jev does not choose the next intent until this returns."""
    EP['local_deadend_n'] = int(EP.get('local_deadend_n', 0)) + 1
    counts = _local_deadend_counts()
    detail = {
        'local_deadend_global_relook': True,
        'reason': reason,
        'counts': counts,
        'retreat': None,
        'relook': None,
        'slow_result': None,
        'slow_detail': None,
    }
    EP['last_local_deadend_reason'] = reason
    # 1) Retreat for a clear front view
    detail['retreat'] = {'raised_m_scale': _retreat_for_global_view(4.5)}
    # 2) Global front relook (agentview keypoints)
    rel, dep = vision._global_front_relook(reason)
    detail['relook'] = rel
    _clear_stale_local_stall_state()
    EP['local_deadend_global_relooks'] = int(EP.get('local_deadend_global_relooks', 0)) + 1
    EP['need_reapproach'] = True
    EP['reapproach_reason'] = f'local_deadend_global_relook:{reason}'
    EP['reapproach_rev'] = int(GOAL_TARGET.get('rev', 0))
    # 3) SYNCHRONOUS slow-think replan — hold here; do not return to Jev until applied
    intent = 'local_deadend_global_relook'
    result = f'deadend:{reason}'
    crit = f'local_deadend_global_relook:{reason}'
    EP['last_stall_issue'] = None  # cleared; slow should replan from global stills
    # _sync_slow reports Codex failures as 'sync_error:...'; anything raised here is a motion/state error.
    sr, so = planning._sync_slow(intent, result, crit=crit)
    detail['slow_result'] = sr
    detail['slow_detail'] = {
        'issue': (so or {}).get('issue') if isinstance(so, dict) else None,
        'intervene': (so or {}).get('intervene') if isinstance(so, dict) else None,
        'plan_done': (so or {}).get('plan_done') if isinstance(so, dict) else None,
        'crit': (so or {}).get('crit') if isinstance(so, dict) else crit,
        'mode': (so or {}).get('mode') if isinstance(so, dict) else None,
    }
    if sr == 'ok':
        assist._consume_help_budget(crit)
    # If sync slow unavailable (cap/disabled), at least force re-approach from above
    if sr in ('slow_cap', 'slow_cap_drop', None) or (isinstance(sr, str) and sr.startswith('sync_error')):
        if not EP.get('force_next_intent'):
            EP['force_next_intent'] = 'move_above_object'
        # Soft fallback: raise_redescend once without push
        ns_fb, _ = observation.namespace()
        _apply_stall_feedback(ns_fb, f'{crit}:fallback_raise', action='raise_redescend')
    if not EP.get('force_next_intent'):
        EP['force_next_intent'] = 'move_above_object'
    CUR.update(fb=f"DEADEND→GLOBAL+SYNC_SLOW:{reason}"[:170], fb_t=T[0],
               slow=f"deadend_global:{detail.get('slow_result')}"[:160], slow_t=T[0])
    # detail already includes local_deadend_global_relook=True — do not pass it twice
    log(t=T[0], counts=counts, goal_id_confirmed=vision.goal_id_confirmed(), goal_target=dict(GOAL_TARGET),
        **{k: v for k, v in detail.items() if k != 'counts'})
    ID_EVENTS.append({'where': 'local_deadend_global_relook', **{k: detail.get(k) for k in
        ('reason', 'slow_result', 'relook')}, 'counts': counts})
    return detail


def _neighbour_block_hint(ns):
    """Geometry / issue signals that a neighbour is blocking approach (no object-name hardcodes)."""
    ns = ns or {}
    l = bool(ns.get('left_contact')); r = bool(ns.get('right_contact')); both = bool(ns.get('contact'))
    hit = bool(ns.get('descent_hit'))
    issue = str(EP.get('last_stall_issue') or '')
    if hit or (l != r and not both):
        return True
    if issue in NEIGHBOUR_STALL_ISSUES:
        return True
    if int(EP.get('clear_reapproach_n', 0)) >= 1 and not both:
        return True
    return False


def _gripper_open_axis_xy():
    """Unit XY along gripper opening (perpendicular to finger-close / yaw axis). Relative geometry only."""
    yaw = np.deg2rad(float(vision._cmd_yaw_deg()))
    v = np.array([-np.sin(yaw), np.cos(yaw)], float)
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def _push_dir_xy_from_ns(ns):
    """Prefer away-from-contacting-pad; else away from grasp centre; else gripper open axis."""
    ns = ns or {}
    l = bool(ns.get('left_contact')); r = bool(ns.get('right_contact'))
    open_ax = _gripper_open_axis_xy()
    if l and not r:
        return open_ax
    if r and not l:
        return -open_ax
    tcp = np.asarray(ns.get('tcp', robot.eef()), float)
    objc = ns.get('objc', PER.get('objc'))
    if objc is None:
        return open_ax
    delta = tcp[:2] - np.asarray(objc, float)[:2]
    if float(np.linalg.norm(delta)) > 1e-4:
        return delta / (float(np.linalg.norm(delta)) + 1e-9)
    return open_ax


def _mark_need_reapproach_after_push(reason='after_push_aside'):
    """Invalidate close until TCP re-aligns to grasp (reuse reapproach-before-close gate)."""
    EP['need_reapproach'] = True
    EP['reapproach_reason'] = str(reason or 'after_push_aside')
    EP['reapproach_rev'] = int(GOAL_TARGET.get('rev', 0))
    if EP.get('force_next_intent') == 'close_gripper':
        EP['force_next_intent'] = 'move_above_object'


def _choose_stall_feedback(ns):
    """Pick ONE general feedback action from observations. Not yaw-only; not scene object names.
    Neighbour-block path: clear_reapproach once, then prefer push_aside (identity-gated) over yaw churn."""
    l = bool(ns.get('left_contact')) if ns else False
    r = bool(ns.get('right_contact')) if ns else False
    both = bool(ns.get('contact')) if ns else False
    hit = bool(ns.get('descent_hit')) if ns else False
    empty = bool(ns.get('empty_close')) if ns else False
    last = str((ns or {}).get('last_result') or EP.get('last_result') or '')
    n = int(EP.get('stall_feedbacks', 0))
    n_clear = int(EP.get('clear_reapproach_n', 0))
    id_ok = bool(vision.goal_id_confirmed())
    issue = str(EP.get('last_stall_issue') or '')
    neighbour_issue = issue in NEIGHBOUR_STALL_ISSUES

    # Path / finger hit unilaterally → clear once; if clear already used + identity OK → push aside
    # (unless push_aside episode/approach cap — then raise instead; deadend path will global-relook).
    push_capped = (int(EP.get('push_aside_n', 0)) >= MAX_PUSH_ASIDE
                   or int(EP.get('push_aside_approach_n', 0)) >= MAX_PUSH_ASIDE_PER_APPROACH)
    if hit or (l != r and not both):
        if n_clear >= 1 and id_ok and not push_capped:
            return 'push_aside'
        if push_capped and n_clear >= 1:
            return 'raise_redescend'
        return 'clear_reapproach'

    # Explicit neighbour-block from slow-think, or clear exhausted without bilateral contact.
    # tip_push / object_pushed: clear once first; blocked_by_neighbour or post-clear → push.
    if id_ok and not both and not empty and (neighbour_issue or n_clear >= 1):
        if (n_clear >= 1 or issue == 'blocked_by_neighbour') and not push_capped:
            return 'push_aside'
        if push_capped:
            return 'raise_redescend'
        return 'clear_reapproach'

    # Closed on nothing → raise / re-center before another close.
    if empty:
        return 'raise_redescend' if (n % 2 == 0) else 'lateral_recenter'
    # Stalled near target with no bilateral contact → rotate through multi-action menu.
    if last in ('stalled', 'timeout') and not both:
        menu = ('raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact')
        return menu[n % len(menu)]
    # Default progress stall: raise then redescend; later tries change orientation.
    menu = ('raise_redescend', 'change_approach_yaw', 'lateral_recenter', 'confirm_contact')
    return menu[n % len(menu)]


def _apply_relative_approach_change(ns, reason):
    """Attempt upward motion with a new yaw; require position and orientation convergence."""
    EP['approach_changes'] = int(EP.get('approach_changes', 0)) + 1
    cur, delta, new_yaw = _approach_yaw_plan(ns or {}, EP['approach_changes'])
    PATCH['yaw_deg'] = new_yaw
    # Shallower fingertip target (positive dz = higher grasp) — observed stall often needs less depth.
    PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.15, -0.5, 0.5))
    vision._note_patch_mutation('approach_change', {'reason': reason, 'yaw_to': new_yaw})
    _raise_tcp(2.0, ori_tol=.08)
    motion = EP.get('last_motion') or {}
    measured = {'motion_result': motion.get('result'), 'orientation': motion.get('orientation')}
    log(t=T[0], approach_change=True, reason=reason, yaw_from=round(cur, 1), yaw_delta=round(delta, 1),
        yaw_to=round(new_yaw, 1), grasp_dz_objh=PATCH['grasp_dz_objh'], approach_changes=EP['approach_changes'], **measured)
    return {'yaw_from': cur, 'yaw_delta': delta, 'yaw_to': new_yaw, 'grasp_dz_objh': PATCH['grasp_dz_objh'], **measured}


def _execute_push_aside(ns, reason='stall', plan_step=None, dep=None):
    """Identity-gated open-finger lateral push of a neighbour blocker, then re-approach gate.
    Uses Astra blocker_px when provided; else gripper-open-axis / contact / away-from-grasp heuristic.
    Does NOT hard-code object names or fixed scene yaw. Episode/approach caps prevent horizon burn."""
    detail = {'pushed': False, 'reason': reason}
    if int(EP.get('push_aside_n', 0)) >= MAX_PUSH_ASIDE:
        detail['error'] = f'push_aside_episode_cap:{EP.get("push_aside_n")}>={MAX_PUSH_ASIDE}'
        _raise_tcp(2.0)
        EP['force_next_intent'] = 'move_above_object'
        return detail
    if int(EP.get('push_aside_approach_n', 0)) >= MAX_PUSH_ASIDE_PER_APPROACH:
        detail['error'] = f'push_aside_approach_cap:{EP.get("push_aside_approach_n")}>={MAX_PUSH_ASIDE_PER_APPROACH}'
        _raise_tcp(2.0)
        EP['force_next_intent'] = 'move_above_object'
        return detail
    id_ok = bool(vision.goal_id_confirmed())
    if not id_ok:
        detail['error'] = 'push_aside_refused_identity'
        _raise_tcp(2.0)
        EP['force_next_intent'] = 'move_above_object'
        return detail
    # Do not shove while holding a confirmed / verified grasp; skip neighbour push.
    if EP.get('grasp_confirm') or (EP.get('grasp_secured_once') and EP.get('closed')):
        detail['error'] = 'push_aside_skipped_grasp_secured'
        detail['grasp_confirm'] = bool(EP.get('grasp_confirm'))
        EP['force_next_intent'] = 'lift' if 'lift' in {i['name'] for i in rt.LIB.get('intents', [])} else 'move_above_object'
        return detail
    holding_verified = bool(EP.get('closed')) and bool(PER.get('seen_held')) and id_ok
    if EP.get('closed') and not holding_verified:
        detail['error'] = 'push_aside_refused_unverified_hold'
        EP['closed'] = False; robot.grip(-1, 15); EP['released'] = True
        _raise_tcp(2.0)
        EP['force_next_intent'] = 'move_above_object'
        return detail
    if holding_verified:
        detail['error'] = 'push_aside_skipped_holding_goal'
        EP['force_next_intent'] = 'lift' if 'lift' in {i['name'] for i in rt.LIB.get('intents', [])} else 'move_above_object'
        return detail

    # Keep grip OPEN during neighbour push (safer than closing on a non-goal).
    if EP.get('closed'):
        EP['closed'] = False; robot.grip(-1, 15)
    else:
        robot.grip(-1, 8)

    EP['push_aside_n'] = int(EP.get('push_aside_n', 0)) + 1
    EP['push_aside_approach_n'] = int(EP.get('push_aside_approach_n', 0)) + 1
    detail['push_aside_n'] = EP['push_aside_n']
    detail['push_aside_approach_n'] = EP['push_aside_approach_n']
    ow = float(PER.get('obj_w', 0.03) or 0.03)
    oh = float(PER.get('obj_h', 0.05) or 0.05)
    dist_objw = 1.0
    st = plan_step or {}
    try:
        dist_objw = float(np.clip(float(st.get('distance_objw', 1.0)), 0.5, 2.0))
    except (TypeError, ValueError):
        dist_objw = 1.0

    used_astra = False
    v = None
    start = None
    if st.get('blocker_px') is not None and dep is not None:
        try:
            blk = robot.bp(st['blocker_px'], dep)
            dn = st.get('direction', 'away_from_target')
            if dn == 'away_from_target':
                vv = blk - PER['objc']
            else:
                cv = {'toward_image_left': [-1, 0, 0], 'toward_image_right': [1, 0, 0],
                      'toward_camera': [0, -1, 0], 'away_from_camera': [0, 1, 0]}.get(dn)
                vv = (rt.RM @ np.array(cv, float)) if cv is not None else (blk - PER['objc'])
            vv = np.asarray(vv, float); vv[2] = 0
            v = vv / (np.linalg.norm(vv) + 1e-9)
            zc = max(float(blk[2]) - 0.5 * oh, float(PER['grasp'][2]) - 0.3 * oh)
            start = blk - v * (1.3 * ow); start[2] = zc
            used_astra = True
            detail['blocker_mode'] = 'astra_px'
            detail['direction'] = dn
        except (KeyError, TypeError, ValueError, IndexError) as ex:
            detail['astra_blocker_error'] = str(ex)
            used_astra = False

    if not used_astra:
        v2 = _push_dir_xy_from_ns(ns)
        v = np.array([float(v2[0]), float(v2[1]), 0.0])
        tcp = np.asarray(ns.get('tcp', robot.eef()), float)
        objc = np.asarray(ns.get('objc', PER.get('objc', tcp)), float)
        # Start just outside the grasp footprint along push dir (away from goal centre).
        start = objc.copy(); start[:2] = objc[:2] + v[:2] * (0.7 * ow)
        start[2] = float(PER.get('grasp', tcp)[2]) if PER.get('grasp') is not None else float(tcp[2])
        detail['blocker_mode'] = 'heuristic_open_axis'
        detail['dir_xy'] = [round(float(v[0]), 4), round(float(v[1]), 4)]

    L = dist_objw * ow
    g_open = -1.0
    # Approach from above, descend open, shove laterally, raise clear.
    robot.servo(start + np.array([0.0, 0.0, 2.0 * oh]), g_open, 1.0, max_steps=40, tol=0.01)
    robot.servo(start, g_open, 0.5, max_steps=40, tol=0.01)
    r = robot.servo(start + v * (L + 1.0 * ow), g_open, 0.4, max_steps=50, tol=0.015)
    robot.servo(robot.eef() + np.array([0.0, 0.0, 2.0 * oh]), g_open, 1.0, max_steps=40, tol=0.01)
    detail['servo'] = r
    detail['pushed'] = True
    detail['distance_objw'] = dist_objw

    # Always reopen (should already be open) and force re-approach before any close.
    EP['closed'] = False; robot.grip(-1, 10)
    _mark_need_reapproach_after_push(f'after_push_aside:{reason}')
    EP['force_next_intent'] = 'move_above_object'
    detail['need_reapproach'] = True
    # Shallow + small recenter toward goal after clearing neighbour.
    PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.1, -0.5, 0.5))
    detail['grasp_dz_objh'] = PATCH['grasp_dz_objh']
    return detail


def _apply_stall_feedback(ns, reason, action=None, plan_step=None, dep=None):
    """Multi-action stall feedback (raise/redescend, lateral re-center, relative yaw, confirm contact, clear/reapproach, push_aside).
    Observation-driven; no hard-coded object names or fixed scene angles.
    Optional plan_step/dep: Astra push_aside with blocker_px uses pixel geometry; else open-axis heuristic.
    Local-deadend: when local loop has no solution, abort and run sync global-relook + slow-think (holds Jev)."""
    ns = ns or {}
    if action is not None and action not in STALL_FB_ACTIONS:
        action = None
    action = action or _choose_stall_feedback(ns)
    # Progress bookkeeping
    grasping._note_grasp_progress(ns)
    # Confirmed grasp: never undo with neighbour/local-loop stall feedback — go lift/carry.
    if EP.get('grasp_confirm') or (EP.get('grasp_secured_once') and EP.get('closed') and grasping._grasp_secured_now(ns)):
        lift_name = 'lift' if 'lift' in {i['name'] for i in rt.LIB.get('intents', [])} else 'move_above_object'
        EP['force_next_intent'] = lift_name
        return {
            'action': 'skip_secured_grasp',
            'reason': reason,
            'stall_feedback_n': int(EP.get('stall_feedbacks', 0)),
            'grasp_confirm': True,
            'force_next': lift_name,
            'pending_suppressed': action,
        }
    # Hard local-deadend gate (skip while already inside deadend recovery to avoid recursion)
    if not EP.get('in_local_deadend'):
        fire, why = _local_deadend_should_trigger(pending_action=action)
        if fire:
            EP['in_local_deadend'] = True
            try:
                dd = _local_deadend_global_relook_sync_slow(why or 'local_deadend', ns=ns)
            finally:
                EP['in_local_deadend'] = False
            return {
                'action': 'local_deadend_global_relook',
                'reason': reason,
                'stall_feedback_n': int(EP.get('stall_feedbacks', 0)),
                'local_deadend_global_relook': True,
                'deadend_reason': why,
                'deadend_detail': {
                    'slow_result': (dd or {}).get('slow_result'),
                    'relook_matched': ((dd or {}).get('relook') or {}).get('matched'),
                },
            }
    EP['stall_feedbacks'] = int(EP.get('stall_feedbacks', 0)) + 1
    before_recovery = {'tcp_xyz_m': robot.eef().tolist(), 'commanded_orientation_deg': vision._motion_orientation()}
    detail = {'action': action, 'reason': reason, 'stall_feedback_n': EP['stall_feedbacks']}
    l = bool(ns.get('left_contact')); r = bool(ns.get('right_contact')); both = bool(ns.get('contact'))
    grip = -1.0 if not EP['closed'] else 1.0
    ow = float(PER.get('obj_w', 0.03) or 0.03)

    if action == 'raise_redescend':
        PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.2, -0.5, 0.5))
        detail['raised_m_scale'] = _raise_tcp(2.5)
        detail['grasp_dz_objh'] = PATCH['grasp_dz_objh']
        EP['force_next_intent'] = 'move_above_object'

    elif action == 'lateral_recenter':
        tcp = np.asarray(ns.get('tcp', robot.eef()), float)
        objc = np.asarray(ns.get('objc', PER.get('objc', tcp)), float)
        delta = objc[:2] - tcp[:2]
        nrm = float(np.linalg.norm(delta)) + 1e-9
        # Shift grasp estimate toward object centre (object-width units); raise then re-approach.
        dx, dy = float(delta[0] / nrm) * 0.35, float(delta[1] / nrm) * 0.35
        detail['grasp_dxy_objw'] = _nudge_grasp_xy(dx, dy)
        PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.1, -0.5, 0.5))
        tgt = robot.eef().copy(); tgt[:2] = tcp[:2] + np.array([dx, dy]) * ow; tgt[2] = tcp[2] + max(float(PER.get('obj_h', 0.05)), 0.04)
        robot.servo(tgt, grip, 1.0, max_steps=40, tol=0.01)
        detail['grasp_dz_objh'] = PATCH['grasp_dz_objh']
        EP['force_next_intent'] = 'move_above_object'

    elif action == 'change_approach_yaw':
        ch = _apply_relative_approach_change(ns, reason)
        detail.update(ch)
        EP['force_next_intent'] = 'move_above_object' if ch['motion_result'] == 'reached' else None

    elif action == 'confirm_contact':
        # Require bilateral contact before treating the grasp as secure / before lift.
        if both and not EP['closed']:
            if EP.get('need_reapproach') and not vision._tcp_near_grasp():
                EP['force_next_intent'] = 'move_above_object'
                detail['confirm'] = 'both_contact_but_need_reapproach'
            else:
                EP['force_next_intent'] = 'close_gripper'
                detail['confirm'] = 'both_contact_close'
        elif EP['closed'] and not both:
            EP['force_next_intent'] = 'reopen_and_retry' if 'reopen_and_retry' in {i['name'] for i in rt.LIB['intents']} else 'move_above_object'
            PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.15, -0.5, 0.5))
            _raise_tcp(2.0)
            detail['confirm'] = 'closed_no_bilateral_reopen'
        elif (l or r) and not both and not EP['closed']:
            # Partial contact: shallow + relative yaw, then close only after re-approach.
            ch = _apply_relative_approach_change(ns, reason + ':partial_contact')
            detail.update(ch)
            detail['confirm'] = 'partial_contact_reapproach'
            EP['force_next_intent'] = 'move_above_object' if ch['motion_result'] == 'reached' else None
        else:
            PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.15, -0.5, 0.5))
            _raise_tcp(2.0)
            detail['confirm'] = 'no_contact_raise'
            EP['force_next_intent'] = 'move_above_object'

    elif action == 'clear_reapproach':
        # Blocked / unilateral hit: raise clear + XY recenter only. Do NOT swing yaw here —
        # yaw changes only via change_approach_yaw (stall menu) or an explicit Codex plan step.
        EP['clear_reapproach_n'] = int(EP.get('clear_reapproach_n', 0)) + 1
        detail['clear_reapproach_n'] = EP['clear_reapproach_n']
        _raise_tcp(3.0)
        detail['grasp_dxy_objw'] = _nudge_grasp_xy(*_clear_reapproach_delta(ns))
        # Shallower fingertip target for vertical re-approach from clearance (no yaw mutation).
        PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.15, -0.5, 0.5))
        detail['grasp_dz_objh'] = PATCH['grasp_dz_objh']
        detail['yaw_preserved'] = vision._cmd_yaw_deg()
        EP['force_next_intent'] = 'move_above_object'
        detail['note'] = 'raised_cleared_recenter_no_yaw; yaw only via change_approach_yaw or Codex plan'

    elif action == 'push_aside':
        # Neighbour shove: identity-gated, open-finger lateral push, then force re-approach before close.
        detail.update(_execute_push_aside(ns, reason=reason, plan_step=plan_step, dep=dep))

    else:
        detail['error'] = f'unknown_action:{action}'
        _raise_tcp(2.0)
        EP['force_next_intent'] = 'move_above_object'

    CUR.update(fb=f"STALLFB:{action}", fb_t=T[0])
    vision._note_patch_mutation(f'stall_feedback:{action}', detail)
    if action in ('clear_reapproach', 'change_approach_yaw', 'raise_redescend', 'lateral_recenter'):
        EP['last_recovery_preparation'] = {
            'step': T[0], 'action': action, 'before': before_recovery,
            'after': {'tcp_xyz_m': robot.eef().tolist(), 'commanded_orientation_deg': vision._motion_orientation()},
            'orientation_attempt': detail.get('orientation'),
            'motion_result': detail.get('motion_result'),
            'status': 'preparation_attempted_descent_unverified',
        }
    _record_local_stall_family(action)
    log(t=T[0], stall_feedback=True, **detail, goal_target=dict(GOAL_TARGET),
        local_fb_family_hist=list(EP.get('local_fb_family_hist') or []))
    # Post-action deadend: N consecutive local loops just completed
    if (not EP.get('in_local_deadend')) and action in LOCAL_LOOP_FAMILY:
        fire2, why2 = _local_deadend_should_trigger(pending_action=None)
        if fire2:
            EP['in_local_deadend'] = True
            try:
                dd = _local_deadend_global_relook_sync_slow(why2 or 'post_local_loop', ns=ns)
            finally:
                EP['in_local_deadend'] = False
            detail = dict(detail)
            detail['local_deadend_global_relook'] = True
            detail['deadend_reason'] = why2
            detail['deadend_slow_result'] = (dd or {}).get('slow_result')
            return detail
    return detail


def _escape_identity_deadlock(reason):
    """Non-no-op escape when identity refuse + relook budget can no longer help: raise, relative approach change, escape look.
    Never leave the arm frozen selecting close_gripper until decision_cap."""
    EP['escapes'] = int(EP.get('escapes', 0)) + 1
    ns, _ = observation.namespace()
    detail = {'reason': reason, 'escape_n': EP['escapes'], 'relooks': P['relook'], 'close_refusals': GOAL_ID.get('close_refusals', 0)}
    if EP['escapes'] > MAX_ESCAPE:
        # Still move (raise) so we do not freeze, but stop burning Astra.
        up_z = max(2.0 * float(PER.get('obj_h', 0.05)), 0.06)
        robot.servo(robot.eef() + np.array([0.0, 0.0, up_z]), -1.0 if not EP['closed'] else 1.0, 1.0, max_steps=30, tol=0.01)
        EP['force_next_intent'] = 'retreat' if 'retreat' in {i['name'] for i in rt.LIB['intents']} else 'move_above_object'
        log(t=T[0], identity_escape='raise_only_cap', **detail)
        return 'identity_escape_cap'
    ch = _apply_stall_feedback(ns, reason)  # multi-action (not yaw-only)
    detail['stall_feedback'] = ch
    lr = vision._escape_look(reason)
    detail['escape_look'] = lr
    # Prefer re-approach from above after escape; never force another immediate close.
    if not EP.get('force_next_intent'):
        EP['force_next_intent'] = 'move_above_object'
    CUR.update(fb=f'ESCAPE:{reason}->{lr} fb->{ch.get("action")}', fb_t=T[0])
    log(t=T[0], identity_escape=True, **detail)
    ID_EVENTS.append({'where': 'identity_escape', **detail})
    return f'identity_escape:{lr}'


def push_aside(st, dep):
    """Episode-only neighbour shove. Identity-gated; prefers OPEN fingers; forces re-approach before close.
    Direction from blocker_px relative to target/camera — no fixed scene yaw / object names."""
    ns, _ = observation.namespace()
    det = _execute_push_aside(ns, reason='plan_push_aside', plan_step=st, dep=dep)
    if det.get('error'):
        return det['error']
    return str(det.get('servo') or 'pushed')
