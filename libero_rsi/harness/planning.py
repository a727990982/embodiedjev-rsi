"""Codex slow-think (sync and async jobs) and the episode-start plan.

Codex slow-think: async watch most of the time; SYNC pause at critical points.
Async results older than 1 completed intent are discarded. Never watches mid-episode video."""

import json
import os
import queue
import re
import threading
import time

import numpy as np
from PIL import Image

import dsl_libero as dsl
import jev_experience as JX
import task_todo as TODO

from . import observation, progress, recovery, robot, vision
from .config import (AUTO_RECOVERY_RULES, FAIL_RESULTS, HORIZON, MAX_PUSH_ASIDE, MAX_SLOW, NEIGHBOUR_STALL_ISSUES,
                     SLOW_MIN_GAP_S, SLOW_STALE_INTENTS, STAGE_FAIL_N)
from .state import (CUR, DONE, EP, FB_LESSONS, GOAL_TARGET, ID_EVENTS, P, PATCH, PER, SLOW_JOB, SLOW_JOB_SEQ,
                    SLOW_ORPHAN_COSTS, SLOW_ORPHAN_WAITERS, START_ID, T, log, nonfatal, rt, warn)


def _next_slow_job_id():
    SLOW_JOB_SEQ[0] += 1
    return int(SLOW_JOB_SEQ[0])


def _new_slow_result_box(job_id):
    return {'job_id': int(job_id), 'parsed': None, 'rec': None, 'err': None, 'done': False}


def _apply_episode_plan(p, dep, source='slow'):
    """Apply episode-only plan steps from Codex.
    Single shared goal: keypoint refresh overwrites PER+clears position PATCH; approach yaw/tilt preserved unless new yaw explicit; adjust/stall mutates PATCH only.
    On stall with identity confirmed: relook must run perception; never treat a log marker as a target refresh."""
    def f(d_, k, lo, hi, d):
        try: return float(np.clip(float(d_[k]), lo, hi))
        except (KeyError, TypeError, ValueError): return d
    # Confirmed grasp: never apply neighbour/local-loop undo plans; keep hold → lift.
    if (EP.get('grasp_confirm') or EP.get('grasp_secured_once')) and EP.get('closed'):
        p = dict(p or {})
        undo = {'push_aside', 'clear_reapproach', 'raise_redescend', 'lateral_recenter',
                'change_approach_yaw', 'confirm_contact'}
        p['plan'] = [st for st in (p.get('plan') or []) if st.get('step') not in undo]
        iss = str(p.get('issue') or p.get('failure_class') or '')
        if iss in NEIGHBOUR_STALL_ISSUES or iss in ('other', 'empty_grasp'):
            p['issue'] = 'ok'
            p['intervene'] = False
        lift_name = 'lift' if 'lift' in {i['name'] for i in rt.LIB.get('intents', [])} else 'move_above_object'
        EP['force_next_intent'] = lift_name
        if not p.get('plan'):
            with nonfatal('grasp_secured_plan_skip_log'):
                log(t=T[0], grasp_secured_plan_skip=True, source=source, force_next=lift_name,
                    grasp_confirm=bool(EP.get('grasp_confirm')))
            return ['skip_plan_grasp_secured']
    igo = p.get('is_goal_object')
    if igo is True or str(igo).lower() == 'true': igo_n = True
    elif igo is False or str(igo).lower() == 'false': igo_n = False
    else: igo_n = None
    issue = str(p.get('issue') or p.get('failure_class') or '')
    mlg = vision.identity_fields(p)['matched_language_goal'] if ('matched_language_goal' in p or 'grasp_px' in p) else None
    # Uncertain ≠ proceed for TARGET identity (matched_language_goal) and for HELD identity after close.
    # is_goal_object=null while open usually means "nothing held" — that alone does not clear GOAL_ID.
    # wrong_object / unmatched target → identity bad. empty_grasp/tip_push with confirmed target is STALL feedback, not id wipe.
    stall_issue = issue in ('empty_grasp', 'tip_push', 'blocked_by_neighbour', 'object_pushed_by_descent')
    if stall_issue:
        EP['last_stall_issue'] = issue
    held_bad = (igo_n is False) or issue in ('wrong_object',) or (igo_n is None and EP['closed'] and not stall_issue and mlg is not True)
    target_bad = (mlg is False) or (mlg is None and 'matched_language_goal' in p and not stall_issue)
    id_bad = held_bad or target_bad
    if id_bad:
        p['plan'] = [{'step': 'relook'}] + [st for st in (p.get('plan') or []) if st.get('step') not in ('adjust', 'raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach', 'push_aside')]
        vision._reset_patch(reason=f'{source}_id_bad')
        vision.set_goal_id(False, f'{source}_id_bad', {'is_goal_object': igo_n, 'matched_language_goal': mlg, 'issue': issue})
    elif stall_issue and mlg is True and AUTO_RECOVERY_RULES:
        # Keep goal identity; ensure plan has a non-relook feedback step.
        plan = list(p.get('plan') or [])
        stall_ok = ('raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach', 'push_aside', 'adjust')
        if not any(st.get('step') in stall_ok or st.get('step') == 'relook' for st in plan):
            # Neighbour block → prefer push_aside (blocker_px used if later present); else raise.
            if issue == 'blocked_by_neighbour' and vision.goal_id_confirmed():
                default_step = 'push_aside'
            elif issue == 'empty_grasp':
                default_step = 'raise_redescend'
            else:
                default_step = 'clear_reapproach' if issue in ('tip_push', 'object_pushed_by_descent') else 'raise_redescend'
            p['plan'] = [{'step': default_step}] + [st for st in plan if st.get('step') != 'relook']
    if _should_open_on_intervention(igo_n=igo_n, issue=issue):
        EP['closed'] = False; robot.grip(-1, 15); EP['released'] = True
    has_kp = any(k in p for k in ('grasp_px', 'object_center_px', 'place_px'))
    target_updated = False
    if has_kp:
        # A planning reply is not a perception result; refresh using the visual model.
        return [f"relook:{vision.do_relook(f'{source}:planner_keypoints')}"]
    done_steps = []
    stall_steps = {'raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach'}
    id_ok = vision.goal_id_confirmed()
    last = str(EP.get('last_result') or '')
    stallish = last in ('stalled', 'timeout') or issue in ('empty_grasp', 'tip_push', 'blocked_by_neighbour', 'object_pushed_by_descent', 'other')
    for st in (p.get('plan') or [])[:3]:
        k = st.get('step')
        if k == 'adjust':
            PATCH.update(grasp_dz_objh=f(st, 'grasp_dz_objh', -.5, .5, PATCH['grasp_dz_objh']), tilt_deg=f(st, 'tilt_deg', 0, 30, PATCH['tilt_deg']),
                         yaw_deg=f(st, 'yaw_deg', -90, 90, PATCH['yaw_deg']))
            td = f(st, 'tilt_dir_deg', -180, 180, None)
            if td is not None:
                th = np.deg2rad(td); dw = rt.RM @ np.array([np.cos(th), -np.sin(th), 0]); PATCH['tilt_dir_deg'] = float(np.degrees(np.arctan2(dw[1], dw[0])))
            vision._note_patch_mutation(f'{source}:adjust', dict(st))
            done_steps.append('adjust'); target_updated = True
        elif k in stall_steps:
            ns_fb, _ = observation.namespace()
            det = recovery._apply_stall_feedback(ns_fb, f'{source}:{k}', action=k)
            done_steps.append(f'{k}:{det.get("action")}'); target_updated = True
        elif k == 'push_aside':
            ns_fb, _ = observation.namespace()
            det = recovery._apply_stall_feedback(ns_fb, f'{source}:push_aside', action='push_aside', plan_step=st, dep=dep)
            done_steps.append(f"push_aside:{det.get('action')}:{det.get('error') or ('ok' if det.get('pushed') else 'noop')}")
            target_updated = True
        elif k == 'relook':
            # A real visual refresh replaces the old target. Remaining plan steps were
            # based on the old view, so return to Jev with a fresh namespace afterward.
            done_steps.append(f"relook:{vision.do_relook(f'{source}:plan')}")
            return done_steps
    # Stall + identity confirmed ⇒ shared target MUST change (raise/shift/relook-refresh). Empty relook-only forbidden.
    if id_ok and stallish and not target_updated and AUTO_RECOVERY_RULES:
        ns_fb, _ = observation.namespace()
        det = recovery._apply_stall_feedback(ns_fb, f'{source}:forced_target_update')
        done_steps.append(f'forced_stall_feedback:{det.get("action")}')
        target_updated = True
    return done_steps


def _run_episode_start_plan():
    """Great Wall: predictive global plan after start identity, before first approach.
    Astra via Codex writes the todo and start_plan.json before target localization.
    The todo owns task progress; phase_order remains historical planner advice."""
    import coding_agent_slow as CAS
    mode = 'codex'
    out = {'mode': mode, 'applied': False, 'plan_done': [], 'error': None}
    try:
        brief = JX.build_jev_experience_brief(rt.LIB, facts={}, ns={}, uses=EP.get('uses') or {}, dsl=dsl)
    except Exception as ex:
        warn('start_plan_brief', ex)
        brief = {}
    # Seed general start priors into the brief the planner sees (not task/brand hardcodes).
    try:
        priors = list((brief.get('lessons') or []))
        seed_ids = {L.get('id') for L in priors}
        seeds = [
            {
                'id': 'hp_start_predict_neighbour',
                'when': 'neighbour near language-goal grasp at episode start',
                'instead': 'plan push_aside or clear_reapproach before first approach',
                'status': 'hypothesis',
            },
            {
                'id': 'hp_start_place_then_retreat',
                'when': 'after place near a container or basket rim',
                'instead': 'plan retreat / lift clear of the rim before the next approach',
                'status': 'hypothesis',
            },
            {
                'id': 'hp_start_multi_object_cycles',
                'when': 'language goal names multiple objects or both / and',
                'instead': 'plan a full approach→grasp→lift→place→retreat cycle for each object',
                'status': 'hypothesis',
            },
        ]
        for s in reversed(seeds):
            if s['id'] in seed_ids:
                continue
            # Also skip near-duplicate neighbour seed by content
            if s['id'] == 'hp_start_predict_neighbour' and any(
                'push_aside' in str(L.get('instead', '')).lower()
                or 'neighbour' in str(L.get('when', '')).lower()
                for L in priors
            ):
                continue
            priors.insert(0, s)
            seed_ids.add(s['id'])
        brief = dict(brief)
        brief['lessons'] = priors[:8]
    except Exception as ex:
        warn('start_plan_seed_priors', ex)
    _vfinal = START_ID.get('verify_after_relook') if START_ID.get('forced_relook') else START_ID.get('verify')
    start_hints = {
        'goal_id_confirmed': bool(vision.goal_id_confirmed()),
        'identity_matched': (START_ID.get('final') or {}).get('matched_language_goal'),
        'verify_confirmed': (_vfinal or {}).get('confirmed'),
        'identity': START_ID.get('final'),
        'verify': _vfinal,
        'perception_named': {
            'goal': (rt.per0 or {}).get('goal_object_named'),
            'where': (rt.per0 or {}).get('goal_object_where'),
            'place': (rt.per0 or {}).get('place_region_named'),
            'place_where': (rt.per0 or {}).get('place_region_where'),
        },
        'approach_geometry': observation._approach_geometry(),
        'neighbour_prior': (
            'If another object sits close to the grasp target so the open gripper may collide '
            'before a bilateral close, prefer clear_reapproach or change_approach_yaw before approach. '
            'Clearance requires confirmed identity; if identity is unsure, relook first.'
        ),
    }
    # Annotate the proposed route so geometry can be related to the same front view.
    geometry, annotated = observation.astra_geometry('start_plan', route=start_hints['approach_geometry'])
    start_hints['robot_geometry'] = geometry
    images = []
    for name in ('start.jpg', 'start_wrist.jpg'):
        fp = os.path.join(rt.ep_dir, name)
        if os.path.isfile(fp):
            images.append(fp)
    images.extend(annotated or [observation._save_approach_overlay(start_hints['approach_geometry'])])
    parsed, rec = None, {}
    try:
        parsed, rec = CAS.harness_call_start_plan(
            lang=rt.LANG, experience_brief=brief, start_hints=start_hints,
            ep_dir=rt.ep_dir, image_paths=images, mode='codex')
        _account_slow_rec(rec)
        observation.remember_geometry_advice(parsed.get('robot_geometry_advice'), geometry, 'start_plan')
        proposed_plan = parsed
        if vision.identity_fields(parsed)['matched_language_goal'] is False:
            vision.set_goal_id(False, 'episode_start_plan_rejected', vision.identity_fields(parsed))
        parsed = CAS.gate_start_plan(parsed, vision.goal_id_confirmed())
        # Persist full artifact for RSI reflection
        art = {'plan': parsed, 'proposed_plan': proposed_plan, 'rec': rec, 'start_hints': start_hints, 'mode': mode, 'experience_brief': brief}
        with open(os.path.join(rt.ep_dir, 'start_plan.json'), 'w') as plan_file:
            json.dump(art, plan_file, indent=1, default=str)
        EP['task_todo'] = TODO.initialize(parsed.get('task_todo'), rt.LANG, T[0])
        progress._save_task_todo('start_plan')
        progress._retarget_todo()
        EP['start_plan'] = parsed
        EP['start_plan_rec'] = {k: rec.get(k) for k in ('backend', 'wall_s', 'cost_usd', 'model', 'effort', 'validate_ok', 'validate_errs') if k in (rec or {})}
        out['plan'] = parsed
        out['rec'] = EP['start_plan_rec']
        log(t=T[0], episode_start_plan=True, mode=mode, backend=(rec or {}).get('backend'),
            issue=(parsed or {}).get('issue'), intervene=(parsed or {}).get('intervene'),
            step1_must=(parsed or {}).get('step1_must'),
            predicted_risks=(parsed or {}).get('predicted_risks'),
            phase_order=(parsed or {}).get('phase_order'),
            plan=(parsed or {}).get('plan'))
        # Keep start recovery advice as a recorded proposal. The first acting
        # decision uses the newly localized todo target and current images.
        if not vision.goal_id_confirmed():
            EP['force_next_intent'] = 'relook'
        return out
    except Exception as ex:
        out['error'] = str(ex)[:400]
        with nonfatal('start_plan_error_record'):
            with open(os.path.join(rt.ep_dir, 'start_plan.json'), 'w') as plan_file:
                json.dump(out, plan_file, indent=1, default=str)
            log(t=T[0], episode_start_plan_error=True, mode=mode, err=out['error'])
        return out


def _stage_fail_grip_unchanged(consume=False):
    """General escalate: same phase failing (timeout/stall) while gripper open/closed cmd unchanged.
    Intent-name agnostic — covers one stuck intent OR oscillating intents that never change grip.
    If consume=True, arm debounce so we do not re-fire until STAGE_FAIL_N new failures accrue."""
    rh, gh = EP.get('result_hist') or [], EP.get('grip_hist') or []
    n = STAGE_FAIL_N
    if len(rh) < n or len(gh) < n:
        return None
    recent_r, recent_g = rh[-n:], gh[-n:]
    if not all(r in FAIL_RESULTS for r in recent_r):
        return None
    if len(set(bool(g) for g in recent_g)) != 1:
        return None
    if len(rh) - int(EP.get('stage_fail_mark') or 0) < n:
        return None
    if consume:
        EP['stage_fail_mark'] = len(rh)
    return 'stage_fail_grip_unchanged'


def _slow_critical(intent, result):
    """SYNC pause only here: after close, after open/release, no-progress, or stage-fail+grip-unchanged."""
    if result == 'closed': return 'after_close'
    if EP.pop('just_released', False): return 'after_release'
    # general: repeated timeout/stall with gripper cmd frozen (not intent-name-specific)
    sf = _stage_fail_grip_unchanged(consume=True)
    if sf: return sf
    h, th = EP['hist'], EP['tcp_hist']
    if len(h) >= 3 and h[-1] not in ('finish',) and len(th) >= 3:
        moved = float(np.linalg.norm(th[-1] - th[-3]))
        if len(set(h[-3:])) == 1 and moved < 0.01: return 'no_progress_same_x3'
        if len(set(h[-3:])) <= 2 and moved < 0.015: return 'no_progress_2_3'
    return None


def _slow_ctx(intent, result):
    ns, preds = observation.namespace()
    return json.dumps({'just_finished_intent': intent, 'intent_result': result, 'language_goal': rt.LANG,
                       'motion_context': observation._motion_context(),
                       'recent_intents': EP['hist'][-6:], 'recent_intent_results': EP.get('result_hist', [])[-6:],
                       'recent_gripper_closed': EP.get('grip_hist', [])[-6:],
                       'times_closed': EP['n_close'], 'close_attempts': EP['closes'][-3:],
                       'gripper_closed_command': ns['closed_cmd'], 'fingers_opening_m': round(ns['opening'], 4),
                       'both_fingers_contact': ns['contact'], 'left_finger_contact': ns['left_contact'],
                       'right_finger_contact': ns['right_contact'], 'astra_says_object_held': ns['seen_held'],
                       'gripper_xyz_m': [round(float(x), 3) for x in ns['tcp']],
                       'object_height_estimate_m': round(PER['obj_h'], 3),
                       'goal_id_confirmed': vision.goal_id_confirmed(),
                       'predicates': {k: bool(preds.get(k)) for k in ('descent_servo_stalled', 'descent_stalled', 'descent_hit', 'empty_close', 'at_grasp_relaxed')},
                       'stall_feedback_count': EP.get('stall_feedbacks', 0),
                       'push_aside_n': int(EP.get('push_aside_n', 0)),
                       'push_aside_approach_n': int(EP.get('push_aside_approach_n', 0)),
                       'clear_reapproach_n': int(EP.get('clear_reapproach_n', 0)),
                       'local_fb_family_hist': list(EP.get('local_fb_family_hist') or [])[-6:],
                       'local_deadend_n': int(EP.get('local_deadend_n', 0)),
                       'local_deadend_global_relook': bool(EP.get('in_local_deadend')) or (
                           'local_deadend' in str(intent) or 'local_deadend' in str(result)),
                       'max_push_aside': MAX_PUSH_ASIDE,
                       'grasp_confirm': bool(EP.get('grasp_confirm')),
                       'grasp_secured': bool(EP.get('grasp_secured_once') or EP.get('grasp_confirm')),
                       'feedback_menu': ['raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach', 'push_aside'],
                       'hint': (
                           'If grasp_secured/grasp_confirm is true: do NOT push_aside or retreat; prefer lift/carry (intervene false). '
                           'If identity confirmed and stalled/empty/no bilateral contact: choose a feedback_menu step, not relook-only. '
                           'If local_deadend_global_relook is true OR push_aside_n at/over max_push_aside: do NOT repeat push_aside; '
                           'prefer raise_redescend / lateral_recenter / change_approach_yaw / relook from the GLOBAL front view.'
                       )}, default=float)


def _finish_slow(parsed, rec, dep, both, intent, result, mode, crit, snap_dec):
    """Apply a finished Codex slow-think reply on the main thread (env-safe)."""
    if rec.get('robot_geometry_source'):
        observation.remember_geometry_advice(parsed.get('robot_geometry_advice'), rec['robot_geometry_source'], mode)
    _account_slow_rec(rec)
    issue = str(parsed.get('issue') or 'ok')
    _igo = parsed.get('is_goal_object')
    if _igo is True or str(_igo).lower() == 'true': _igo_n = True
    elif _igo is False or str(_igo).lower() == 'false': _igo_n = False
    else: _igo_n = None
    _mlg = vision.identity_fields(parsed)['matched_language_goal'] if 'matched_language_goal' in parsed else None
    # Held-object check only meaningful after a close (or when Codex flags wrong_object).
    # Before close, is_goal_object=null ("nothing held") must NOT clear the grasp-target GOAL_ID gate.
    stall_issue = issue in ('empty_grasp', 'tip_push')
    held_id_bad = (_igo_n is False) or (issue == 'wrong_object') or (_igo_n is None and EP['closed'] and not stall_issue and _mlg is not True)
    target_id_bad = (_mlg is False) or (_mlg is None and 'matched_language_goal' in parsed and not stall_issue)
    if held_id_bad or target_id_bad:
        if issue in ('ok', '', 'other'): issue = 'wrong_object'
        parsed['issue'] = issue
        parsed['intervene'] = True
        if not any(st.get('step') == 'relook' for st in (parsed.get('plan') or [])):
            parsed['plan'] = [{'step': 'relook'}] + list(parsed.get('plan') or [])
        vision.set_goal_id(False, 'slow_think_igo', {'is_goal_object': _igo_n, 'matched_language_goal': _mlg,
                                              'issue': issue, 'closed': EP['closed']})
    elif stall_issue:
        parsed['intervene'] = True
        plan = list(parsed.get('plan') or [])
        if AUTO_RECOVERY_RULES and not any(st.get('step') in ('raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach', 'push_aside', 'adjust', 'relook') for st in plan):
            if issue == 'empty_grasp':
                _def = 'raise_redescend'
            elif issue == 'blocked_by_neighbour' and vision.goal_id_confirmed():
                _def = 'push_aside'
            else:
                _def = 'clear_reapproach'
            parsed['plan'] = [{'step': _def}] + [st for st in plan if st.get('step') != 'relook']
    elif _igo_n is True and EP['closed']:
        vision.set_goal_id(True, 'slow_think_held_ok', {'is_goal_object': True})
    intervene = bool(parsed.get('intervene')) and issue != 'ok'
    les = str(parsed.get('lesson') or '').strip()
    if les and not any(ch.isdigit() for ch in les):
        FB_LESSONS.append({'failure_class': issue, 'diagnosis': re.sub(r'\d', '#', str(parsed.get('diagnosis', ''))),
                           'lesson': les, 'source': 'slow_think'})
    out = {'issue': issue, 'intervene': intervene, 'diagnosis': parsed.get('diagnosis'), 'lesson': les,
           'is_goal_object': parsed.get('is_goal_object'), 'goal_object_where': parsed.get('goal_object_where'),
           'held_or_between_fingers': parsed.get('held_or_between_fingers'), 'mode': mode, 'crit': crit,
           'snap_dec': snap_dec, 'apply_dec': rt.dec}
    if not intervene:
        Image.fromarray(both).save(os.path.join(rt.ep_dir, f'slow_ok_c{P["planning_calls"]}.jpg'))
        ID_EVENTS.append({'where': 'slow_think', 'intent': intent, 'result': result, **{k: out[k] for k in
            ('issue', 'intervene', 'is_goal_object', 'goal_object_where', 'held_or_between_fingers', 'diagnosis', 'mode', 'crit')}})
        return 'ok', out
    if P['slow'] >= MAX_SLOW and crit != 'reserved_motion_obstruction':
        out['intervene'] = False; return 'slow_cap_drop', out
    if crit != 'reserved_motion_obstruction': P['slow'] += 1
    done_steps = _apply_episode_plan(parsed, dep, source='slow')
    image_name = f'slow_motion_reserved_{T[0]}.jpg' if crit == 'reserved_motion_obstruction' else f'slow{P["slow"]}.jpg'
    Image.fromarray(both).save(os.path.join(rt.ep_dir, image_name))
    CUR.update(slow=f"{mode}/{crit or '-'}:{issue}->{done_steps}"[:160], slow_t=T[0])
    out.update(plan_done=done_steps, patch=dict(PATCH))
    ID_EVENTS.append({'where': 'slow_think', 'intent': intent, 'result': result, **{k: out.get(k) for k in
        ('issue', 'intervene', 'is_goal_object', 'goal_object_where', 'held_or_between_fingers', 'diagnosis', 'mode', 'crit')},
        'plan_done': done_steps})
    return 'intervened', out


def _prepare_slow_snapshot(ctx, crit, phase, both, image_tag):
    import coding_agent_slow as CAS
    ns, preds = observation.namespace()
    brief = JX.build_jev_experience_brief(
        rt.LIB, facts=observation.facts(ns, preds), ns=ns, uses=EP.get('uses') or {}, dsl=dsl,
        episode_lessons=FB_LESSONS, episode_start_plan=EP.get('start_plan'))
    # Unique image and detached policy context belong to this request.
    frame = os.path.join(rt.ep_dir, f'slow_input_{image_tag}.jpg')
    Image.fromarray(both).save(frame)
    geometry, annotated = observation.astra_geometry(image_tag)
    context = CAS.prepare_slow_context(
        lang=rt.LANG, trigger=str(crit or phase), slow_ctx_json=ctx,
        experience_brief=brief, patch=PATCH,
        goal_id={'confirmed': vision.goal_id_confirmed(), 'target': GOAL_TARGET},
        ep_dir=rt.ep_dir, keyframe_paths=[p for p in (EP.get('failed_motion_before_path') or EP.get('motion_before_path'), frame) if p] + annotated)
    context['robot_geometry'] = geometry
    context['motion_images_note'] = 'Optional failed-motion image first; current front+wrist composite next; annotated current robot-only projections last. Overlays are not future states or visibility masks.'
    context['task_todo'] = json.loads(json.dumps(EP.get('task_todo')))
    context['active_goal'] = TODO.active_item(context['task_todo']) if context['task_todo'] else None
    return {'t': T[0], 'coding_context': context}


def _slow_worker(job_id, result_box, snapshot):
    """Codex only: publish results and log events into this job's private box."""
    events = result_box.setdefault('events', [])
    try:
        import coding_agent_slow as CAS
        parsed, rec = CAS.call_monitor(snapshot['coding_context'], mode='codex')
        geom = snapshot['coding_context'].get('robot_geometry', {})
        rec = dict(rec, robot_geometry_source={k: geom[k] for k in ('observation_step', 'target_id', 'status') if k in geom})
        events.append(dict(t=snapshot['t'], coding_agent_slow=True, mode='codex', job_id=job_id,
            issue=parsed.get('issue'), intervene=parsed.get('intervene'),
            plan=parsed.get('plan'), backend=rec.get('backend')))
        result_box['parsed'], result_box['rec'] = parsed, rec
    except Exception as e:
        events.append(dict(t=snapshot['t'], coding_agent_slow_error=True,
                           mode='codex', job_id=job_id, err=str(e)[:300]))
        result_box['err'] = e
    finally:
        result_box['done'] = True


def _log_slow_result(box):
    # Called only after joining/checking completion, on the main thread.
    if box:
        for event in box.pop('events', []):
            log(**event)


def _account_slow_rec(rec):
    if not rec:
        return
    P['planning_calls'] += 1
    P['last_planning_wall'] = time.time()
    try:
        P['planning_cost'] += float(rec.get('cost_usd') or 0.0)
    except (AttributeError, TypeError, ValueError) as ex:
        warn('planning_cost', ex)


def _drain_orphan_slow_costs():
    while True:
        try:
            box, info = SLOW_ORPHAN_COSTS.get_nowait()
        except queue.Empty:
            break
        _log_slow_result(box)
        rec = (box or {}).get('rec')
        _account_slow_rec(rec)
        log(t=T[0], async_abandoned=True, **info, had_rec=bool(rec))


def _clear_slow_job():
    """Drop current ownership pointers. Does not revoke a worker's private result_box (already detached on abandon)."""
    SLOW_JOB.update(thread=None, snap_dec=None, dep=None, both=None, intent=None, result=None,
                    mode=None, crit=None, out=None, err=None, rec=None, parsed=None,
                    job_id=None, result_box=None, target_rev=None, closed=None,
                    n_close=None, goal_confirmed=None)


def _abandon_async_slow(reason='abandon'):
    """Invalidate current async ownership so a late worker cannot apply. Orphan thread may still finish for cost."""
    th = SLOW_JOB.get('thread')
    box = SLOW_JOB.get('result_box')
    jid = SLOW_JOB.get('job_id')
    if th is None and box is None:
        return False
    # Revoke ownership FIRST so any concurrent poll/finish cannot apply this job.
    SLOW_JOB['job_id'] = None
    meta = {'snap_dec': SLOW_JOB.get('snap_dec'), 'intent': SLOW_JOB.get('intent'), 'crit': SLOW_JOB.get('crit'),
            'abandoned_job_id': jid, 'reason': reason}
    _clear_slow_job()
    if th is not None and th.is_alive():
        def _orphan_join(t=th, result_box=box, info=meta):
            t.join()
            SLOW_ORPHAN_COSTS.put((result_box, info))
        waiter = threading.Thread(target=_orphan_join, daemon=True)
        SLOW_ORPHAN_WAITERS.append(waiter)
        waiter.start()
    elif box is not None:
        # Already finished: account cost, never apply
        _log_slow_result(box)
        _account_slow_rec(box.get('rec'))
        with nonfatal('async_discarded_log'):
            log(t=T[0], async_discarded_finished=True, **meta, issue=(box.get('parsed') or {}).get('issue'))
    return True


def _finalize_slow_jobs():
    """Join bounded CLI calls and record their results without moving the robot."""
    th = SLOW_JOB.get('thread')
    if th is not None:
        th.join()  # Codex calls enforce their own timeout.
        box = SLOW_JOB.get('result_box') or {}
        _log_slow_result(box)
        _account_slow_rec(box.get('rec'))
        log(t=T[0], async_final_discard=True, job_id=SLOW_JOB.get('job_id'),
            error=str(box['err']) if box.get('err') else None)
        _clear_slow_job()
    for waiter in SLOW_ORPHAN_WAITERS:
        waiter.join()
    SLOW_ORPHAN_WAITERS.clear()
    _drain_orphan_slow_costs()


def _poll_async_slow(wait=False, timeout=None):
    """If the CURRENT owned async watch finished: apply if <= SLOW_STALE_INTENTS late, else discard.
    Results from a revoked/abandoned job_id are never applied (B01)."""
    _drain_orphan_slow_costs()
    th = SLOW_JOB.get('thread')
    box = SLOW_JOB.get('result_box')
    job_id = SLOW_JOB.get('job_id')
    if th is None or box is None or job_id is None:
        return None
    # Ownership check: private box must still belong to the current job.
    if box.get('job_id') != job_id:
        _clear_slow_job()
        return 'async_ownership_mismatch', {'job_id': job_id, 'box_job_id': box.get('job_id')}
    if wait:
        th.join(timeout=timeout)
    if th.is_alive():
        return None if not wait else ('async_timeout', None)
    # Re-check ownership after join (abandon may have raced).
    if SLOW_JOB.get('job_id') != job_id or SLOW_JOB.get('result_box') is not box:
        return 'async_abandoned_race', {'job_id': job_id}
    _log_slow_result(box)
    snap_dec = SLOW_JOB['snap_dec']; stale = rt.dec - snap_dec
    err = box.get('err')
    intent, result = SLOW_JOB['intent'], SLOW_JOB['result']
    both, dep = SLOW_JOB['both'], SLOW_JOB['dep']
    mode, crit = SLOW_JOB['mode'], SLOW_JOB['crit']
    parsed, rec = box.get('parsed'), box.get('rec')
    if err is not None:
        _clear_slow_job()
        return f'async_error:{err}', None
    if (SLOW_JOB['target_rev'] != GOAL_TARGET['rev']
            or SLOW_JOB['closed'] != bool(EP.get('closed'))
            or SLOW_JOB['n_close'] != int(EP.get('n_close', 0))
            or SLOW_JOB['goal_confirmed'] != vision.goal_id_confirmed()):
        _account_slow_rec(rec)
        _clear_slow_job()
        return 'async_state_changed_discard', {'snap_dec': snap_dec, 'apply_dec': rt.dec, 'job_id': job_id}
    if stale > SLOW_STALE_INTENTS:
        _account_slow_rec(rec)
        _clear_slow_job()
        return 'async_stale_discard', {'snap_dec': snap_dec, 'apply_dec': rt.dec, 'stale_intents': stale,
                                      'issue': (parsed or {}).get('issue'), 'job_id': job_id}
    _clear_slow_job()
    return _finish_slow(parsed, rec, dep, both, intent, result, mode or 'async', crit, snap_dec)


def _launch_async_slow(intent, result, crit=None, replace=False):
    """Start an async watch owned by a fresh job_id. If replace=True, abandon any in-flight job first."""
    _drain_orphan_slow_costs()
    if SLOW_JOB.get('thread') is not None and SLOW_JOB['thread'].is_alive():
        if not replace:
            return 'async_busy', None
        _abandon_async_slow(reason='replace_launch')
    gap = time.time() - P['last_planning_wall']
    if P['last_planning_wall'] > 0 and gap < SLOW_MIN_GAP_S:
        return 'rate_limited', None
    if P['slow'] >= MAX_SLOW: return 'slow_cap', None
    both, rgb, dep = robot._combo_rgb()
    ctx = _slow_ctx(intent, result)
    job_id = _next_slow_job_id()
    result_box = _new_slow_result_box(job_id)
    snapshot = _prepare_slow_snapshot(ctx, crit, 'slow_think_async', both, f'async_{job_id}')
    SLOW_JOB.update(snap_dec=rt.dec, dep=dep.copy(), both=both.copy(), intent=intent, result=result,
                    mode='async', crit=crit, out=None, err=None, rec=None, parsed=None,
                    job_id=job_id, result_box=result_box, target_rev=GOAL_TARGET['rev'],
                    closed=bool(EP.get('closed')), n_close=int(EP.get('n_close', 0)),
                    goal_confirmed=vision.goal_id_confirmed())
    th = threading.Thread(target=_slow_worker, args=(job_id, result_box, snapshot), daemon=True)
    SLOW_JOB['thread'] = th; th.start()
    return 'async_launched', {'snap_dec': rt.dec, 'intent': intent, 'crit': crit, 'job_id': job_id}


def _sync_slow(intent, result, crit):
    """Block on Codex using fresh stills, then apply before the next Jev intent."""
    # Abandon in-flight async without applying (critical needs a fresh post-event view).
    # Invalidate job_id so a late worker cannot overwrite / be polled as current (B01).
    if SLOW_JOB.get('thread') is not None or SLOW_JOB.get('job_id') is not None:
        th = SLOW_JOB.get('thread')
        if th is not None and not th.is_alive():
            # Finished but we intentionally discard (account cost, no apply)
            box = SLOW_JOB.get('result_box') or {}
            _log_slow_result(box)
            _account_slow_rec(box.get('rec') or SLOW_JOB.get('rec'))
            _clear_slow_job()
        else:
            _abandon_async_slow(reason=f'sync:{crit}')
    _drain_orphan_slow_costs()
    reserved = (crit in ('motion_obstruction', 'recovery_no_progress') and MAX_SLOW > 0 and
                P['slow'] >= MAX_SLOW and not EP.get('motion_help_used'))
    if P['slow'] >= MAX_SLOW and not reserved: return 'slow_cap', None
    if reserved:
        EP['motion_help_used'] = True  # spend once even when the request fails
        crit = 'reserved_motion_obstruction'
    both, rgb, dep = robot._combo_rgb()
    ctx = _slow_ctx(intent, result)
    try:
        import coding_agent_slow as CAS
        snapshot = _prepare_slow_snapshot(ctx, crit, 'sync', both, f'sync_{_next_slow_job_id()}')
        parsed, rec = CAS.call_monitor(snapshot['coding_context'], mode='codex')
        geom = snapshot['coding_context'].get('robot_geometry', {})
        rec = dict(rec, robot_geometry_source={k: geom[k] for k in ('observation_step', 'target_id', 'status') if k in geom})
    except Exception as e:
        log(t=T[0], coding_agent_slow_error=True, mode='codex', sync=True, err=str(e)[:300])
        return f'sync_error:coding_agent:{e}', None
    log(t=T[0], coding_agent_slow=True, mode='codex', sync=True, crit=crit,
        issue=parsed.get('issue'), intervene=parsed.get('intervene'),
        plan=parsed.get('plan'), backend=rec.get('backend'))
    # Applying the reply moves the robot; failures there are harness errors, not Codex errors.
    return _finish_slow(parsed, rec, dep, both, intent, result, 'sync_codex', crit, rt.dec)


def do_slow_think(intent, result, force=False):
    """Hybrid slow-think after a Jev intent.
    - Most intents: fire async Codex watch on stills; discard if >1 intent stale when ready.
    - Critical (after close / release / no-progress / stage-fail+grip-unchanged): SYNC pause, look, maybe intervene.
    - Hard force (stall / identity-unstable), acting ask_astra, and spectator-gate defer also SYNC; all share MAX_SLOW.
    Mid-episode never uses video; end-of-episode lesson update still watches the full mp4.
    """
    if not rt.FALLBACK: return 'disabled', None
    if DONE[0] or T[0] >= HORIZON: return 'done', None
    if intent in ('finish',) and result == 'finish': return 'skip_finish', None
    crit = _slow_critical(intent, result) if not force else (force if isinstance(force, str) else 'forced')
    if force is True: crit = 'forced'
    # After close with proprio+lift confirm: do not run neighbour undo via sync slow — go lift.
    if crit == 'after_close' and (EP.get('grasp_confirm') or (EP.get('grasp_secured_once') and EP.get('closed'))):
        lift_name = 'lift' if 'lift' in {i['name'] for i in rt.LIB.get('intents', [])} else 'move_above_object'
        EP['force_next_intent'] = lift_name
        out = {'issue': 'ok', 'intervene': False, 'mode': 'grasp_confirm_skip', 'crit': 'after_close',
               'grasp_confirm': True, 'force_next': lift_name}
        with nonfatal('grasp_confirm_skip_log'):
            log(t=T[0], slow_think='grasp_confirm_skip', intent=intent, intent_result=result, detail=out)
        return 'ok', out
    if crit:
        # critical: do not apply a pending async view (may be stale vs post-close/release); sync on fresh stills
        return _sync_slow(intent, result, crit)
    # non-critical: collect finished async (apply if fresh), else launch a new watch
    polled = _poll_async_slow(wait=False)
    if polled is not None:
        return polled
    if result in ('at_target',) and EP['n_same'] > 2:
        return 'skip_at_target', None
    return _launch_async_slow(intent, result)


def _should_open_on_intervention(*, igo_n, issue=None, failure_class=None):
    """Open a closed gripper only when identity is wrong, grasp is empty, or hold is unverified.
    Verified hold (igo True) / proprio+lift grasp_confirm must NOT be dropped for neighbour repair."""
    if not EP.get('closed'):
        return False
    issue = str(issue or failure_class or '')
    if EP.get('grasp_confirm') or EP.get('grasp_secured_once'):
        if issue in ('wrong_object',):
            return True
        if igo_n is False:
            return True
        return False  # confirmed grasp — keep through neighbour/path repair
    if issue in ('wrong_object', 'empty_grasp', 'tip_push'):
        return True
    if igo_n is False:
        return True
    if igo_n is True:
        return False  # verified hold — keep during path/placement repair
    # igo null/unsure while closed: open (do not keep an unknown object)
    return True
