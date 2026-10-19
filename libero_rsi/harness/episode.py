"""Episode entry point: start-up, decision loop and result file.

    python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIBRARY.json OUTDIR [nofallback]
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import dsl_libero as dsl
import fine_steps as FS
import jev_choice as JC
import jev_observation as JO
import recovery_control as RC
import task_todo as TODO
import recovery_tracking as TRACK
from validate import validate

from . import assist, executor, grasping, observation, planning, pose, progress, recovery, robot, state, vision
from .config import (ASK_ASTRA_NAMES, FAIL_RESULTS, FINE_STEPS, HORIZON, MAX_DEC, MOTION_RECOVERIES, RECOVERY_MODE,
                     STALL_FB_ACTIONS)
from .state import (CUR, DONE, EP, FB_LESSONS, GOAL_ID, GOAL_TARGET, ID_EVENTS, NONFATAL, P, PATCH, PER, RSHAPE,
                    START_ID, T, log, new_episode_state, nonfatal, rt)

QUIET_SLOW_RESULTS = frozenset({'rate_limited', 'skip_at_target', 'skip_finish', 'disabled', 'done', 'slow_cap',
                                'async_busy', 'ask_astra_disabled', 'ask_astra_cap', 'spectator_disabled',
                                'spectator_cap'})


def _ensure_ask_astra_intents(lib):
    """Inject ask_astra / consult_astra into any library so Jev can request Codex planning help."""
    names = {i['name'] for i in lib.get('intents', [])}
    ask = {
        'id': 'ask_astra@v0', 'name': 'ask_astra', 'action': 'ask_astra', 'fingers': 'keep',
        'definition': ('Request Codex slow-think help when stuck, identity unsure, approach looks wrong, '
                       'or experience_brief suggests consulting vision. Not every step — budget-limited; '
                       'prefer after stall/timeout/empty close/identity doubt over blind close_gripper.'),
        'refpoint': 'tcp', 'status': 'hypothesis', 'evidence': 'Codex planning help',
        'tilt_deg': '0', 'tilt_dir_deg': '0',
    }
    consult = dict(ask, id='consult_astra@v0', name='consult_astra', action='consult_astra',
                   definition='Alias of ask_astra: consult Codex for an episode-only diagnosis and recovery plan (shares the slow-think budget).')
    fin_i = next((i for i, it in enumerate(lib['intents']) if it['name'] == 'finish'), len(lib['intents']))
    if 'ask_astra' not in names:
        lib['intents'].insert(fin_i, ask); fin_i += 1; names.add('ask_astra')
    if 'consult_astra' not in names:
        lib['intents'].insert(fin_i, consult); names.add('consult_astra')
    tip = (' When stuck, identity unsure, or descent stalled/empty close, you may select ask_astra '
           '(or consult_astra) to request Codex planning help — not every step; it shares the slow-think budget.')
    instr = lib.get('intent_instructions', '') or ''
    if 'ask_astra' not in instr:
        lib['intent_instructions'] = instr + tip


def load_library(path):
    with open(path) as f:
        lib = json.load(f)
    if FINE_STEPS:
        FS.apply_fine_steps(lib, True)
    validate(lib)  # Reject invalid libraries before creating a simulator or making model calls.
    _ensure_ask_astra_intents(lib)
    if pose._install_pose_menu(lib):
        validate(lib)
    for name in sorted(MOTION_RECOVERIES):
        if name not in {it['name'] for it in lib['intents']}:
            lib['intents'].append({'id': name + '@v0', 'name': name,
                'action': 'move', 'fingers': 'keep', 'refpoint': 'tcp',
                'definition': 'Bounded motion recovery using current observations.',
                'status': 'hypothesis', 'evidence': 'executor recovery'})
    return lib


def start_episode(argv):
    """Load the library, create the simulator and lock the start target (no Jev decisions yet)."""
    state.reset()
    robot.diag.__dict__.pop('_xyz0', None)
    rt.suite, rt.tid, rt.init_i, rt.LIBP, rt.OUT = argv[0], int(argv[1]), int(argv[2]), argv[3], argv[4]
    rt.FALLBACK = 'nofallback' not in argv[5:]
    rt.LIB = load_library(rt.LIBP)
    rt.ep_dir = os.path.join(rt.OUT, f'{rt.suite}_t{rt.tid}_i{rt.init_i}'); os.makedirs(rt.ep_dir, exist_ok=True)
    rt.logf = open(os.path.join(rt.ep_dir, 'decisions.jsonl'), 'w')
    rt.META = {'bench': 'libero', 'suite': rt.suite, 'task': rt.tid, 'init': rt.init_i,
               'mode': 'rsi_adapter_v2_shape_fbpatch', 'lib': rt.LIB.get('version')}
    from libero.libero import benchmark
    rt.b = benchmark.get_benchmark_dict()[rt.suite](); rt.task = rt.b.get_task(rt.tid); rt.LANG = rt.task.language
    rt.env, rt.obs = robot.make_env(); rt.sim = rt.env.env.sim
    rt.DIAGF = open(os.path.join(rt.OUT, f'{rt.suite}_t{rt.tid}_i{rt.init_i}_diag.jsonl'), 'w') if os.environ.get('DIAG') else None
    # always snapshot start poses for end-of-episode moved_most (diagnostic only)
    robot.diag()
    rt.VIDP = os.environ.get('VIDEO', os.path.join(rt.ep_dir, 'episode.mp4'))
    if rt.VIDP:
        rt.VID = robot._start_video(rt.VIDP)
    for _ in range(10): robot.step([0, 0, 0, 0, 0, 0, -1])
    robot.init_sensors()
    vision.start_identity_check()
    rt.Q0 = Rot.from_quat(rt.obs['robot0_eef_quat'])
    rt.HOME_EEF = robot.eef().copy()
    EP.update(new_episode_state())


def _apply_recovery_rules(ns, intents, forced):
    """Fire library recovery rules; return (forced intent, whether pose/PATCH changed)."""
    recovery_mutated = False
    for rc in rt.LIB.get('recoveries', []):
        try:
            if forced: break
            trig_ok = dsl.evaluate(rc['trigger'], ns)
            uses = EP['uses'].get(rc['id'], 0)
            max_u = int(rc.get('max_uses', 3 if 'force_intent' in rc else 1))
            if not trig_ok or uses >= max_u:
                continue
            if RC.skip_rule(rc, RECOVERY_MODE):
                continue
            # Fire once: optional multi-action stall feedback / approach change + shape + force_intent.
            EP['uses'][rc['id']] = uses + 1
            if rc.get('stall_feedback'):
                sf = rc.get('stall_feedback')
                forced_act = None
                if isinstance(sf, str) and sf in STALL_FB_ACTIONS:
                    forced_act = sf
                elif rc.get('stall_feedback_action') in STALL_FB_ACTIONS:
                    forced_act = rc.get('stall_feedback_action')
                fb = recovery._apply_stall_feedback(ns, f"recovery:{rc['id']}", action=forced_act)
                recovery_mutated = True
                log(dec=rt.dec, t=T[0], recovery=rc['id'], stall_feedback=fb)
                # Prefer the feedback menu's chosen next intent over a static force_intent.
                if EP.get('force_next_intent') and EP['force_next_intent'] in intents:
                    forced = EP['force_next_intent']; EP['force_next_intent'] = None
                elif 'force_intent' in rc and rc['force_intent'] in intents:
                    forced = rc['force_intent']
                break
            if rc.get('approach_change'):
                recovery._apply_relative_approach_change(ns, f"recovery:{rc['id']}")
                recovery_mutated = True
            sh = rc.get('shape') or {}
            if sh:
                e = lambda k: float(dsl.evaluate(str(sh[k]), ns))
                for k, rk in (('tilt_deg', 'tilt'), ('tilt_dir_deg', 'tdir'), ('yaw_deg', 'yaw')):
                    if k in sh: RSHAPE[rk] = e(k)
                if 'grasp_dz_objh' in sh: RSHAPE['dz'] = e('grasp_dz_objh')
                # Relative yaw delta (dimensionless degrees) added to current command — preferred over absolute yaw_deg.
                if 'yaw_delta_deg' in sh:
                    cur = vision._cmd_yaw_deg(); RSHAPE['yaw'] = float(np.clip(cur + e('yaw_delta_deg'), -90, 90))
                    PATCH['yaw_deg'] = RSHAPE['yaw']
                recovery_mutated = True
                vision._note_patch_mutation(f"recovery:{rc['id']}", sh)
                log(dec=rt.dec, t=T[0], shape_rule=rc['id'], rshape=dict(RSHAPE), patch_yaw=PATCH.get('yaw_deg'))
            if 'force_intent' in rc and rc['force_intent'] in intents:
                forced = rc['force_intent']; break
        except Exception as ex:
            log(dec=rt.dec, t=T[0], recovery_error=rc.get('id'),
                error_type=type(ex).__name__, error=str(ex))
            # A rule can fail after moving the arm or partially updating state.
            # Stop this episode; continuing would use an unknown recovery state.
            raise
    return forced, recovery_mutated


def _request_help(it, r):
    """Codex planning help after an executed intent; return a slow-think event to record, or None.

    ask_astra already ran sync in execute; else hard-force; else spectator gate; else normal slow-think.
    """
    if it == 'open_gripper' and EP.pop('just_released', False) and r not in FAIL_RESULTS:
        planning._abandon_async_slow('post_release_settle')
        log(t=T[0], post_release_help='deferred_to_current_image_review')
        return None
    if it not in ASK_ASTRA_NAMES and TRACK.handled(EP, T[0]):
        log(t=T[0], automatic_help_skipped='recovery_already_handled')
        return None
    if it in ASK_ASTRA_NAMES:
        sr, so = f'ask_done:{r}', {'via': 'execute', 'result': r}
        log(dec=rt.dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so, source='ask_astra')
        return {'dec': rt.dec, 'intent': it, 'intent_result': r, 'result': sr, 'detail': so, 'source': 'ask_astra'}
    hard = assist._hard_astra_trigger(it, r)
    if hard:
        sr, so = planning._sync_slow(it, r, crit=hard)
        if sr == 'ok':
            assist._consume_help_budget(hard)
        log(dec=rt.dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so, source=hard)
        return {'dec': rt.dec, 'intent': it, 'intent_result': r, 'result': sr, 'detail': so, 'source': hard}
    # Spectator Jev (旁观者清): additive gate — may request Codex help when acting looks stuck
    if assist._spectator_gate_eligible(it, rt.dec):
        signals = assist._spectator_distress_signals(it, r)
        if signals:
            # Fresh ns/facts after execute (and any recovery) for spectator inputs (B05).
            ns_gate, preds_gate = observation.namespace(); f_gate = observation.facts(ns_gate, preds_gate)
            defer, gdet = assist._spectator_jev_should_ask_astra(it, r, signals, ns_gate, f_gate, rt.dec)
            if defer:
                reason = ','.join(signals)
                sr, so = assist.do_spectator_astra(it, r, reason)
                log(dec=rt.dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so,
                    source='spectator_jev', gate=gdet)
                return {'dec': rt.dec, 'intent': it, 'intent_result': r, 'result': sr,
                        'detail': so, 'source': 'spectator_jev', 'gate': gdet}
    sr, so = planning.do_slow_think(it, r, force=False)
    if so is not None or sr not in QUIET_SLOW_RESULTS:
        log(dec=rt.dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so)
        return {'dec': rt.dec, 'intent': it, 'intent_result': r, 'result': sr, 'detail': so}
    return None


def run_decision_loop(run):
    """Jev / recovery decisions until success, finish, decision cap or horizon."""
    intents = {i['name']: i for i in rt.LIB['intents']}
    while T[0] < HORIZON and not DONE[0] and rt.dec < MAX_DEC:
        progress._refresh_task_progress()
        if EP.get('todo_review_due'):
            progress._review_task_todo(EP.pop('todo_review_due'))
        if EP['closed'] and not any(robot.contacts()):
            EP['grasp_confirm'] = False
            EP['grasp_secured_once'] = False
            PER['seen_held'] = False
        trig = assist.fb_trigger()
        if trig:  # early perception-and-Codex recovery: episode-only re-perception (never edits the library)
            P['fb'] += 1; r, pt = assist.do_fallback(trig); run['fb_events'].append({'dec': rt.dec, 'trigger': trig, 'result': r, 'patch': pt})
            log(dec=rt.dec, t=T[0], fallback=trig, result=r, patch=pt)
            CUR.update(fb=f"{trig} -> {r} " + (f"dz={pt['grasp_dz_objh']:+.2f}objh tilt={pt['tilt_deg']} yaw={pt['yaw_deg']}" if pt else ''), fb_t=T[0])
        if DONE[0] or T[0] >= HORIZON:
            break  # Recovery itself can finish the task; do not select/execute another intent.
        ns, preds = observation.namespace(); f = observation.facts(ns, preds); forced = None
        # Escape / recovery may have queued a non-close next intent — honor before Jev.
        if EP.get('force_next_intent') and EP['force_next_intent'] in intents:
            forced = EP['force_next_intent']; EP['force_next_intent'] = None
        forced, recovery_mutated = _apply_recovery_rules(ns, intents, forced)
        if DONE[0] or T[0] >= HORIZON:
            break
        # B05: recovery mutates pose/PATCH — recompute ns/facts before selection and execute.
        if recovery_mutated:
            ns, preds = observation.namespace(); f = observation.facts(ns, preds)
            with nonfatal('post_recovery_log', dec=rt.dec):
                log(dec=rt.dec, t=T[0], post_recovery_ns_refresh=True, forced=forced,
                    grasp=list(np.asarray(ns['grasp'], float).round(4)),
                    tcp=list(np.asarray(ns['tcp'], float).round(4)))
        progress._refresh_task_progress()
        if forced:
            it, probs = forced, None
        else:
            jev_state, assets = observation._jev_input(include_execution_history=True)
            menu = observation._acting_menu(intents, ns)
            execution = observation._candidate_execution(menu, ns, jev_state)
            criteria = JO.candidate_definitions(menu, execution)
            log(t=T[0], candidate_execution=execution, candidate_definitions=criteria)
            it, probs, _ = JC.choose(jev_state, JO.INSTRUCTIONS,
                                     criteria, assets=assets, meta=rt.META)
            run['jev_calls'] += 1
        CUR['intent'] = it; r = executor.execute(it, ns); rt.dec += 1
        progress._refresh_task_progress(it, r)
        EP['last_result'] = r
        EP['todo_review_due'] = f'{it}:{r}'
        EP['n_same'] = EP['n_same'] + 1 if it == EP['prev'] else 1; EP['prev'] = it; EP['step'] += 1
        EP['hist'].append(it); EP['tcp_hist'].append(robot.eef())
        EP['result_hist'].append(r); EP['grip_hist'].append(bool(EP['closed']))
        contact_state = robot.robot_contacts()
        EP.setdefault('recovery_contact_hist', []).append(
            contact_state['environment_contact_parts'] + contact_state['self_contact_parts'])
        if it == 'descend_to_grasp' and r in ('reached', 'at_target', 'stalled', 'timeout'):
            EP['last_descent_step'] = T[0]
        # Place-phase thrash tracking → maybe queue force open_gripper for next decision
        grasping._note_place_phase(it, r)
        grasping._maybe_queue_place_force_release()
        # grip change clears stage-fail streak debounce so a later stuck phase can fire again
        if len(EP['grip_hist']) >= 2 and EP['grip_hist'][-1] != EP['grip_hist'][-2]:
            EP['stage_fail_mark'] = 0
        log(dec=rt.dec, t=T[0], intent=it, forced=forced, result=r, facts=f, probabilities=probs)
        if r != 'finish' and not DONE[0] and T[0] < HORIZON:
            event = _request_help(it, r)
            if event:
                run['slow_events'].append(event)
        progress._refresh_task_progress()
        if r == 'finish': break
    # LIBERO official eval does not stop on a policy 'finish': keep no-op stepping (gripper holds last command) until done or HORIZON
    stop = 'finish' if EP['prev'] == 'finish' else ('decision_cap' if rt.dec >= MAX_DEC else None); t_stop = T[0]
    while not DONE[0] and T[0] < HORIZON:
        robot.step([0, 0, 0, 0, 0, 0, 1 if EP['closed'] else -1])
    if not DONE[0] and EP.get('todo_review_due'):
        progress._review_task_todo(EP.pop('todo_review_due'), retarget=False)
    if DONE[0]: end = 'success' if stop is None else f'{stop}_then_success'
    else: end = f'{stop}_not_success' if stop else 'horizon'
    progress._refresh_task_progress()
    run['res']['stopped_at_step'] = t_stop if stop else None
    return end


def _moved_most():
    """Diagnostic: which sim body moved most (never shown to policy)."""
    try:
        E = rt.env.env; xyz_now = {n: E.sim.data.body_xpos[i].copy() for n, i in E.obj_body_id.items()}
        if hasattr(robot.diag, '_xyz0'):
            moved = sorted(((n, float(np.linalg.norm(xyz_now[n] - robot.diag._xyz0[n]))) for n in robot.diag._xyz0 if n in xyz_now), key=lambda x: -x[1])
            return [{'name': n, 'disp_m': round(d, 4)} for n, d in moved[:5]]
        # capture start poses now if DIAG was off during steps
        return [{'name': n, 'disp_m': None} for n in list(xyz_now)[:5]]
    except Exception as _e:
        return [{'error': str(_e)[:120]}]


def finish_episode(run, end):
    planning._finalize_slow_jobs()
    if DONE[0] and EP.get('task_todo'):
        EP['task_todo'] = TODO.official_completion(EP['task_todo'], T[0])
        progress._save_task_todo('official_success')
    res = run['res']
    res.update(success=bool(DONE[0]), end_reason=end, env_steps=T[0], decisions=rt.dec, jev_calls=run['jev_calls'], astra_calls=P['astra_calls'],
               astra_cost_usd=round(P['astra_cost'], 4), planning_calls=P['planning_calls'],
               planning_cost_usd=round(P['planning_cost'], 4), relooks=P['relook'], fallback_events=run['fb_events'], slow_think_events=run['slow_events'],
               slow_interventions=P['slow'], gate_jev_calls=EP.get('gate_jev_calls', 0), final_patch=PATCH, fallback_lessons=FB_LESSONS, learned_shape=RSHAPE,
               task_todo=EP.get('task_todo'), todo_review_calls=EP.get('todo_calls', 0), todo_review_model=JC.MODEL,
               start_identity=START_ID, identity_events=ID_EVENTS, goal_id=GOAL_ID, goal_target=GOAL_TARGET,
               close_refusals=GOAL_ID.get('close_refusals', 0), moved_most=_moved_most(), wall_s=round(time.time() - run['t0'], 1),
               motion_diagnosis_used=bool(EP.get('motion_help_used')),
               place_force_release=bool(EP.get('place_force_release')),
               place_force_release_reason=EP.get('place_force_release_reason'),
               place_lower_stall_n=int(EP.get('place_lower_stall_n', 0)),
               place_lift_lower_cycles=int(EP.get('place_lift_lower_cycles', 0)))
    if rt.VID:
        try:
            rt.VID.stdin.close()
        except BrokenPipeError:
            pass
        video_rc = rt.VID.wait()
        res['video_path'] = rt.VIDP
        res['video_exit_code'] = video_rc
    res['recovery_control'] = RECOVERY_MODE
    if pose._pose_menu_visible():
        res['pose_menu_counts'] = dict(EP.get('pose_menu_counts') or {})
    if FINE_STEPS or EP.get('fine_step_events'):
        res['fine_steps_enabled'] = bool(FINE_STEPS)
        res['fine_step_events'] = list(EP.get('fine_step_events') or [])
    with nonfatal('env_close'):
        rt.env.close()  # explicit EGL teardown (avoid __del__ EGLError at interpreter exit)
    res['nonfatal_errors'] = list(NONFATAL)
    with open(os.path.join(rt.ep_dir, 'result.json'), 'w') as f:
        json.dump(res, f, indent=1, default=lambda o: np.asarray(o).tolist())
    print(json.dumps({k: res[k] for k in ('suite', 'task_id', 'init', 'success', 'end_reason', 'env_steps', 'decisions', 'astra_cost_usd', 'slow_interventions', 'close_refusals', 'wall_s') if k in res}))
    sys.stdout.flush(); rt.logf.close()
    return res


def main(argv):
    """Run one episode; return the process exit code (1 for harness_error)."""
    start_episode(argv)
    # ---- episode-start Great Wall global plan (before first approach) ----
    rt.dec = 0
    try:
        start_plan = planning._run_episode_start_plan()
    except Exception as _sp_ex:
        start_plan = {'mode': 'error', 'error': str(_sp_ex)[:300]}
        log(t=T[0], episode_start_plan_error=True, err=start_plan['error'])
    run = {'t0': time.time(), 'jev_calls': 0, 'fb_events': [], 'slow_events': [],
           'res': {'suite': rt.suite, 'task_id': rt.tid, 'init': rt.init_i, 'language': rt.LANG, 'library': rt.LIBP,
                   'library_version': rt.LIB.get('version'), 'fallback': rt.FALLBACK, 'perception0': rt.per0,
                   'start_plan': start_plan}}
    try:
        if start_plan.get('error'):
            raise RuntimeError(f"episode_start_plan_failed:{start_plan['error']}")
        end = run_decision_loop(run)
    except Exception:
        end = 'harness_error'; run['res']['error'] = traceback.format_exc()[-1200:]
    finish_episode(run, end)
    return 1 if end == 'harness_error' else 0
