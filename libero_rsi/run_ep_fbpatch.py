"""LIBERO honest adapter + RSI library policy (Astra perception, Jev intent choice, typed rule library).
python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIBRARY.json OUTDIR [nofallback]
Policy-visible signals (documented in ADAPTER.md):
  proprioception: robot0_eef_pos (tcp), gripper finger qpos -> opening, last gripper command (closed_cmd)
  tactile proxy: per-finger contact flags from MuJoCo contact list (finger pad/collision geoms of gripper0 touching any
                 geom that is not robot/gripper/mount and not a static fixture body [world/table/floor]). Only booleans; no
                 identity, pose or force of the touched object.
  Astra: one RGB-D (agentview) keypoint perception at t=0 (grasp/object-centre/place pixels -> depth backprojection with
         known camera), plus re-looks (library intent `relook` or early fallback) from the current image.
  Grasp-target ownership (single shared goal): PER holds keypoint base; PATCH holds episode offsets on that same base.
         Priority: latest relook keypoints > slow-think patch on current target > episode-start perception.
         Relook / keypoint refresh overwrites PER and clears position offsets in PATCH; confirmed approach yaw/tilt in PATCH
         survive unless new keypoints explicitly change yaw (or identity-bad full reset). Slow-think / stall mutates PATCH only.
         On stall with identity confirmed: must update the shared target (raise/shift/relook-refresh); relook executes fresh perception.
  success: official env (BDDL check_success) -> done flag. Binary only, ends the episode.
No simulator object/target ground-truth poses reach the policy.
Jev receives task todo, same-frame images/proprioception, measured motion/contact feedback and historical startup obstacle advice.
Close/release applicability is checked by the executor; unchanged failed waypoints require recovery.
Codex planning help: the existing ask_astra/consult_astra intent names request Codex; hard rules and a spectator Jev may request synchronous help on distress signals; general help shares MAX_SLOW; one additional motion-obstruction diagnosis is available after exhaustion (if MAX_SLOW > 0).
Codex handles episode-start global planning and sync/async slow-think. Astra, invoked through Codex CLI, handles visual perception and identity verification; Jev selects each intent. Model roles are fixed.
Start planning includes task_todo; Astra reviews outcome progress on current images. Jev selects applicable actions for the active objective.
After close_gripper: proprio+lift grasp confirm (log grasp_confirm); secured → lift, suppress push_aside/local_deadend undo.
Hard local-deadend rule: repeated local stall family (push_aside/clear_reapproach/raise_redescend/lateral) without progress, or push_aside episode/approach cap → retreat + global front relook + SYNCHRONOUS slow-think replan (Codex slow-think); Jev motion held until plan applied.
Place-phase: after carry_to_place reached, N consecutive lower_to_place stall/timeout (MAX_PLACE_LOWER_STALL) or M lift↔lower cycles (MAX_PLACE_LIFT_LOWER_CYCLES) while TCP/held XY is over the place region → force open_gripper (log place_force_release); avoid endless lift↔lower thrash over basket.
Freshness/isolation: each async slow-think job has a unique job_id + private result_box (abandoned workers cannot overwrite a newer job); after keypoint refresh, close is blocked until TCP re-approaches the new grasp XY; recovery mutations recompute namespace/facts before execute and spectator.
"""
import os, sys, json, time, gc, re, threading, queue
os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# The bundled TypeSafe client is part of this checkout; no external text-only
# jev_choice module may shadow the production image transport.
import jev_choice as JC
import jev_observation as JO
import task_todo as TODO
import motion_feedback as MF
import fine_steps as FS
import numpy as np
import dsl_libero as dsl
from validate import validate
import jev_experience as JX
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import camera_utils as CU
from scipy.spatial.transform import Rotation as Rot

suite, tid, init_i, LIBP, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
FALLBACK = 'nofallback' not in sys.argv[6:]
S = 512
HORIZON = int(os.environ.get('HORIZON', '600'))
MAX_DEC = int(os.environ.get('MAX_DEC', '60'))
MAX_RELOOK, MAX_FB = 4, int(os.environ.get('MAX_FB', '3'))
MAX_SLOW = int(os.environ.get('MAX_SLOW', '4'))  # light slow-think interventions per episode
MAX_PUSH_ASIDE = int(os.environ.get('MAX_PUSH_ASIDE', '2'))  # episode cap on push_aside (burns horizon if unbounded)
MAX_PUSH_ASIDE_PER_APPROACH = int(os.environ.get('MAX_PUSH_ASIDE_PER_APPROACH', '1'))  # per approach streak
LOCAL_DEADEND_N = int(os.environ.get('LOCAL_DEADEND_N', '3'))  # consecutive local-loop stall family → global relook
MAX_LOCAL_DEADEND_GLOBAL = int(os.environ.get('MAX_LOCAL_DEADEND_GLOBAL', '2'))  # max deadend→global sync slow per episode
MAX_PLACE_LOWER_STALL = int(os.environ.get('MAX_PLACE_LOWER_STALL', '3'))  # consecutive lower_to_place stall/timeout → force release
MAX_PLACE_LIFT_LOWER_CYCLES = int(os.environ.get('MAX_PLACE_LIFT_LOWER_CYCLES', '2'))  # lift↔lower thrash cap during place
PLACE_XY_M = float(os.environ.get('PLACE_XY_M', '0.04'))  # held-object XY near place → allow force release
PLACE_CLEARANCE_DEFAULT = float(os.environ.get('PLACE_CLEARANCE_DEFAULT', '0.05'))  # default place_clearance_m if Astra omits
ASTRA_EFFORT = os.environ.get('ASTRA_EFFORT', 'xhigh')  # Astra perception/identity effort through Codex CLI
SLOW_MIN_GAP_S = float(os.environ.get('SLOW_MIN_GAP_S', '5'))  # gap between completed Codex planning calls (unless critical)
STAGE_FAIL_N = int(os.environ.get('STAGE_FAIL_N', '3'))  # consecutive timeout/stall + unchanged gripper -> sync slow-think
TRANSPORT_INTENTS = frozenset({'lift', 'carry_to_place', 'lower_to_place'})
GRASP_FAILED = 'grasp_failed:transport_not_executed'
CLOSE_NOT_READY = 'close_not_executed:not_at_grasp'
MOTION_RECOVERIES = frozenset({'back_off', 'clear_reapproach', 'change_approach_yaw'})
FAIL_RESULTS = frozenset({'timeout', 'stalled', GRASP_FAILED, CLOSE_NOT_READY} |
                         {f'{name}_not_executed:grasp_unconfirmed' for name in TRANSPORT_INTENTS})
OPEN_MAX = 0.08  # Panda gripper spec (robot spec, not scene)
FINE_STEPS = FS.enabled_flag()
LIB = json.load(open(LIBP))
if FINE_STEPS:
    FS.apply_fine_steps(LIB, True)
validate(LIB)  # Reject invalid libraries before creating a simulator or making model calls.
ASK_ASTRA_NAMES = frozenset({'ask_astra', 'consult_astra'})
GATE_JEV = os.environ.get('GATE_JEV', '1') != '0'  # spectator Jev gate (旁观者清); 0 disables
GATE_JEV_COOLDOWN = int(os.environ.get('GATE_JEV_COOLDOWN', '3'))  # min decisions between spectator Jev calls
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
_ensure_ask_astra_intents(LIB)
for _name in sorted(MOTION_RECOVERIES):
    if _name not in {it['name'] for it in LIB['intents']}:
        LIB['intents'].append({'id': _name + '@v0', 'name': _name,
            'action': 'move', 'fingers': 'keep', 'refpoint': 'tcp',
            'definition': 'Bounded motion recovery using current observations.',
            'status': 'hypothesis', 'evidence': 'executor recovery'})
ep_dir = os.path.join(OUT, f'{suite}_t{tid}_i{init_i}'); os.makedirs(ep_dir, exist_ok=True)
logf = open(os.path.join(ep_dir, 'decisions.jsonl'), 'w')
def log(**k): logf.write(json.dumps(k, default=lambda o: np.asarray(o).round(4).tolist()) + '\n'); logf.flush()
META = {'bench': 'libero', 'suite': suite, 'task': tid, 'init': init_i, 'mode': 'rsi_adapter_v2_shape_fbpatch', 'lib': LIB.get('version')}

b = benchmark.get_benchmark_dict()[suite](); task = b.get_task(tid); LANG = task.language
def make_env():
    # EGL fix: retry context creation after gc (stale contexts from previous processes / parallel jobs)
    last = None
    for a in range(4):
        try:
            e = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file),
                                   camera_heights=S, camera_widths=S, camera_depths=True, horizon=HORIZON)
            e.seed(0); e.reset(); o = e.set_init_state(b.get_task_init_states(tid)[init_i]); return e, o
        except Exception as ex:
            last = ex; gc.collect(); time.sleep(3 + 3 * a)
    raise last
env, obs = make_env(); sim = env.env.sim
T = [0]; DONE = [False]
DIAGF = open(os.path.join(OUT, f'{suite}_t{tid}_i{init_i}_diag.jsonl'), 'w') if os.environ.get('DIAG') else None
def diag():  # DIAGNOSTIC ONLY (never shown to policy): per-step BDDL goal predicate parts + poses
    E = env.env; osd = E.object_states_dict; out = {'t': T[0], 'check_success': bool(E._check_success()), 'done_flag': DONE[0]}
    for g in E.parsed_problem['goal_state']:
        if g[0].lower() == 'in':
            A, B = osd[g[1]], osd[g[2]]
            out[f'In({g[1]},{g[2]})'] = {'contact': bool(B.check_contact(A)), 'contain': bool(B.check_contain(A)),
                'obj_xyz': np.round(E.sim.data.body_xpos[E.obj_body_id[g[1]]], 4).tolist(),
                'region_xyz': np.round(E.sim.data.get_site_xpos(g[2]), 4).tolist() if g[2] in E.object_sites_dict else None}
    out['all_obj_xyz'] = {n: np.round(E.sim.data.body_xpos[i], 3).tolist() for n, i in E.obj_body_id.items()}
    if not hasattr(diag, '_xyz0'): diag._xyz0 = {n: np.asarray(v, float).copy() for n, v in out['all_obj_xyz'].items()}
    else:
        moved = sorted(((n, float(np.linalg.norm(np.asarray(out['all_obj_xyz'][n]) - diag._xyz0[n]))) for n in diag._xyz0), key=lambda x: -x[1])
        out['moved_most'] = [{'name': n, 'disp_m': round(d, 4)} for n, d in moved[:3] if d > 0.01]
    if DIAGF: DIAGF.write(json.dumps(out) + '\n'); DIAGF.flush()
# always snapshot start poses for end-of-episode moved_most (diagnostic only)
diag()
def _start_video(path):
    import subprocess
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x572', '-r', '20', '-i', '-',
                             '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', '-crf', '26', '-preset', 'medium', '-movflags', '+faststart', path], stdin=subprocess.PIPE)


VIDP = os.environ.get('VIDEO', os.path.join(ep_dir, 'episode.mp4')); VID = None; CUR = {'intent': '-', 'fb': '', 'fb_t': -999, 'slow': '', 'slow_t': -999}
if VIDP:
    from PIL import Image as _I, ImageDraw as _D
    VID = _start_video(VIDP)
def vid_frame(a):
    im = _I.new('RGB', (1024, 572)); im.paste(_I.fromarray(obs['agentview_image'][::-1]), (0, 0))
    if 'robot0_eye_in_hand_image' in obs: im.paste(_I.fromarray(obs['robot0_eye_in_hand_image'][::-1]).resize((512, 512)), (512, 0))
    d = _D.Draw(im); g = 'CLOSED' if a[6] > 0 else 'OPEN'
    op = float(obs['robot0_gripper_qpos'][0] - obs['robot0_gripper_qpos'][1])
    d.text((6, 516), f"step {T[0]}/{HORIZON}   intent: {CUR['intent']}   gripper cmd: {g}  opening {op*100:.1f}cm   {'SUCCESS' if DONE[0] else ''}", fill=(255, 255, 0))
    if T[0] - CUR['fb_t'] < 60: d.text((6, 536), f"FALLBACK: {CUR['fb']}"[:170], fill=(255, 80, 80))
    elif T[0] - CUR['slow_t'] < 60: d.text((6, 536), f"SLOW: {CUR['slow']}"[:170], fill=(80, 200, 255))
    d.text((6, 554), 'left: agentview   right: wrist cam', fill=(180, 180, 180))
    VID.stdin.write(im.tobytes())
def step(a):
    global obs
    if DONE[0] or T[0] >= HORIZON:
        return
    obs, r, done, info = env.step(np.clip(a, -1, 1)); T[0] += 1
    if VID: vid_frame(a)
    if DIAGF and globals().get('EP', {}).get('n_close', 0) > 0: diag()
    if env.env._check_success(): DONE[0] = True
for _ in range(10): step([0, 0, 0, 0, 0, 0, -1])

# ---- tactile proxy ----
m = sim.model
FING = {'L': {m.geom_name2id(n) for n in ('gripper0_finger1_collision', 'gripper0_finger1_pad_collision')},
        'R': {m.geom_name2id(n) for n in ('gripper0_finger2_collision', 'gripper0_finger2_pad_collision')}}
def _fixture(g):
    bid = m.geom_bodyid[g]; n = (m.body_id2name(bid) or '').lower()
    return bid == 0 or n.startswith(('robot0', 'gripper0', 'mount0')) or 'table' in n or 'floor' in n
FIX = {g for g in range(m.ngeom) if _fixture(g)}
def contacts():
    l = r = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]; a, bb = c.geom1, c.geom2
        for f, o in ((a, bb), (bb, a)):
            if o in FIX: continue
            if f in FING['L']: l = True
            if f in FING['R']: r = True
    return l, r

# Robot geometry names describe our own sensor locations, never the touched body.
ROBOT_PARTS = {}
for _g in range(m.ngeom):
    _body = (m.body_id2name(m.geom_bodyid[_g]) or '').lower()
    if _g in FING['L']: ROBOT_PARTS[_g] = 'left_finger'
    elif _g in FING['R']: ROBOT_PARTS[_g] = 'right_finger'
    elif _body.startswith('gripper0'): ROBOT_PARTS[_g] = 'palm_or_wrist'
    elif _body.startswith('robot0'): ROBOT_PARTS[_g] = 'arm'

def robot_contacts():
    pairs = [(sim.data.contact[i].geom1, sim.data.contact[i].geom2)
             for i in range(sim.data.ncon)]
    return MF.contact_summary(pairs, ROBOT_PARTS)

def camera():
    cid = m.camera_name2id('agentview'); return sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy(), 0.5 * S / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
CP, RM, F = camera()
def rgbd():
    return obs['agentview_image'][::-1].copy(), CU.get_real_depth_map(sim, obs['agentview_depth'])[..., 0][::-1].copy()
def bp(uv, dep):
    u, v = int(np.clip(round(uv[0]), 2, S - 3)), int(np.clip(round(uv[1]), 2, S - 3))
    d = float(np.median(dep[v - 2:v + 3, u - 2:u + 3])); return CP + RM @ np.array([(uv[0] - S / 2) / F, -(uv[1] - S / 2) / F, -1.0]) * d

P = {'astra_cost': 0.0, 'astra_calls': 0, 'relook': 0, 'fb': 0, 'slow': 0, 'last_astra_wall': 0.0,
     'planning_calls': 0, 'planning_cost': 0.0, 'last_planning_wall': 0.0}
ID_EVENTS = []  # identity checks at start / relook / fallback (policy-visible reflection evidence)
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
# Episode gate: only True counts as confirmed. null/False/unsure => refuse close until a relook matches.
GOAL_ID = {'confirmed': False, 'source': None, 'detail': None, 'close_refusals': 0, 'events': []}
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
    item = TODO.active_item(globals().get('EP', {}).get('task_todo'))
    if item is None:
        return LANG
    return f"{LANG}\nCurrent TODO target: {item['target']}. Current objective: {item['goal']}. Localize and verify ONLY this target."


def perceive(context=None, phase=None):
    """Use Codex CLI to ask Astra for visual keypoints at start and on relook."""
    import coding_agent_slow as CAS
    from PIL import Image as _PILImage
    rgb, dep = rgbd()
    ph = phase or ('perceive' if not context else 'relook')
    frame = os.path.join(ep_dir, f'perc_{ph}_{P["astra_calls"]}.jpg')
    _PILImage.fromarray(rgb).save(frame)
    images = [frame]
    wrist = obs.get('robot0_eye_in_hand_image')
    if wrist is not None:
        wrist_frame = os.path.join(ep_dir, f'perc_{ph}_{P["astra_calls"]}_wrist.jpg')
        _PILImage.fromarray(wrist[::-1]).save(wrist_frame)
        images.append(wrist_frame)
    parsed, rec = CAS.harness_call_perceive(
        lang=_active_language_goal(), phase=ph, image_paths=images, context=context,
        img_size=S, mode='codex', effort=ASTRA_EFFORT)
    P['astra_cost'] += rec['cost_usd']; P['astra_calls'] += 1; P['last_astra_wall'] = time.time()
    log(t=T[0], perception=True, phase=ph, backend=rec.get('backend'),
        model=rec.get('model'), effort=rec.get('effort'), wall_s=rec.get('wall_s'))
    return parsed, rgb, dep

from PIL import Image
PER = {}
FB_LESSONS = []  # qualitative fallback reflections (no numbers) -> RSI evidence; quantitative patches stay episode-only
# episode-only action patches from the early perception-and-Codex recovery (NEVER written to the library); override library shape when set
PATCH = {'grasp_dz_objh': 0.0, 'tilt_deg': None, 'tilt_dir_deg': None, 'yaw_deg': None, 'grasp_dxy_objw': [0.0, 0.0]}
# Single grasp-target owner: one PER base + one PATCH offset layer. Never two competing goals.
# source priority for the effective grasp: relook/keypoint refresh > slow_patch > episode_start
GOAL_TARGET = {'source': 'episode_start', 'base_from': None, 'last_patch_from': None, 'rev': 0}
RSHAPE = {'tilt': None, 'tdir': None, 'yaw': None, 'dz': 0.0}  # learned conditional overrides (library recoveries), episode-scoped once triggered
SHAPE = {'tilt': 0.0, 'tdir': 0.0, 'yaw': None}  # from the current library intent

def _empty_patch():
    return {'grasp_dz_objh': 0.0, 'tilt_deg': None, 'tilt_dir_deg': None, 'yaw_deg': None, 'grasp_dxy_objw': [0.0, 0.0]}

YAW_EXPLICIT_EPS_DEG = 15.0  # new keypoint yaw counts as explicit if it differs from prior PER yaw by this much

def _reset_patch(reason='relook_overwrite'):
    """Clear episode offsets so stale position patches cannot fight a newly locked keypoint base.
    Approach yaw/tilt may be restored by _commit_keypoints after a relook overwrite."""
    old = {k: (list(v) if isinstance(v, list) else v) for k, v in PATCH.items()}
    PATCH.clear(); PATCH.update(_empty_patch())
    GOAL_TARGET['rev'] = int(GOAL_TARGET.get('rev', 0)) + 1
    try:
        log(t=T[0], patch_reset=True, reason=reason, old_patch=old, goal_target=dict(GOAL_TARGET))
    except Exception:
        pass
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
    except Exception:
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
            except Exception:
                pass
    return restored

def _commit_keypoints(p, dep, source, held_now=False):
    """Write the shared keypoint BASE (PER). Relook / keypoint-refresh overwrites PER and clears position offsets.
    Confirmed approach yaw/tilt in PATCH survive unless new keypoints explicitly change yaw.
    Episode-start only seeds PER. Slow-think adjust/stall must NOT go through here — use _note_patch_mutation.
    A material target XY change while open marks need_reapproach so close cannot fire at the old TCP."""
    old_patch = {k: (list(v) if isinstance(v, list) else v) for k, v in PATCH.items()}
    old_per_yaw = PER.get('yaw')
    old_grasp = None
    try:
        if 'grasp' in PER:
            old_grasp = np.asarray(PER['grasp'], float).copy()
    except Exception:
        old_grasp = None
    # Capture commanded yaw BEFORE reset clears PATCH (needed for explicit-yaw gate).
    try:
        old_cmd_yaw = float(old_patch['yaw_deg']) if old_patch.get('yaw_deg') is not None else (
            float(RSHAPE['yaw']) if RSHAPE.get('yaw') is not None else float(PER.get('yaw', 0.0) or 0.0)
        )
    except Exception:
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
        try:
            log(t=T[0], approach_pose_preserve=True, source=src, explicit_new_yaw=explicit,
                old_per_yaw=old_per_yaw, new_per_yaw=PER.get('yaw'),
                old_patch_yaw=old_patch.get('yaw_deg'), preserved=preserved or None,
                goal_target=dict(GOAL_TARGET))
        except Exception:
            pass
    else:
        GOAL_TARGET['source'] = 'episode_start'
    GOAL_TARGET['base_from'] = src
    GOAL_TARGET['rev'] = int(GOAL_TARGET.get('rev', 0)) + 1
    # Skip start-seed commits (EP may not exist yet); only mid-episode refreshes can unlock a stale close.
    if (not is_start and not held_now and src not in ('perceive_start_relook',)
            and isinstance(globals().get('EP'), dict) and 'need_reapproach' in EP):
        try:
            _mark_reapproach_if_needed(old_grasp, f'keypoints:{src}')
        except Exception:
            pass
    try:
        log(t=T[0], goal_commit_keypoints=True, source=src, goal_target=dict(GOAL_TARGET),
            approach_preserved=preserved or None,
            need_reapproach=bool(EP.get('need_reapproach')),
            grasp=list(np.asarray(PER.get('grasp'), float).round(4)) if 'grasp' in PER else None)
    except Exception:
        pass
    return dict(GOAL_TARGET)

def _note_patch_mutation(source, detail=None):
    """Slow-think / stall feedback mutates the SAME shared PATCH on the current PER base (does not create a second goal)."""
    GOAL_TARGET['source'] = 'slow_patch'
    GOAL_TARGET['last_patch_from'] = str(source or 'patch')
    GOAL_TARGET['rev'] = int(GOAL_TARGET.get('rev', 0)) + 1
    try:
        log(t=T[0], goal_patch_mutation=True, source=source, detail=detail, patch=dict(PATCH), goal_target=dict(GOAL_TARGET))
    except Exception:
        pass
    return dict(GOAL_TARGET)

def set_perc(p, dep, held_now=False):
    PER['seen_held'] = bool(p.get('object_held_by_gripper', False))
    PER['place'] = bp(p['place_px'], dep)
    if not held_now or 'grasp' not in PER:
        PER['grasp'] = bp(p['grasp_px'], dep) - [0, 0, float(np.clip(p.get('grasp_depth_m', .02), 0, .06))]
        PER['objc'] = bp(p.get('object_center_px', p['grasp_px']), dep)
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
    except Exception: return {'confirmed': None, 'reason': 'no_grasp_px'}, rgb
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
    frame = os.path.join(ep_dir, f'identity_{phase}_{P["astra_calls"]}.jpg')
    im.save(frame)
    parsed, rec = CAS.call_identity_check(prompt, [frame], effort=ASTRA_EFFORT)
    log(t=T[0], identity_verification=True, phase=phase, backend=rec.get('backend'),
        model=rec.get('model'), effort=rec.get('effort'), wall_s=rec.get('wall_s'))
    P['astra_cost'] += rec['cost_usd']; P['astra_calls'] += 1; P['last_astra_wall'] = time.time()
    return parsed, np.asarray(im)

# ---- start-of-episode identity-checked perception (general library knowledge; not fallback-only) ----
START_ID = {'forced_relook': False, 'verify': None, 'attempts': []}
per0, RGB0, DEP0 = perceive(phase='perceive_start')
Image.fromarray(RGB0).save(os.path.join(ep_dir, 'start.jpg'))
try:
    _wr0 = obs.get('robot0_eye_in_hand_image')
    if _wr0 is not None:
        Image.fromarray(_wr0[::-1]).save(os.path.join(ep_dir, 'start_wrist.jpg'))
except Exception:
    pass
START_ID['attempts'].append({**identity_fields(per0), 'phase': 'perceive_start'})
ID_EVENTS.append({'where': 'start', **identity_fields(per0)})
# Always verify the locked grasp pixel against the language goal (overconfident wrong locks are the soup failure mode)
try:
    vrec, vimg = verify_grasp_identity(per0, RGB0)
    Image.fromarray(vimg).save(os.path.join(ep_dir, 'start_id_verify.jpg'))
    conf = vrec.get('is_language_goal_object')
    if conf is True or str(conf).lower() == 'true': confirmed = True
    elif conf is False or str(conf).lower() == 'false': confirmed = False
    else: confirmed = None
    START_ID['verify'] = {'confirmed': confirmed, 'marked_object_what': vrec.get('marked_object_what'),
                          'language_goal_where_if_different': vrec.get('language_goal_where_if_different'),
                          'reason': vrec.get('reason')}
    ID_EVENTS.append({'where': 'start_verify', **START_ID['verify']})
except Exception as ex:
    confirmed = None; START_ID['verify'] = {'confirmed': None, 'error': str(ex)[:200]}
need_relook = (not identity_ok(per0)) or (confirmed is not True)
if need_relook:
    # one forced relook before any motion; still no hard-coded appearance — language goal + current image only
    START_ID['forced_relook'] = True
    ctx = json.dumps({'reason': 'start identity check failed, unsure, or grasp-mark verify rejected',
                      'language_goal': LANG,
                      'previous': identity_fields(per0),
                      'verify': START_ID.get('verify'),
                      'instruction': 'Name the language-goal object in THIS image and lock grasp/object_center ONLY on that object; set matched_language_goal true only if sure. Do not lock a neighbour can/bottle/box.'})
    per0, RGB0, DEP0 = perceive(ctx, phase='perceive_start_relook')
    Image.fromarray(RGB0).save(os.path.join(ep_dir, 'start_id_relook.jpg'))
    START_ID['attempts'].append({**identity_fields(per0), 'phase': 'perceive_start_relook'})
    ID_EVENTS.append({'where': 'start_relook', **identity_fields(per0)})
    # one more mark-verify after relook (still before motion)
    try:
        vrec2, vimg2 = verify_grasp_identity(per0, RGB0, phase='perceive_start_verify2')
        Image.fromarray(vimg2).save(os.path.join(ep_dir, 'start_id_verify2.jpg'))
        conf2 = vrec2.get('is_language_goal_object')
        confirmed2 = True if conf2 is True or str(conf2).lower() == 'true' else (False if conf2 is False or str(conf2).lower() == 'false' else None)
        START_ID['verify_after_relook'] = {'confirmed': confirmed2, 'marked_object_what': vrec2.get('marked_object_what'),
                                          'language_goal_where_if_different': vrec2.get('language_goal_where_if_different'),
                                          'reason': vrec2.get('reason')}
        ID_EVENTS.append({'where': 'start_verify_after_relook', **START_ID['verify_after_relook']})
    except Exception as ex:
        START_ID['verify_after_relook'] = {'confirmed': None, 'error': str(ex)[:200]}
START_ID['final'] = identity_fields(per0)
_commit_keypoints(per0, DEP0, 'perceive_start_relook' if START_ID.get('forced_relook') else 'episode_start')
# Gate closes on start identity: matched_language_goal True AND mark-verify confirmed True. Unsure/false => not confirmed.
_vfinal = START_ID.get('verify_after_relook') if START_ID.get('forced_relook') else START_ID.get('verify')
_vconf = (_vfinal or {}).get('confirmed')
_start_ok = identity_ok(per0) and (_vconf is True)
set_goal_id(_start_ok, 'start', {'matched': START_ID['final'].get('matched_language_goal'), 'verify_confirmed': _vconf,
                                  'forced_relook': START_ID.get('forced_relook')})
START_ID['goal_id_confirmed'] = _start_ok
ID_EVENTS.append({'where': 'start_goal_id_gate', 'confirmed': _start_ok, 'matched': START_ID['final'].get('matched_language_goal'),
                  'verify_confirmed': _vconf})

eef = lambda: obs['robot0_eef_pos'].copy()
opening = lambda: float(obs['robot0_gripper_qpos'][0] - obs['robot0_gripper_qpos'][1])
yaw_now = lambda: Rot.from_quat(obs['robot0_eef_quat']).as_euler('xyz')[2]
YAW0 = yaw_now(); Q0 = Rot.from_quat(obs['robot0_eef_quat'])
EP = {'closed': False, 'n_same': 0, 'prev': 'none', 'step': 0, 'ever_contact': False, 'n_close': 0, 'released': False,
      'tcp_at_close': None, 'opening_at_close': None, 'uses': {}, 'closes': [], 'hist': [], 'tcp_hist': [],
      'result_hist': [], 'grip_hist': [], 'stage_fail_mark': 0,
      'last_motion': None, 'motion_before_path': None, 'blocked_motions': {},
      'motion_help_used': False,
      'approach_changes': 0, 'escapes': 0, 'escape_looks': 0, 'force_next_intent': None,
      'last_result': 'none', 'stall_feedbacks': 0, 'gate_jev_last_dec': None, 'gate_jev_calls': 0,
      'need_reapproach': False, 'reapproach_reason': None, 'reapproach_rev': 0,
      'clear_reapproach_n': 0, 'push_aside_n': 0, 'push_aside_approach_n': 0, 'last_stall_issue': None,
      'start_plan': None, 'start_plan_rec': None, 'task_todo': None, 'todo_calls': 0,
      'todo_target_id': None, 'todo_review_due': None, 'vision_serial': 0,
      'local_fb_family_hist': [], 'local_deadend_n': 0, 'local_deadend_global_relooks': 0,
      'last_local_deadend_reason': None, 'grasp_secured_once': False, 'in_local_deadend': False,
      'grasp_confirm': False, 'grasp_confirm_detail': None,
      'place_carry_reached_once': False, 'place_lower_stall_n': 0, 'place_lift_lower_cycles': 0,
      'place_cycle_armed': False, 'place_force_release': False, 'place_force_release_reason': None}
# After a keypoint refresh that moves the target away from the current TCP, block close until re-approach.
REAPPROACH_XY_M = float(os.environ.get('REAPPROACH_XY_M', '0.03'))

def _xy(v):
    a = np.asarray(v, float).reshape(-1)
    return a[:2]

def _reapproach_tol():
    return max(REAPPROACH_XY_M, 0.5 * float(PER.get('obj_w', 0.03) or 0.03))

def _tcp_near_grasp(tcp=None, grasp=None, tol=None):
    """True when TCP XY is close enough to the effective grasp to allow close."""
    try:
        tcp = eef() if tcp is None else np.asarray(tcp, float)
        if grasp is None:
            dxy = PATCH.get('grasp_dxy_objw') or [0.0, 0.0]
            grasp = PER['grasp'] + np.array([float(dxy[0]) * PER['obj_w'], float(dxy[1]) * PER['obj_w'],
                                            (PATCH['grasp_dz_objh'] + RSHAPE['dz']) * PER['obj_h']])
        tol = float(tol if tol is not None else _reapproach_tol())
        return float(np.linalg.norm(_xy(tcp) - _xy(grasp))) <= tol
    except Exception:
        return False

def _place_xy_tol():
    return max(PLACE_XY_M, 0.5 * float(PER.get('obj_w', 0.03) or 0.03))

def _tcp_near_place_region(tcp=None, tol=None):
    """True when held-object XY (tcp - grasp + objc) is over the place region — same geometry as above_place."""
    try:
        tcp = eef() if tcp is None else np.asarray(tcp, float)
        held_xy = _xy(tcp) - _xy(PER['grasp']) + _xy(PER['objc'])
        tol = float(tol if tol is not None else _place_xy_tol())
        return float(np.linalg.norm(held_xy - _xy(PER['place']))) <= tol
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
            tcp=list(np.asarray(eef(), float).round(4)),
            place=list(np.asarray(PER.get('place'), float).round(4)) if 'place' in PER else None)
    except Exception:
        pass
    return True

def _clear_reapproach_if_aligned(reason='aligned'):
    if not EP.get('need_reapproach'):
        return False
    if _tcp_near_grasp():
        EP['need_reapproach'] = False
        EP['reapproach_reason'] = None
        try:
            log(t=T[0], reapproach_cleared=True, reason=reason, goal_target=dict(GOAL_TARGET))
        except Exception:
            pass
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
    try:
        near = _tcp_near_grasp(grasp=new_g)
    except Exception:
        near = False
    shifted = False
    if old_grasp is not None:
        try:
            shifted = float(np.linalg.norm(_xy(new_g) - _xy(old_grasp))) >= _reapproach_tol()
        except Exception:
            shifted = True
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
    try:
        log(t=T[0], need_reapproach=True, reason=EP['reapproach_reason'], rev=EP['reapproach_rev'],
            old_grasp=list(np.asarray(old_grasp, float).round(4)) if old_grasp is not None else None,
            new_grasp=list(np.asarray(new_g, float).round(4)),
            tcp=list(np.asarray(eef(), float).round(4)), goal_target=dict(GOAL_TARGET))
    except Exception:
        pass
    return True

def namespace():
    l, r = contacts(); c = l and r; EP['ever_contact'] |= c
    tcp = eef()
    dxy = PATCH.get('grasp_dxy_objw') or [0.0, 0.0]
    g = PER['grasp'] + np.array([float(dxy[0]) * PER['obj_w'], float(dxy[1]) * PER['obj_w'],
                                 (PATCH['grasp_dz_objh'] + RSHAPE['dz']) * PER['obj_h']])
    ns = {'fine_unit': FS.FINE_UNIT_M, 'tcp': tcp, 'opening': opening(), 'left_contact': l, 'right_contact': r, 'contact': c, 'ever_contact': EP['ever_contact'],
          'closed_cmd': EP['closed'], 'released': EP['released'], 'seen_held': PER['seen_held'],
          'grasp': g, 'objc': PER['objc'], 'place': PER['place'], 'obj_h': PER['obj_h'], 'obj_w': PER['obj_w'], 'clear': PER['clear'], 'open_max': OPEN_MAX,
          'tcp_at_close': EP['tcp_at_close'] if EP['tcp_at_close'] is not None else tcp,
          'opening_at_close': EP['opening_at_close'] if EP['opening_at_close'] is not None else opening(),
          'n_same': EP['n_same'], 'step': EP['step'], 'n_close': EP['n_close'], 'n_relook': P['relook'], 'prev': EP['prev'],
          'last_result': EP.get('last_result', 'none'),
          'clear_reapproach_n': int(EP.get('clear_reapproach_n', 0)),
          'push_aside_n': int(EP.get('push_aside_n', 0)),
          'place_lower_stall_n': int(EP.get('place_lower_stall_n', 0)),
          'place_lift_lower_cycles': int(EP.get('place_lift_lower_cycles', 0)),
          'place_carry_reached_once': bool(EP.get('place_carry_reached_once')),
          'place_force_release': bool(EP.get('place_force_release'))}
    preds = {}
    for p in LIB.get('predicates', []):
        try: preds[p['name']] = bool(dsl.evaluate(p['expr'], {**ns, **preds}))
        except Exception: preds[p['name']] = False
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
    left, right = contacts()
    proprio = {
        'gripper_xyz_m': obs['robot0_eef_pos'].tolist(),
        'gripper_orientation_xyzw': obs['robot0_eef_quat'].tolist(),
        'finger_joint_positions_m': obs['robot0_gripper_qpos'].tolist(),
        'fingers_opening_m': opening(), 'gripper_closed_command': bool(EP['closed']),
        'left_finger_contact': bool(left), 'right_finger_contact': bool(right),
    }
    frames = {'front': obs['agentview_image'][::-1].copy()}
    if obs.get('robot0_eye_in_hand_image') is not None:
        frames['wrist'] = obs['robot0_eye_in_hand_image'][::-1].copy()
    return proprio, frames


def _jev_input(*, include_execution_history=False):
    proprio, frames = _current_observation()
    state, assets = JO.build_state(LANG, EP['task_todo'], proprio, frames, T[0], DONE[0])
    state['motion_context'] = _motion_context()
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
    # Store the exact observation sent to Jev; no privileged diagnostic fields.
    EP['vision_serial'] += 1
    paths = {}
    for name, pixels in frames.items():
        path = os.path.join(ep_dir, f'jev_{EP["vision_serial"]}_{name}.png')
        Image.fromarray(pixels).save(path)
        paths[name] = path
    log(t=T[0], jev_observation=True, state=state, frames=paths)
    return state, assets


def _save_task_todo(reason):
    with open(os.path.join(ep_dir, 'task_todo.json'), 'w') as output:
        json.dump(EP['task_todo'], output, indent=2)
    log(t=T[0], task_todo=EP['task_todo'], todo_reason=reason)


def _refresh_task_progress(intent=None, result=None):
    """Main-thread sensor refresh; checkpoints never retarget or call a model."""
    target_rev = GOAL_TARGET['rev']
    updated = TODO.refresh_progress(
        EP['task_todo'], EP['todo_target_id'], T[0],
        identity_confirmed=goal_id_confirmed(), grasp_ready=_transport_grasp_ready(),
        gripper_closed=EP['closed'], intent=intent, result=result,
        waypoints_changed=EP.get('todo_progress_target_rev', target_rev) != target_rev)
    EP['todo_progress_target_rev'] = target_rev
    if updated != EP['task_todo']:
        EP['task_todo'] = updated
        _save_task_todo('stage_progress')


def _retarget_todo():
    """Invalidate old grasp state; localize and verify the active todo object."""
    item = TODO.active_item(EP['task_todo'])
    new_id = item['id'] if item else None
    if EP.get('todo_target_id') == new_id:
        return
    _abandon_async_slow('todo_target_changed')
    old_id = EP.get('todo_target_id')
    EP['todo_target_id'] = new_id
    EP['recovery_history_start'] = len(EP.get('hist', []))
    EP['last_recovery_preparation'] = None
    EP['last_failed_motion'] = None
    EP['failed_motion_before_path'] = None
    EP['motion_before_path'] = None
    EP.update(force_next_intent=None, grasp_confirm=False, grasp_secured_once=False,
              ever_contact=False, place_carry_reached_once=False, place_lower_stall_n=0,
              place_lift_lower_cycles=0, place_cycle_armed=False, place_force_release=False,
              place_force_release_reason=None, uses={})
    _reset_patch('todo_target_changed')
    RSHAPE.update(tilt=None, tdir=None, yaw=None, dz=0.0)
    EP.update(tcp_at_close=None, opening_at_close=None, n_same=0,
              last_stall_issue=None, local_fb_family_hist=[], push_aside_approach_n=0)
    PER['seen_held'] = False
    set_goal_id(False, 'todo_target_changed', {'old': old_id, 'new': new_id})
    GOAL_TARGET['rev'] += 1
    if item is not None:
        p, rgb, dep = perceive(phase='todo_target')
        _commit_keypoints(p, dep, 'todo_target', held_now=False)
        verification, _ = verify_grasp_identity(p, rgb, phase='todo_target')
        matched = identity_ok(p) and verification.get('is_language_goal_object') is True
        set_goal_id(matched, 'todo_target_verified', verification)
        if not matched:
            EP['force_next_intent'] = 'relook'
    log(t=T[0], todo_target_switch=True, old_id=old_id, new_id=new_id)


def _review_task_todo(reason):
    """Only the main thread reviews and commits progress from current frames."""
    import coding_agent_slow as CAS
    _abandon_async_slow('todo_review')
    proprio, frames = _current_observation()
    paths = []
    for name, pixels in frames.items():
        path = os.path.join(ep_dir, f'todo_{EP["todo_calls"]}_{name}.png')
        Image.fromarray(pixels).save(path)
        paths.append(path)
    updated, rec = CAS.call_todo_review(todo=EP['task_todo'], proprioception=proprio,
                                       image_paths=paths, step=T[0], reason=reason)
    EP['todo_calls'] += 1
    P['astra_calls'] += 1
    P['astra_cost'] += rec['cost_usd']
    EP['task_todo'] = updated
    _save_task_todo(reason)
    _retarget_todo()

def _motion_orientation(shape=None):
    shape = SHAPE if shape is None else shape
    def pick(pk, sk, default):
        return next((v for v in (PATCH.get(pk), RSHAPE.get(sk), shape.get(sk)) if v is not None), default)
    return [float(pick('yaw_deg', 'yaw', PER['yaw'])),
            float(pick('tilt_deg', 'tilt', 0)), float(pick('tilt_dir_deg', 'tdir', 0))]


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


def _motion_context():
    ns, _ = namespace()
    return {'current_robot_contacts': robot_contacts(), 'last_motion': EP.get('last_motion'),
            'last_failed_motion': EP.get('last_failed_motion'),
            'recovery_validation': _recovery_context(),
            'close_position_check': MF.grasp_position(eef().tolist(), ns['grasp'].tolist(), ns['obj_w'], ns['obj_h']),
            'blocked_intents': list(EP.get('blocked_motions', {})),
            'slow_help_remaining': max(0, MAX_SLOW - P['slow']),
            'reserved_motion_diagnosis_remaining': int(FALLBACK and MAX_SLOW > 0 and not EP.get('motion_help_used'))}


def _save_approach_overlay(geometry):
    from PIL import ImageDraw
    im = Image.fromarray(obs['agentview_image'][::-1].copy())
    draw = ImageDraw.Draw(im)
    previous = None
    for label, key, color in [('TCP', 'tcp_xyz_m', 'cyan'),
                               ('approach', 'approach_xyz_m', 'yellow'),
                               ('grasp', 'descent_xyz_m', 'red')]:
        pixel = MF.project(geometry[key], CP, RM, F, S)
        if pixel is None: previous = None; continue
        x, y = pixel
        if previous is not None: draw.line([previous, (x, y)], fill=color, width=2)
        draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
        draw.text((x+5, y+5), label, fill=color)
        previous = (x, y)
    draw.text((5, 5), 'Proposed TCP route only; NOT full hand/arm clearance', fill='yellow')
    path = os.path.join(ep_dir, 'start_approach_overlay.png')
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
            'tcp_xyz_m': eef().tolist(), 'tcp_orientation_xyzw': obs['robot0_eef_quat'].tolist(),
            'approach_xyz_m': [grasp[0], grasp[1], grasp[2] + 2 * ns['obj_h']],
            'descent_xyz_m': grasp, 'place_xyz_m': ns['place'].tolist(),
            'commanded_yaw_tilt_direction_deg': _motion_orientation(),
            'maximum_finger_opening_m': OPEN_MAX, 'current_finger_opening_m': opening(),
            'object_width_estimate_m': ns['obj_w'], 'object_height_estimate_m': ns['obj_h'],
            'current_robot_contacts': robot_contacts(),
            'limitations': 'Finger opening is NOT total hand width. Palm/arm swept volume is not '
                          'collision-checked; occluded clearance must be reported as uncertain.'}


def servo(target, grip, gain=1.0, max_steps=60, tol=.006):
    last = eef(); stall = 0
    start, start_step = last.copy(), T[0]
    before_contacts = robot_contacts()
    during = {'environment': set(), 'self': set()}
    before_image = _combo_rgb()[0].copy()
    def finish(result):
        EP['last_motion'] = MF.motion_report(start.tolist(), eef().tolist(), target.tolist(),
            start_step, T[0], result, before_contacts, during, robot_contacts())
        EP['motion_before_path'] = None
        if result in ('stalled', 'timeout'):
            EP['last_failed_motion'] = EP['last_motion']
            path = os.path.join(ep_dir, f'motion_before_{start_step}_{T[0]}.png')
            Image.fromarray(before_image).save(path)
            EP['motion_before_path'] = path
            EP['failed_motion_before_path'] = path
        log(t=T[0], motion_feedback=EP['last_motion'])
        return result
    pick = lambda pk, sk, d: next((v for v in (PATCH[pk], RSHAPE[sk], SHAPE[sk]) if v is not None), d)
    yw = pick('yaw_deg', 'yaw', PER['yaw']); tl = float(np.clip(pick('tilt_deg', 'tilt', 0), 0, 30)); td = pick('tilt_dir_deg', 'tdir', 0)
    Rt = Rot.from_rotvec(np.deg2rad(tl) * np.array([np.cos(np.deg2rad(td + 90)), np.sin(np.deg2rad(td + 90)), 0])) * Rot.from_euler('z', np.deg2rad(yw)) * Q0
    for _ in range(max_steps):
        if T[0] >= HORIZON or DONE[0]: return finish('horizon_or_done')
        d = target - eef(); a = np.zeros(7); a[:3] = np.clip(d / .05 * 1.6, -1, 1) * float(np.clip(gain, .15, 1.0))
        er = (Rt * Rot.from_quat(obs['robot0_eef_quat']).inv()).as_rotvec(); a[3:6] = np.clip(er / .5 * 1.2, -1, 1)
        a[6] = grip; step(a)  # gripper binary open/close (EmbodiedEvo-style)
        contact_state = robot_contacts()
        during['environment'].update(contact_state['environment_contact_parts'])
        during['self'].update(contact_state['self_contact_parts'])
        if np.linalg.norm(target - eef()) < tol: return finish('reached')
        mv = np.linalg.norm(eef() - last); last = eef(); stall = stall + 1 if mv < .0005 else 0
        if stall >= 8: return finish('stalled')
    return finish('timeout')
def grip(g, n):
    for _ in range(n):
        if T[0] >= HORIZON or DONE[0]: break
        step([0, 0, 0, 0, 0, 0, g])


MAX_ESCAPE = int(os.environ.get('MAX_ESCAPE', '3'))
MAX_ESCAPE_LOOK = int(os.environ.get('MAX_ESCAPE_LOOK', '2'))

def _cmd_yaw_deg():
    """Current commanded grasp yaw (episode patch overrides perception)."""
    if PATCH.get('yaw_deg') is not None:
        return float(PATCH['yaw_deg'])
    if RSHAPE.get('yaw') is not None:
        return float(RSHAPE['yaw'])
    return float(PER.get('yaw', 0.0))

def _relative_yaw_delta_deg(ns):
    """Sign from observed finger contact asymmetry; else alternate. Magnitude is a relative step, not a scene angle."""
    l = bool(ns.get('left_contact')) if ns else False
    r = bool(ns.get('right_contact')) if ns else False
    if l and not r:
        sign = 1.0
    elif r and not l:
        sign = -1.0
    else:
        sign = 1.0 if (EP.get('approach_changes', 0) % 2 == 0) else -1.0
    # Relative quarter-turn/2 step about vertical; not the lucky scene-specific 90° from one success.
    return sign * 45.0

def _raise_tcp(scale=2.0, max_steps=40):
    """Lift gripper in place (open or closed as currently commanded)."""
    up_z = max(scale * float(PER.get('obj_h', 0.05)), 0.06)
    try:
        servo(eef() + np.array([0.0, 0.0, up_z]), -1.0 if not EP['closed'] else 1.0, 1.0, max_steps=max_steps, tol=0.01)
    except Exception:
        pass
    return up_z

def _nudge_grasp_xy(dx_objw, dy_objw):
    """Episode-only lateral grasp offset in object-width units (no hard-coded metres)."""
    cur = list(PATCH.get('grasp_dxy_objw') or [0.0, 0.0])
    cur[0] = float(np.clip(cur[0] + float(dx_objw), -1.5, 1.5))
    cur[1] = float(np.clip(cur[1] + float(dy_objw), -1.5, 1.5))
    PATCH['grasp_dxy_objw'] = cur
    return cur

STALL_FB_ACTIONS = (
    'raise_redescend', 'lateral_recenter', 'change_approach_yaw',
    'confirm_contact', 'clear_reapproach', 'push_aside',
)
NEIGHBOUR_STALL_ISSUES = ('blocked_by_neighbour', 'tip_push', 'object_pushed_by_descent')

LOCAL_LOOP_FAMILY = frozenset({
    'push_aside', 'clear_reapproach', 'raise_redescend', 'lateral_recenter',
})
# Same-family local thrash without progress → force retreat + global front relook + sync slow replan.

def _transport_grasp_ready():
    """Confirmed target grasp with current sensor support, not a historical latch."""
    if not (EP.get('closed') and EP.get('grasp_confirm') and goal_id_confirmed()):
        return False
    left, right = contacts()
    return bool((left and right) or ((left or right) and opening() >= 0.10 * OPEN_MAX))


def _action_block_reason(name, it, ns):
    held = _transport_grasp_ready()
    if name == 'close_gripper' and not EP['closed']:
        if not goal_id_confirmed(): return 'close_refused_identity'
        if not MF.grasp_position(eef().tolist(), ns['grasp'].tolist(), ns['obj_w'], ns['obj_h'])['ready']:
            return CLOSE_NOT_READY
    if held and (it.get('fingers') == 'open' or name in ('clear_reapproach', 'change_approach_yaw')):
        if name != 'open_gripper': return 'motion_not_executed:preserve_confirmed_grasp'
        if not MF.release_position(eef().tolist(), ns['place'].tolist(), ns['clear']):
            return 'release_not_executed:not_at_destination'
    if name in EP.get('blocked_motions', {}):
        if name == 'back_off':
            target = (eef() + [0, 0, max(.04, float(PER['obj_h']))]).tolist()
        else:
            target = np.asarray(dsl.evaluate(it['refpoint'], ns), float).tolist()
        shape = {key: float(dsl.evaluate(str(it[field]), ns)) if field in it else None
                 for key, field in [('yaw', 'yaw_deg'), ('tilt', 'tilt_deg'), ('tdir', 'tilt_dir_deg')]}
        orientation = _motion_orientation() if name == 'back_off' else _motion_orientation(shape)
        if MF.same_motion(EP['blocked_motions'][name], target, orientation):
            return 'motion_not_executed:unchanged_failed_waypoint'
    return None


def _acting_menu(intents, ns=None):
    ns = ns if ns is not None else namespace()[0]
    ready = _transport_grasp_ready()
    return {name: intent for name, intent in intents.items()
            if (name != 'close_gripper' or goal_id_confirmed())
            and (name not in TRANSPORT_INTENTS or ready)
            and not _action_block_reason(name, intent, ns)
            and (name not in ASK_ASTRA_NAMES or (FALLBACK and P['slow'] < MAX_SLOW))
            and (name != 'relook' or P['relook'] < MAX_RELOOK)}


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
        op = float(opening())
        open_max = float(OPEN_MAX) if OPEN_MAX else 0.08
        l, r = contacts()
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
                grip(-1, 12)
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
                grip(-1, 12)
                EP['released'] = True
            except Exception:
                EP['closed'] = False
            EP['force_next_intent'] = 'move_above_object'
            return False, detail
        # Short lift probe while commanding closed — drop/collapse ⇒ not secured
        oh = float(PER.get('obj_h', 0.05) or 0.05)
        lift_m = float(max(0.5 * oh, 0.025))
        detail['lift_probe_m'] = round(lift_m, 4)
        tcp0 = eef().copy()
        try:
            servo(tcp0 + np.array([0.0, 0.0, lift_m]), 1.0, 1.0, max_steps=30, tol=0.008)
        except Exception as ex:
            detail['lift_err'] = str(ex)[:120]
        op2 = float(opening())
        l2, r2 = contacts()
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
            lift_name = 'lift' if 'lift' in {i['name'] for i in LIB.get('intents', [])} else 'move_above_object'
            EP['force_next_intent'] = lift_name
            detail['force_next'] = lift_name
        else:
            try:
                EP['closed'] = False
                grip(-1, 15)
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

def _local_deadend_should_trigger(pending_action=None):
    """True when local view/actions have no useful solution — force global relook + sync slow.
    General (no object-name hacks). Skips if grasp already secured or global budget exhausted."""
    if EP.get('grasp_secured_once') or _grasp_secured_now():
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
    try:
        if EP.get('closed') and not (bool(PER.get('seen_held')) or EP.get('grasp_confirm') or EP.get('grasp_secured_once')):
            EP['closed'] = False
            grip(-1, 12)
    except Exception:
        pass
    raised = _raise_tcp(scale, max_steps=50)
    # Extra lift toward a safer clearance above current grasp estimate
    try:
        tcp = eef()
        oh = float(PER.get('obj_h', 0.05) or 0.05)
        tgt = tcp.copy()
        tgt[2] = max(float(tcp[2]), float((PER.get('grasp') or tcp)[2]) + max(3.0 * oh, 0.12))
        servo(tgt, -1.0 if not EP['closed'] else 1.0, 1.0, max_steps=40, tol=0.015)
    except Exception:
        pass
    return raised

def _global_front_relook(reason):
    """Force keypoint refresh from current agentview (front) after retreat — not wrist-only."""
    detail = {'reason': reason, 'matched': None, 'error': None}
    ns, _ = namespace()
    ap = _relook_approach_context(reason)
    ctx = json.dumps({
        'reason': reason,
        'language_goal': LANG,
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
        Image.fromarray(rgb).save(os.path.join(ep_dir, f'{tag}.jpg'))
        # Also save combo (front|wrist) for slow-think stills
        try:
            both, _, _ = _combo_rgb()
            Image.fromarray(both).save(os.path.join(ep_dir, f'{tag}_combo.jpg'))
        except Exception:
            pass
        detail.update(matched=matched, phase=phase, tag=tag, identity=idf)
        return detail, dep
    except Exception as ex:
        detail['error'] = str(ex)[:300]
        return detail, None

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
    rel, dep = _global_front_relook(reason)
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
    try:
        # Enrich last markers so _slow_ctx / mock see the deadend
        EP['last_stall_issue'] = None  # cleared; slow should replan from global stills
        sr, so = _sync_slow(intent, result, crit=crit)
        detail['slow_result'] = sr
        detail['slow_detail'] = {
            'issue': (so or {}).get('issue') if isinstance(so, dict) else None,
            'intervene': (so or {}).get('intervene') if isinstance(so, dict) else None,
            'plan_done': (so or {}).get('plan_done') if isinstance(so, dict) else None,
            'crit': (so or {}).get('crit') if isinstance(so, dict) else crit,
            'mode': (so or {}).get('mode') if isinstance(so, dict) else None,
        }
        if sr == 'ok':
            _consume_help_budget(crit)
        # If sync slow unavailable (cap/disabled), at least force re-approach from above
        if sr in ('slow_cap', 'slow_cap_drop', None) or (isinstance(sr, str) and sr.startswith('sync_error')):
            if not EP.get('force_next_intent'):
                EP['force_next_intent'] = 'move_above_object'
            # Soft fallback: raise_redescend once without push
            try:
                ns_fb, _ = namespace()
                _apply_stall_feedback(ns_fb, f'{crit}:fallback_raise', action='raise_redescend')
            except Exception as ex:
                detail['fallback_raise_error'] = str(ex)[:200]
    except NameError:
        # _sync_slow not yet bound (should not happen at runtime) — still force re-approach
        EP['force_next_intent'] = 'move_above_object'
        detail['slow_result'] = 'sync_slow_unavailable'
    except Exception as ex:
        detail['slow_result'] = f'sync_error:{ex}'
        EP['force_next_intent'] = 'move_above_object'
    if not EP.get('force_next_intent'):
        EP['force_next_intent'] = 'move_above_object'
    CUR.update(fb=f"DEADEND→GLOBAL+SYNC_SLOW:{reason}"[:170], fb_t=T[0],
               slow=f"deadend_global:{detail.get('slow_result')}"[:160], slow_t=T[0])
    # detail already includes local_deadend_global_relook=True — do not pass it twice
    log(t=T[0], counts=counts, goal_id_confirmed=goal_id_confirmed(), goal_target=dict(GOAL_TARGET),
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
    yaw = np.deg2rad(float(_cmd_yaw_deg()))
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
    try:
        tcp = np.asarray(ns.get('tcp', eef()), float)
        objc = np.asarray(ns.get('objc', PER.get('objc', tcp)), float)
        delta = tcp[:2] - objc[:2]
        if float(np.linalg.norm(delta)) > 1e-4:
            return delta / (float(np.linalg.norm(delta)) + 1e-9)
    except Exception:
        pass
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
    try:
        id_ok = bool(goal_id_confirmed())
    except Exception:
        id_ok = False
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
    """Raise then rotate approach about vertical relative to current command. Mutates PATCH; physical raise."""
    EP['approach_changes'] = int(EP.get('approach_changes', 0)) + 1
    delta = _relative_yaw_delta_deg(ns or {})
    cur = _cmd_yaw_deg()
    new_yaw = float(np.clip(cur + delta, -90.0, 90.0))
    # If clip removed the change, flip the other way once.
    if abs(new_yaw - cur) < 1.0:
        new_yaw = float(np.clip(cur - delta, -90.0, 90.0))
    PATCH['yaw_deg'] = new_yaw
    # Shallower fingertip target (positive dz = higher grasp) — observed stall often needs less depth.
    PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.15, -0.5, 0.5))
    _note_patch_mutation('approach_change', {'reason': reason, 'yaw_to': new_yaw})
    _raise_tcp(2.0)
    log(t=T[0], approach_change=True, reason=reason, yaw_from=round(cur, 1), yaw_delta=round(delta, 1),
        yaw_to=round(new_yaw, 1), grasp_dz_objh=PATCH['grasp_dz_objh'], approach_changes=EP['approach_changes'])
    return {'yaw_from': cur, 'yaw_delta': delta, 'yaw_to': new_yaw, 'grasp_dz_objh': PATCH['grasp_dz_objh']}


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
    try:
        id_ok = bool(goal_id_confirmed())
    except Exception:
        id_ok = False
    if not id_ok:
        detail['error'] = 'push_aside_refused_identity'
        _raise_tcp(2.0)
        EP['force_next_intent'] = 'move_above_object'
        return detail
    # Do not shove while holding a confirmed / verified grasp; skip neighbour push.
    if EP.get('grasp_confirm') or (EP.get('grasp_secured_once') and EP.get('closed')):
        detail['error'] = 'push_aside_skipped_grasp_secured'
        detail['grasp_confirm'] = bool(EP.get('grasp_confirm'))
        EP['force_next_intent'] = 'lift' if 'lift' in {i['name'] for i in LIB.get('intents', [])} else 'move_above_object'
        return detail
    holding_verified = bool(EP.get('closed')) and bool(PER.get('seen_held')) and id_ok
    if EP.get('closed') and not holding_verified:
        detail['error'] = 'push_aside_refused_unverified_hold'
        try:
            EP['closed'] = False; grip(-1, 15); EP['released'] = True
        except Exception:
            pass
        _raise_tcp(2.0)
        EP['force_next_intent'] = 'move_above_object'
        return detail
    if holding_verified:
        detail['error'] = 'push_aside_skipped_holding_goal'
        EP['force_next_intent'] = 'lift' if 'lift' in {i['name'] for i in LIB.get('intents', [])} else 'move_above_object'
        return detail

    # Keep grip OPEN during neighbour push (safer than closing on a non-goal).
    try:
        if EP.get('closed'):
            EP['closed'] = False; grip(-1, 15)
        else:
            grip(-1, 8)
    except Exception:
        pass

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
    except Exception:
        dist_objw = 1.0

    used_astra = False
    v = None
    start = None
    if st.get('blocker_px') is not None and dep is not None:
        try:
            blk = bp(st['blocker_px'], dep)
            dn = st.get('direction', 'away_from_target')
            if dn == 'away_from_target':
                vv = blk - PER['objc']
            else:
                cv = {'toward_image_left': [-1, 0, 0], 'toward_image_right': [1, 0, 0],
                      'toward_camera': [0, -1, 0], 'away_from_camera': [0, 1, 0]}.get(dn)
                vv = (RM @ np.array(cv, float)) if cv is not None else (blk - PER['objc'])
            vv = np.asarray(vv, float); vv[2] = 0
            v = vv / (np.linalg.norm(vv) + 1e-9)
            zc = max(float(blk[2]) - 0.5 * oh, float(PER['grasp'][2]) - 0.3 * oh)
            start = blk - v * (1.3 * ow); start[2] = zc
            used_astra = True
            detail['blocker_mode'] = 'astra_px'
            detail['direction'] = dn
        except Exception as ex:
            detail['astra_blocker_error'] = str(ex)
            used_astra = False

    if not used_astra:
        v2 = _push_dir_xy_from_ns(ns)
        v = np.array([float(v2[0]), float(v2[1]), 0.0])
        tcp = np.asarray(ns.get('tcp', eef()), float)
        objc = np.asarray(ns.get('objc', PER.get('objc', tcp)), float)
        # Start just outside the grasp footprint along push dir (away from goal centre).
        start = objc.copy(); start[:2] = objc[:2] + v[:2] * (0.7 * ow)
        start[2] = float(PER.get('grasp', tcp)[2]) if PER.get('grasp') is not None else float(tcp[2])
        detail['blocker_mode'] = 'heuristic_open_axis'
        detail['dir_xy'] = [round(float(v[0]), 4), round(float(v[1]), 4)]

    L = dist_objw * ow
    g_open = -1.0
    try:
        # Approach from above, descend open, shove laterally, raise clear.
        servo(start + np.array([0.0, 0.0, 2.0 * oh]), g_open, 1.0, max_steps=40, tol=0.01)
        servo(start, g_open, 0.5, max_steps=40, tol=0.01)
        r = servo(start + v * (L + 1.0 * ow), g_open, 0.4, max_steps=50, tol=0.015)
        servo(eef() + np.array([0.0, 0.0, 2.0 * oh]), g_open, 1.0, max_steps=40, tol=0.01)
        detail['servo'] = r
        detail['pushed'] = True
        detail['distance_objw'] = dist_objw
    except Exception as ex:
        detail['error'] = f'push_aside_motion:{ex}'
        try:
            _raise_tcp(2.5)
        except Exception:
            pass

    # Always reopen (should already be open) and force re-approach before any close.
    try:
        EP['closed'] = False; grip(-1, 10)
    except Exception:
        EP['closed'] = False
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
    try:
        _note_grasp_progress(ns)
    except Exception:
        pass
    # Confirmed grasp: never undo with neighbour/local-loop stall feedback — go lift/carry.
    if EP.get('grasp_confirm') or (EP.get('grasp_secured_once') and EP.get('closed') and _grasp_secured_now(ns)):
        lift_name = 'lift' if 'lift' in {i['name'] for i in LIB.get('intents', [])} else 'move_above_object'
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
    before_recovery = {'tcp_xyz_m': eef().tolist(), 'commanded_orientation_deg': _motion_orientation()}
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
        tcp = np.asarray(ns.get('tcp', eef()), float)
        objc = np.asarray(ns.get('objc', PER.get('objc', tcp)), float)
        delta = objc[:2] - tcp[:2]
        nrm = float(np.linalg.norm(delta)) + 1e-9
        # Shift grasp estimate toward object centre (object-width units); raise then re-approach.
        dx, dy = float(delta[0] / nrm) * 0.35, float(delta[1] / nrm) * 0.35
        detail['grasp_dxy_objw'] = _nudge_grasp_xy(dx, dy)
        PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.1, -0.5, 0.5))
        try:
            tgt = eef().copy(); tgt[:2] = tcp[:2] + np.array([dx, dy]) * ow; tgt[2] = tcp[2] + max(float(PER.get('obj_h', 0.05)), 0.04)
            servo(tgt, grip, 1.0, max_steps=40, tol=0.01)
        except Exception:
            _raise_tcp(2.0)
        detail['grasp_dz_objh'] = PATCH['grasp_dz_objh']
        EP['force_next_intent'] = 'move_above_object'

    elif action == 'change_approach_yaw':
        ch = _apply_relative_approach_change(ns, reason)
        detail.update(ch)
        EP['force_next_intent'] = 'move_above_object'

    elif action == 'confirm_contact':
        # Require bilateral contact before treating the grasp as secure / before lift.
        if both and not EP['closed']:
            if EP.get('need_reapproach') and not _tcp_near_grasp():
                EP['force_next_intent'] = 'move_above_object'
                detail['confirm'] = 'both_contact_but_need_reapproach'
            else:
                EP['force_next_intent'] = 'close_gripper'
                detail['confirm'] = 'both_contact_close'
        elif EP['closed'] and not both:
            EP['force_next_intent'] = 'reopen_and_retry' if 'reopen_and_retry' in {i['name'] for i in LIB['intents']} else 'move_above_object'
            PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.15, -0.5, 0.5))
            _raise_tcp(2.0)
            detail['confirm'] = 'closed_no_bilateral_reopen'
        elif (l or r) and not both and not EP['closed']:
            # Partial contact: shallow + relative yaw, then close only after re-approach.
            _apply_relative_approach_change(ns, reason + ':partial_contact')
            detail['confirm'] = 'partial_contact_reapproach'
            EP['force_next_intent'] = 'move_above_object'
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
        if l and not r:
            detail['grasp_dxy_objw'] = _nudge_grasp_xy(0.4, 0.0)  # dimensionless lateral bias away from left pad
        elif r and not l:
            detail['grasp_dxy_objw'] = _nudge_grasp_xy(-0.4, 0.0)
        else:
            # No pad signal: small recenter toward object centre if available.
            tcp = np.asarray(ns.get('tcp', eef()), float)
            objc = np.asarray(ns.get('objc', PER.get('objc', tcp)), float)
            delta = objc[:2] - tcp[:2]; nrm = float(np.linalg.norm(delta)) + 1e-9
            detail['grasp_dxy_objw'] = _nudge_grasp_xy(0.25 * delta[0] / nrm, 0.25 * delta[1] / nrm)
        # Shallower fingertip target for vertical re-approach from clearance (no yaw mutation).
        PATCH['grasp_dz_objh'] = float(np.clip(float(PATCH.get('grasp_dz_objh') or 0.0) + 0.15, -0.5, 0.5))
        detail['grasp_dz_objh'] = PATCH['grasp_dz_objh']
        detail['yaw_preserved'] = _cmd_yaw_deg()
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
    _note_patch_mutation(f'stall_feedback:{action}', detail)
    if action in ('clear_reapproach', 'change_approach_yaw', 'raise_redescend', 'lateral_recenter'):
        EP['last_recovery_preparation'] = {
            'step': T[0], 'action': action, 'before': before_recovery,
            'after': {'tcp_xyz_m': eef().tolist(), 'commanded_orientation_deg': _motion_orientation()},
            'status': 'preparation_attempted_descent_unverified',
        }
    try:
        _record_local_stall_family(action)
    except Exception:
        pass
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

def _escape_look(reason):
    """One extra identity look that does NOT consume the normal relook budget (separate escape budget)."""
    if EP.get('escape_looks', 0) >= MAX_ESCAPE_LOOK:
        return 'escape_look_budget_exhausted'
    EP['escape_looks'] = int(EP.get('escape_looks', 0)) + 1
    ns, _ = namespace()
    ap = _relook_approach_context(reason)
    ctx = json.dumps({'reason': reason, 'language_goal': LANG, 'escape_look': EP['escape_looks'],
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
        Image.fromarray(rgb).save(os.path.join(ep_dir, f'escape{EP["escape_looks"]}.jpg'))
        return 'escape_looked_matched' if matched else 'escape_looked_unmatched'
    except Exception as e:
        return f'escape_look_error:{e}'

def _escape_identity_deadlock(reason):
    """Non-no-op escape when identity refuse + relook budget can no longer help: raise, relative approach change, escape look.
    Never leave the arm frozen selecting close_gripper until decision_cap."""
    EP['escapes'] = int(EP.get('escapes', 0)) + 1
    ns, _ = namespace()
    detail = {'reason': reason, 'escape_n': EP['escapes'], 'relooks': P['relook'], 'close_refusals': GOAL_ID.get('close_refusals', 0)}
    if EP['escapes'] > MAX_ESCAPE:
        # Still move (raise) so we do not freeze, but stop burning Astra.
        try:
            up_z = max(2.0 * float(PER.get('obj_h', 0.05)), 0.06)
            servo(eef() + np.array([0.0, 0.0, up_z]), -1.0 if not EP['closed'] else 1.0, 1.0, max_steps=30, tol=0.01)
        except Exception:
            pass
        EP['force_next_intent'] = 'retreat' if 'retreat' in {i['name'] for i in LIB['intents']} else 'move_above_object'
        log(t=T[0], identity_escape='raise_only_cap', **detail)
        return 'identity_escape_cap'
    ch = _apply_stall_feedback(ns, reason)  # multi-action (not yaw-only)
    detail['stall_feedback'] = ch
    lr = _escape_look(reason)
    detail['escape_look'] = lr
    # Prefer re-approach from above after escape; never force another immediate close.
    if not EP.get('force_next_intent'):
        EP['force_next_intent'] = 'move_above_object'
    CUR.update(fb=f'ESCAPE:{reason}->{lr} fb->{ch.get("action")}', fb_t=T[0])
    log(t=T[0], identity_escape=True, **detail)
    ID_EVENTS.append({'where': 'identity_escape', **detail})
    return f'identity_escape:{lr}'

def _relook_approach_context(reason):
    """Stall + current commanded approach for relook/escape prompts. Do not silently revert a working yaw."""
    try:
        yaw = float(_cmd_yaw_deg())
    except Exception:
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
    ns, _ = namespace()
    ap = _relook_approach_context(reason)
    ctx = json.dumps({'reason': reason, 'language_goal': LANG, 'gripper_closed_command': ns['closed_cmd'], 'fingers_opening_m': round(ns['opening'], 4),
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
            ctx2 = json.dumps({'reason': 'library relook identity failed or unsure', 'language_goal': LANG, 'previous': idf,
                               'last_result': ap['last_result'], 'current_approach_yaw_deg': ap['current_approach_yaw_deg'],
                               'current_tilt_deg': ap['current_tilt_deg'],
                               'instruction': 'Name the language-goal object from the instruction + image; do NOT lock a neighbour. '
                                              'Set matched_language_goal honestly. ' + ap['instruction_extra']})
            p, rgb, dep = perceive(ctx2, phase='relook_identity'); idf = identity_fields(p)
            ID_EVENTS.append({'where': 'library_relook_identity', 'reason': reason, **idf})
            Image.fromarray(rgb).save(os.path.join(ep_dir, f'relook{P["relook"]}_id.jpg'))
        _commit_keypoints(p, dep, 'relook', held_now=bool(p.get('object_held_by_gripper')) and EP['closed'])
        matched = identity_ok(p)
        set_goal_id(matched, 'relook', idf)
        Image.fromarray(rgb).save(os.path.join(ep_dir, f'relook{P["relook"]}.jpg'))
        return 'relooked_matched' if matched else 'relooked_unmatched'
    except Exception as e: return 'relook_error'

def _remember_fine_step(rep):
    EP['last_fine_step'] = rep
    EP.setdefault('fine_step_events', []).append(dict(rep))

def _execute_fine_step(it, ns):
    """Servo one clamped world-axis correction. Does not change the gripper command."""
    name = it.get('name', 'fine_step')
    closed_before = bool(EP.get('closed'))
    tcp_now = np.asarray(eef(), float).reshape(3)
    try:
        raw = np.asarray(dsl.evaluate(it['refpoint'], ns), float).reshape(3)
    except Exception:
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
        except Exception:
            gain = 1.0
        grip = 1 if closed_before else -1
        r = servo(tgt, grip, gain, max_steps=FS.FINE_STEP_MAX_STEPS, tol=FS.FINE_STEP_TOL_M)
    if r in ('reached', 'at_target'):
        _clear_reapproach_if_aligned('fine_step')
    rep = FS.fine_step_report(name, r, eef(), delta, bool(EP.get('closed')))
    _remember_fine_step(rep)
    return r

def execute(name, ns):
    it = {i['name']: i for i in LIB['intents']}[name]
    act = it.get('action', 'move'); fing = it.get('fingers', 'keep')
    if act in ASK_ASTRA_NAMES or name in ASK_ASTRA_NAMES:
        return do_ask_astra(f'jev_intent:{name}')
    if act == 'relook': return do_relook('library rule chose relook')
    if act == 'finish':
        EP['todo_review_due'] = 'finish_requested'
        return 'finish' if DONE[0] else 'skip_finish'
    if name in TRANSPORT_INTENTS and not _transport_grasp_ready():
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        return f'{name}_not_executed:grasp_unconfirmed'
    blocked = _action_block_reason(name, it, ns)
    if blocked and blocked != 'close_refused_identity':
        log(t=T[0], action_not_executed=name, reason=blocked)
        return blocked
    if act == 'fine_step': return _execute_fine_step(it, ns)
    if name in MOTION_RECOVERIES:
        if name == 'back_off':
            target = eef().copy(); target[2] += max(.04, float(PER['obj_h']))
            r = servo(target, 1 if EP['closed'] else -1, .5, max_steps=40)
            if r == 'reached': EP['blocked_motions'] = {}
            elif r in ('stalled', 'timeout'):
                EP.setdefault('blocked_motions', {})[name] = {
                    'target_xyz_m': target.tolist(), 'orientation_deg': _motion_orientation()}
            return r
        # Expired hold latches must not suppress recovery after actual contact/identity loss.
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        before_step = T[0]
        detail = _apply_stall_feedback(ns, 'acting_motion_recovery', action=name)
        motion = EP.get('last_motion') or {}
        if motion.get('start_step', -1) >= before_step and motion.get('result') in ('stalled', 'timeout'):
            return motion['result']
        return 'recovery_prepared:' + str(detail.get('action')) + ':descent_unverified'
    if name == 'close_gripper' and EP['closed']:
        if _transport_grasp_ready():
            return 'already_closed'
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        return GRASP_FAILED
    if fing == 'close' and not EP['closed']:
        # HARD GATE: null/false/unsure goal identity ⇒ force relook, refuse close until matched. Uncertain ≠ proceed.
        # When relook budget is exhausted, DO NOT freeze: raise + relative approach change + escape look.
        if not goal_id_confirmed():
            GOAL_ID['close_refusals'] = GOAL_ID.get('close_refusals', 0) + 1
            if P['relook'] >= MAX_RELOOK:
                rr = 'relook_budget_exhausted'
                ID_EVENTS.append({'where': 'close_refused', 'relook_result': rr, 'goal_id': dict(GOAL_ID),
                                  'identity': PER.get('identity'), 'escaped': True})
                log(t=T[0], close_refused=True, relook_result=rr, goal_id_confirmed=False,
                    identity=PER.get('identity'), escaped=True)
                return _escape_identity_deadlock(f'close_refused:{rr}')
            rr = do_relook('close refused: goal identity not confirmed (null/false/unsure ≠ proceed)')
            ID_EVENTS.append({'where': 'close_refused', 'relook_result': rr, 'goal_id': dict(GOAL_ID),
                              'identity': PER.get('identity')})
            log(t=T[0], close_refused=True, relook_result=rr, goal_id_confirmed=goal_id_confirmed(),
                identity=PER.get('identity'))
            if not goal_id_confirmed():
                # Budget just hit during this relook, or repeated unmatched closes → escape instead of silent refuse.
                if rr == 'relook_budget_exhausted' or P['relook'] >= MAX_RELOOK or GOAL_ID['close_refusals'] >= 3:
                    return _escape_identity_deadlock(f'close_refused:{rr}')
                return f'close_refused_identity:{rr}'
            # relook matched — but a NEW/shifted target must be re-approached before close (B03)
        # HARD GATE: after target refresh, do not close while TCP is still over the old place.
        if EP.get('need_reapproach') and not _tcp_near_grasp():
            EP['force_next_intent'] = 'move_above_object'
            try:
                log(t=T[0], close_blocked_reapproach=True, reason=EP.get('reapproach_reason'),
                    rev=EP.get('reapproach_rev'), goal_id_confirmed=goal_id_confirmed(),
                    tcp=list(np.asarray(eef(), float).round(4)),
                    grasp=list(np.asarray(PER.get('grasp'), float).round(4)) if 'grasp' in PER else None)
            except Exception:
                pass
            ID_EVENTS.append({'where': 'close_blocked_reapproach', 'reason': EP.get('reapproach_reason'),
                              'rev': EP.get('reapproach_rev'), 'goal_id': dict(GOAL_ID)})
            return 'close_blocked_need_reapproach'
        # Identity/relook may have moved the grasp estimate since the menu was built.
        current_ns, _ = namespace()
        if not MF.grasp_position(eef().tolist(), current_ns['grasp'].tolist(),
                                 current_ns['obj_w'], current_ns['obj_h'])['ready']:
            return CLOSE_NOT_READY
        _clear_reapproach_if_aligned('pre_close')
        EP['closed'] = True; grip(1, 20); EP['n_close'] += 1; EP['released'] = False
        EP['tcp_at_close'] = eef(); EP['opening_at_close'] = opening()
        EP['closes'].append({'xyz': [round(float(x), 3) for x in eef()], 'opening_after_close': round(opening(), 4),
                             'both_contact': all(contacts()), 'goal_id_confirmed': True})
        try:
            secured, detail = _confirm_grasp_after_close()
        except Exception as exc:
            secured = False
            log(t=T[0], grasp_confirmation_error=True, error_type=type(exc).__name__)
        if not secured:
            EP['grasp_confirm'] = False
            EP['grasp_secured_once'] = False
            PER['seen_held'] = False
            return GRASP_FAILED
        return 'closed'
    if fing == 'open' and EP['closed']:
        EP['closed'] = False; grip(-1, 15); EP['released'] = True; EP['just_released'] = True
        EP['grasp_confirm'] = False
        EP['grasp_secured_once'] = False
        PER['seen_held'] = False
        EP['todo_review_due'] = 'gripper_released'
    if name == 'reopen_and_retry':
        return 'reopened'  # Opening is separate from approaching a potentially stale point.
    try: tgt = np.asarray(dsl.evaluate(it['refpoint'], ns), float).reshape(3)
    except Exception as e: return 'refpoint_error'
    def ev(k):
        try: return float(dsl.evaluate(str(it[k]), ns)) if k in it else None
        except Exception: return None
    SHAPE.update(tilt=ev('tilt_deg') or 0.0, tdir=ev('tilt_dir_deg') or 0.0, yaw=ev('yaw_deg'))
    try: gain = float(dsl.evaluate(str(it.get('gain', '1')), ns))
    except Exception: gain = 1.0
    if np.linalg.norm(tgt - eef()) < .006:
        _clear_reapproach_if_aligned('at_target')
        return 'at_target'
    r = servo(tgt, 1 if EP['closed'] else -1, gain)
    if r in ('stalled', 'timeout'):
        EP.setdefault('blocked_motions', {})[name] = {
            'target_xyz_m': tgt.tolist(), 'orientation_deg': _motion_orientation()}
    elif r == 'reached':
        EP.get('blocked_motions', {}).pop(name, None)
        _clear_reapproach_if_aligned('reached')
    return r

def push_aside(st, dep):
    """Episode-only neighbour shove. Identity-gated; prefers OPEN fingers; forces re-approach before close.
    Direction from blocker_px relative to target/camera — no fixed scene yaw / object names."""
    ns, _ = namespace()
    det = _execute_push_aside(ns, reason='plan_push_aside', plan_step=st, dep=dep)
    if det.get('error'):
        return det['error']
    return str(det.get('servo') or 'pushed')



# Codex slow-think: async watch most of the time; SYNC pause at critical points.
# Async results older than 1 completed intent are discarded. Never watches mid-episode video.
# Per-job ownership: each async request gets a unique job_id + private result_box.
# Abandoned workers may finish later but MUST NOT write into a newer job's slot (B01).
SLOW_JOB = {'thread': None, 'snap_dec': None, 'dep': None, 'both': None, 'intent': None, 'result': None,
            'mode': None, 'crit': None, 'out': None, 'err': None, 'rec': None, 'parsed': None,
            'job_id': None, 'result_box': None, 'target_rev': None, 'closed': None,
            'n_close': None, 'goal_confirmed': None}
SLOW_JOB_SEQ = [0]  # monotonic generation for job_id
SLOW_STALE_INTENTS = 1  # discard async if more than this many intents completed since snapshot
SLOW_ORPHAN_WAITERS = []
SLOW_ORPHAN_COSTS = queue.SimpleQueue()  # completed abandoned boxes; consumed only on main thread

def _next_slow_job_id():
    SLOW_JOB_SEQ[0] += 1
    return int(SLOW_JOB_SEQ[0])

def _new_slow_result_box(job_id):
    return {'job_id': int(job_id), 'parsed': None, 'rec': None, 'err': None, 'done': False}

def _combo_rgb():
    rgb, dep = rgbd(); wr = obs.get('robot0_eye_in_hand_image')
    both = np.concatenate([rgb, np.asarray(Image.fromarray(wr[::-1]).resize((S, S)))], 1) if wr is not None else rgb
    return both, rgb, dep

def _apply_episode_plan(p, dep, source='slow'):
    """Apply episode-only plan steps from Codex.
    Single shared goal: keypoint refresh overwrites PER+clears position PATCH; approach yaw/tilt preserved unless new yaw explicit; adjust/stall mutates PATCH only.
    On stall with identity confirmed: relook must run perception; never treat a log marker as a target refresh."""
    def f(d_, k, lo, hi, d):
        try: return float(np.clip(float(d_[k]), lo, hi))
        except Exception: return d
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
        lift_name = 'lift' if 'lift' in {i['name'] for i in LIB.get('intents', [])} else 'move_above_object'
        EP['force_next_intent'] = lift_name
        if not p.get('plan'):
            try:
                log(t=T[0], grasp_secured_plan_skip=True, source=source, force_next=lift_name,
                    grasp_confirm=bool(EP.get('grasp_confirm')))
            except Exception:
                pass
            return ['skip_plan_grasp_secured']
    igo = p.get('is_goal_object')
    if igo is True or str(igo).lower() == 'true': igo_n = True
    elif igo is False or str(igo).lower() == 'false': igo_n = False
    else: igo_n = None
    issue = str(p.get('issue') or p.get('failure_class') or '')
    mlg = identity_fields(p)['matched_language_goal'] if ('matched_language_goal' in p or 'grasp_px' in p) else None
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
        _reset_patch(reason=f'{source}_id_bad')
        set_goal_id(False, f'{source}_id_bad', {'is_goal_object': igo_n, 'matched_language_goal': mlg, 'issue': issue})
    elif stall_issue and mlg is True:
        # Keep goal identity; ensure plan has a non-relook feedback step.
        plan = list(p.get('plan') or [])
        stall_ok = ('raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach', 'push_aside', 'adjust')
        if not any(st.get('step') in stall_ok or st.get('step') == 'relook' for st in plan):
            # Neighbour block → prefer push_aside (blocker_px used if later present); else raise.
            if issue == 'blocked_by_neighbour' and goal_id_confirmed():
                default_step = 'push_aside'
            elif issue == 'empty_grasp':
                default_step = 'raise_redescend'
            else:
                default_step = 'clear_reapproach' if issue in ('tip_push', 'object_pushed_by_descent') else 'raise_redescend'
            p['plan'] = [{'step': default_step}] + [st for st in plan if st.get('step') != 'relook']
    if _should_open_on_intervention(igo_n=igo_n, issue=issue):
        EP['closed'] = False; grip(-1, 15); EP['released'] = True
    has_kp = any(k in p for k in ('grasp_px', 'object_center_px', 'place_px'))
    target_updated = False
    if has_kp:
        # A planning reply is not a perception result; refresh using the visual model.
        return [f"relook:{do_relook(f'{source}:planner_keypoints')}"]
    done_steps = []
    stall_steps = {'raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach'}
    id_ok = goal_id_confirmed()
    last = str(EP.get('last_result') or '')
    stallish = last in ('stalled', 'timeout') or issue in ('empty_grasp', 'tip_push', 'blocked_by_neighbour', 'object_pushed_by_descent', 'other')
    for st in (p.get('plan') or [])[:3]:
        k = st.get('step')
        if k == 'adjust':
            PATCH.update(grasp_dz_objh=f(st, 'grasp_dz_objh', -.5, .5, PATCH['grasp_dz_objh']), tilt_deg=f(st, 'tilt_deg', 0, 30, PATCH['tilt_deg']),
                         yaw_deg=f(st, 'yaw_deg', -90, 90, PATCH['yaw_deg']))
            td = f(st, 'tilt_dir_deg', -180, 180, None)
            if td is not None:
                th = np.deg2rad(td); dw = RM @ np.array([np.cos(th), -np.sin(th), 0]); PATCH['tilt_dir_deg'] = float(np.degrees(np.arctan2(dw[1], dw[0])))
            _note_patch_mutation(f'{source}:adjust', dict(st))
            done_steps.append('adjust'); target_updated = True
        elif k in stall_steps:
            try:
                ns_fb, _ = namespace()
                det = _apply_stall_feedback(ns_fb, f'{source}:{k}', action=k)
                done_steps.append(f'{k}:{det.get("action")}'); target_updated = True
            except Exception as ex:
                done_steps.append(f'{k}_error:{ex}')
        elif k == 'push_aside':
            try:
                ns_fb, _ = namespace()
                det = _apply_stall_feedback(ns_fb, f'{source}:push_aside', action='push_aside', plan_step=st, dep=dep)
                done_steps.append(f"push_aside:{det.get('action')}:{det.get('error') or ('ok' if det.get('pushed') else 'noop')}")
                target_updated = True
            except Exception as ex:
                done_steps.append(f'push_aside_error:{ex}')
        elif k == 'relook':
            # A real visual refresh replaces the old target. Remaining plan steps were
            # based on the old view, so return to Jev with a fresh namespace afterward.
            done_steps.append(f"relook:{do_relook(f'{source}:plan')}")
            return done_steps
    # Stall + identity confirmed ⇒ shared target MUST change (raise/shift/relook-refresh). Empty relook-only forbidden.
    if id_ok and stallish and not target_updated:
        try:
            ns_fb, _ = namespace()
            det = _apply_stall_feedback(ns_fb, f'{source}:forced_target_update')
            done_steps.append(f'forced_stall_feedback:{det.get("action")}')
            target_updated = True
        except Exception as ex:
            done_steps.append(f'forced_stall_feedback_error:{ex}')
    return done_steps


def _run_episode_start_plan():
    """Great Wall: predictive global plan after start identity, before first approach.
    Astra via Codex writes the todo and start_plan.json before target localization.
    The todo owns task progress; phase_order remains historical planner advice."""
    import coding_agent_slow as CAS
    mode = 'codex'
    out = {'mode': mode, 'applied': False, 'plan_done': [], 'error': None}
    try:
        brief = JX.build_jev_experience_brief(LIB, facts={}, ns={}, uses=EP.get('uses') or {}, dsl=dsl)
    except Exception:
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
    except Exception:
        pass
    _vfinal = START_ID.get('verify_after_relook') if START_ID.get('forced_relook') else START_ID.get('verify')
    start_hints = {
        'goal_id_confirmed': bool(goal_id_confirmed()),
        'identity_matched': (START_ID.get('final') or {}).get('matched_language_goal'),
        'verify_confirmed': (_vfinal or {}).get('confirmed'),
        'identity': START_ID.get('final'),
        'verify': _vfinal,
        'perception_named': {
            'goal': (per0 or {}).get('goal_object_named'),
            'where': (per0 or {}).get('goal_object_where'),
            'place': (per0 or {}).get('place_region_named'),
            'place_where': (per0 or {}).get('place_region_where'),
        },
        'approach_geometry': _approach_geometry(),
        'neighbour_prior': (
            'If another object sits close to the grasp target so the open gripper may collide '
            'before a bilateral close, prefer clear_reapproach or change_approach_yaw before approach. '
            'Clearance requires confirmed identity; if identity is unsure, relook first.'
        ),
    }
    # Annotate the proposed route so geometry can be related to the same front view.
    images = [_save_approach_overlay(start_hints['approach_geometry'])]
    for name in ('start.jpg', 'start_wrist.jpg', 'start_id_verify.jpg', 'start_id_relook.jpg'):
        fp = os.path.join(ep_dir, name)
        if os.path.isfile(fp):
            images.append(fp)
    parsed, rec = None, {}
    try:
        parsed, rec = CAS.harness_call_start_plan(
            lang=LANG, experience_brief=brief, start_hints=start_hints,
            ep_dir=ep_dir, image_paths=images, mode='codex')
        _account_slow_rec(rec)
        proposed_plan = parsed
        if identity_fields(parsed)['matched_language_goal'] is False:
            set_goal_id(False, 'episode_start_plan_rejected', identity_fields(parsed))
        parsed = CAS.gate_start_plan(parsed, goal_id_confirmed())
        # Persist full artifact for RSI reflection
        art = {'plan': parsed, 'proposed_plan': proposed_plan, 'rec': rec, 'start_hints': start_hints, 'mode': mode, 'experience_brief': brief}
        with open(os.path.join(ep_dir, 'start_plan.json'), 'w') as plan_file:
            json.dump(art, plan_file, indent=1, default=str)
        EP['task_todo'] = TODO.initialize(parsed.get('task_todo'), LANG, T[0])
        _save_task_todo('start_plan')
        _retarget_todo()
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
        if not goal_id_confirmed():
            EP['force_next_intent'] = 'relook'
        return out
    except Exception as ex:
        out['error'] = str(ex)[:400]
        try:
            with open(os.path.join(ep_dir, 'start_plan.json'), 'w') as plan_file:
                json.dump(out, plan_file, indent=1, default=str)
            log(t=T[0], episode_start_plan_error=True, mode=mode, err=out['error'])
        except Exception:
            pass
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
    ns, preds = namespace()
    return json.dumps({'just_finished_intent': intent, 'intent_result': result, 'language_goal': LANG,
                       'motion_context': _motion_context(),
                       'recent_intents': EP['hist'][-6:], 'recent_intent_results': EP.get('result_hist', [])[-6:],
                       'recent_gripper_closed': EP.get('grip_hist', [])[-6:],
                       'times_closed': EP['n_close'], 'close_attempts': EP['closes'][-3:],
                       'gripper_closed_command': ns['closed_cmd'], 'fingers_opening_m': round(ns['opening'], 4),
                       'both_fingers_contact': ns['contact'], 'left_finger_contact': ns['left_contact'],
                       'right_finger_contact': ns['right_contact'], 'astra_says_object_held': ns['seen_held'],
                       'gripper_xyz_m': [round(float(x), 3) for x in ns['tcp']],
                       'object_height_estimate_m': round(PER['obj_h'], 3),
                       'goal_id_confirmed': goal_id_confirmed(),
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
    _account_slow_rec(rec)
    issue = str(parsed.get('issue') or 'ok')
    _igo = parsed.get('is_goal_object')
    if _igo is True or str(_igo).lower() == 'true': _igo_n = True
    elif _igo is False or str(_igo).lower() == 'false': _igo_n = False
    else: _igo_n = None
    _mlg = identity_fields(parsed)['matched_language_goal'] if 'matched_language_goal' in parsed else None
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
        set_goal_id(False, 'slow_think_igo', {'is_goal_object': _igo_n, 'matched_language_goal': _mlg,
                                              'issue': issue, 'closed': EP['closed']})
    elif stall_issue:
        parsed['intervene'] = True
        plan = list(parsed.get('plan') or [])
        if not any(st.get('step') in ('raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach', 'push_aside', 'adjust', 'relook') for st in plan):
            if issue == 'empty_grasp':
                _def = 'raise_redescend'
            elif issue == 'blocked_by_neighbour' and goal_id_confirmed():
                _def = 'push_aside'
            else:
                _def = 'clear_reapproach'
            parsed['plan'] = [{'step': _def}] + [st for st in plan if st.get('step') != 'relook']
    elif _igo_n is True and EP['closed']:
        set_goal_id(True, 'slow_think_held_ok', {'is_goal_object': True})
    intervene = bool(parsed.get('intervene')) and issue != 'ok'
    les = str(parsed.get('lesson') or '').strip()
    if les and not any(ch.isdigit() for ch in les):
        FB_LESSONS.append({'failure_class': issue, 'diagnosis': re.sub(r'\d', '#', str(parsed.get('diagnosis', ''))),
                           'lesson': les, 'source': 'slow_think'})
    out = {'issue': issue, 'intervene': intervene, 'diagnosis': parsed.get('diagnosis'), 'lesson': les,
           'is_goal_object': parsed.get('is_goal_object'), 'goal_object_where': parsed.get('goal_object_where'),
           'held_or_between_fingers': parsed.get('held_or_between_fingers'), 'mode': mode, 'crit': crit,
           'snap_dec': snap_dec, 'apply_dec': dec}
    if not intervene:
        Image.fromarray(both).save(os.path.join(ep_dir, f'slow_ok_c{P["planning_calls"]}.jpg'))
        ID_EVENTS.append({'where': 'slow_think', 'intent': intent, 'result': result, **{k: out[k] for k in
            ('issue', 'intervene', 'is_goal_object', 'goal_object_where', 'held_or_between_fingers', 'diagnosis', 'mode', 'crit')}})
        return 'ok', out
    if P['slow'] >= MAX_SLOW and crit != 'reserved_motion_obstruction':
        out['intervene'] = False; return 'slow_cap_drop', out
    if crit != 'reserved_motion_obstruction': P['slow'] += 1
    done_steps = _apply_episode_plan(parsed, dep, source='slow')
    image_name = f'slow_motion_reserved_{T[0]}.jpg' if crit == 'reserved_motion_obstruction' else f'slow{P["slow"]}.jpg'
    Image.fromarray(both).save(os.path.join(ep_dir, image_name))
    CUR.update(slow=f"{mode}/{crit or '-'}:{issue}->{done_steps}"[:160], slow_t=T[0])
    out.update(plan_done=done_steps, patch=dict(PATCH))
    ID_EVENTS.append({'where': 'slow_think', 'intent': intent, 'result': result, **{k: out.get(k) for k in
        ('issue', 'intervene', 'is_goal_object', 'goal_object_where', 'held_or_between_fingers', 'diagnosis', 'mode', 'crit')},
        'plan_done': done_steps})
    return 'intervened', out

def _prepare_slow_snapshot(ctx, crit, phase, both, image_tag):
    import coding_agent_slow as CAS
    ns, preds = namespace()
    brief = JX.build_jev_experience_brief(
        LIB, facts=facts(ns, preds), ns=ns, uses=EP.get('uses') or {}, dsl=dsl,
        episode_lessons=FB_LESSONS, episode_start_plan=EP.get('start_plan'))
    # Unique image and detached policy context belong to this request.
    frame = os.path.join(ep_dir, f'slow_input_{image_tag}.jpg')
    Image.fromarray(both).save(frame)
    context = CAS.prepare_slow_context(
        lang=LANG, trigger=str(crit or phase), slow_ctx_json=ctx,
        experience_brief=brief, patch=PATCH,
        goal_id={'confirmed': goal_id_confirmed(), 'target': GOAL_TARGET},
        ep_dir=ep_dir, keyframe_paths=[p for p in (EP.get('failed_motion_before_path') or EP.get('motion_before_path'), frame) if p])
    context['motion_images_note'] = 'First image, when two supplied, is before last failed motion; last image is current.'
    context['task_todo'] = json.loads(json.dumps(EP.get('task_todo')))
    context['active_goal'] = TODO.active_item(context['task_todo']) if context['task_todo'] else None
    return {'t': T[0], 'coding_context': context}


def _slow_worker(job_id, result_box, snapshot):
    """Codex only: publish results and log events into this job's private box."""
    events = result_box.setdefault('events', [])
    try:
        import coding_agent_slow as CAS
        parsed, rec = CAS.call_monitor(snapshot['coding_context'], mode='codex')
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
    try:
        P['planning_cost'] += float(rec.get('cost_usd') or 0.0)
        P['planning_calls'] += 1
        P['last_planning_wall'] = time.time()
    except Exception:
        pass

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
        try:
            log(t=T[0], async_discarded_finished=True, **meta, issue=(box.get('parsed') or {}).get('issue'))
        except Exception:
            pass
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
    snap_dec = SLOW_JOB['snap_dec']; stale = dec - snap_dec
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
            or SLOW_JOB['goal_confirmed'] != goal_id_confirmed()):
        _account_slow_rec(rec)
        _clear_slow_job()
        return 'async_state_changed_discard', {'snap_dec': snap_dec, 'apply_dec': dec, 'job_id': job_id}
    if stale > SLOW_STALE_INTENTS:
        _account_slow_rec(rec)
        _clear_slow_job()
        return 'async_stale_discard', {'snap_dec': snap_dec, 'apply_dec': dec, 'stale_intents': stale,
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
    both, rgb, dep = _combo_rgb()
    ctx = _slow_ctx(intent, result)
    job_id = _next_slow_job_id()
    result_box = _new_slow_result_box(job_id)
    snapshot = _prepare_slow_snapshot(ctx, crit, 'slow_think_async', both, f'async_{job_id}')
    SLOW_JOB.update(snap_dec=dec, dep=dep.copy(), both=both.copy(), intent=intent, result=result,
                    mode='async', crit=crit, out=None, err=None, rec=None, parsed=None,
                    job_id=job_id, result_box=result_box, target_rev=GOAL_TARGET['rev'],
                    closed=bool(EP.get('closed')), n_close=int(EP.get('n_close', 0)),
                    goal_confirmed=goal_id_confirmed())
    th = threading.Thread(target=_slow_worker, args=(job_id, result_box, snapshot), daemon=True)
    SLOW_JOB['thread'] = th; th.start()
    return 'async_launched', {'snap_dec': dec, 'intent': intent, 'crit': crit, 'job_id': job_id}

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
    both, rgb, dep = _combo_rgb()
    ctx = _slow_ctx(intent, result)
    try:
        import coding_agent_slow as CAS
        snapshot = _prepare_slow_snapshot(ctx, crit, 'sync', both, f'sync_{_next_slow_job_id()}')
        parsed, rec = CAS.call_monitor(snapshot['coding_context'], mode='codex')
        log(t=T[0], coding_agent_slow=True, mode='codex', sync=True, crit=crit,
            issue=parsed.get('issue'), intervene=parsed.get('intervene'),
            plan=parsed.get('plan'), backend=rec.get('backend'))
        return _finish_slow(parsed, rec, dep, both, intent, result, 'sync_codex', crit, dec)
    except Exception as e:
        log(t=T[0], coding_agent_slow_error=True, mode='codex', sync=True, err=str(e)[:300])
        return f'sync_error:coding_agent:{e}', None


def _consume_help_budget(reason):
    """Shared MAX_SLOW slot for ask_astra / hard-force consults (even if Codex says intervene=false)."""
    if P['slow'] < MAX_SLOW:
        P['slow'] += 1
        log(t=T[0], astra_help_budget=True, reason=reason, slow_used=P['slow'], max_slow=MAX_SLOW)
        return True
    return False

def do_ask_astra(reason):
    """Jev-chosen Codex consult (existing ask_astra intent name): sync slow-think / same plan path as critical fallback. Shares MAX_SLOW."""
    if not FALLBACK:
        return 'ask_astra_disabled'
    if P['slow'] >= MAX_SLOW:
        return 'ask_astra_cap'
    prev = EP.get('last_result') or 'consult'
    sr, so = _sync_slow('ask_astra', prev, crit=f'jev_ask:{reason}')
    # Count consult toward shared budget if _finish_slow did not already increment on intervene.
    if sr in ('slow_cap', 'slow_cap_drop'):
        return 'ask_astra_cap'
    if sr == 'intervened':
        pass  # P['slow'] already bumped in _finish_slow
    elif sr == 'ok':
        _consume_help_budget(f'ask_astra_ok:{reason}')
    elif isinstance(sr, str) and sr.startswith('sync_error'):
        return sr
    elif so is not None and P['slow'] < MAX_SLOW:
        _consume_help_budget(f'ask_astra:{sr}:{reason}')
    log(t=T[0], ask_astra=True, ask_result=sr, reason=reason, slow_used=P['slow'])
    if so is not None:
        log(t=T[0], ask_astra_detail=True, issue=(so or {}).get('issue'), intervene=(so or {}).get('intervene'),
            plan_done=(so or {}).get('plan_done'), crit=(so or {}).get('crit'))
    if sr == 'intervened':
        return 'ask_astra_intervened'
    if sr == 'ok':
        return 'ask_astra_ok'
    return f'ask_astra:{sr}'

def _hard_astra_trigger(intent, result):
    """Critical physical stalls retain one bounded diagnosis after general help is exhausted."""
    if not FALLBACK:
        return None
    if _recovery_context()['no_progress'] and MAX_SLOW > 0:
        if P['slow'] < MAX_SLOW or not EP.get('motion_help_used'):
            return 'recovery_no_progress'
    if result in ('stalled', 'timeout') and MAX_SLOW > 0:
        if P['slow'] < MAX_SLOW or not EP.get('motion_help_used'):
            return 'motion_obstruction'
    if P['slow'] >= MAX_SLOW:
        return None
    if intent in ASK_ASTRA_NAMES:
        return None  # already consulted via execute
    if result in FAIL_RESULTS:
        return 'hard_stall'
    if not goal_id_confirmed():
        rs = str(result or '')
        if (rs.startswith('close_refused') or rs.startswith('identity_escape')
                or rs in ('relooked_unmatched', 'close_refused_identity')
                or (intent not in ('relook', 'finish') and EP.get('n_same', 0) >= 2)):
            return 'hard_identity_unstable'
    return None


def _spectator_distress_signals(intent, result):
    """Soft distress cues for spectator gate (hard rules already cover stall/identity force).
    Empty list => skip spectator this step (no spam)."""
    sig = []
    rs = str(result or '')
    h = EP.get('hist') or []
    rh = EP.get('result_hist') or []
    if not goal_id_confirmed():
        sig.append('identity_unconfirmed')
    if GOAL_ID.get('close_refusals', 0) > 0:
        sig.append('close_refusals')
    if rs.startswith('close_refused') or rs.startswith('identity_escape') or rs in (
            'relooked_unmatched', 'close_refused_identity'):
        sig.append('identity_event')
    if result in FAIL_RESULTS:
        sig.append('stall_or_timeout')
    if rs in ('empty_close',) or 'empty' in rs:
        sig.append('empty_close')
    if EP.get('n_same', 0) >= 2 and intent not in ('finish', 'ask_astra', 'consult_astra'):
        sig.append('repeat_intent')
    if EP.get('stall_feedbacks', 0) > 0 and (
            result in FAIL_RESULTS or EP.get('n_same', 0) >= 2 or 'empty' in rs):
        sig.append('post_stall_feedback')
    if len(h) >= 6 and len(set(h[-6:])) <= 2:
        flips = sum(a != b for a, b in zip(h[-6:], h[-5:]))
        if flips >= 4:
            sig.append('intent_oscillation')
    if len(rh) >= 3 and sum(1 for x in rh[-3:] if x in FAIL_RESULTS or str(x).startswith('close_refused')) >= 2:
        sig.append('recent_fail_cluster')
    # de-dupe preserve order
    seen = set(); out = []
    for s in sig:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def _spectator_gate_eligible(intent, dec):
    """Budget / cooldown / enable checks before spending a spectator Jev call."""
    if not FALLBACK or not GATE_JEV:
        return False
    if P['slow'] >= MAX_SLOW:
        return False
    if intent in ASK_ASTRA_NAMES:
        return False  # acting Jev already chose ask_astra
    last = EP.get('gate_jev_last_dec')
    if last is not None and (dec - last) < GATE_JEV_COOLDOWN:
        return False
    return True

def _spectator_jev_state(intent, result, signals, ns, facts):
    """Current image/todo/proprioception plus actual execution events."""
    state, assets = _jev_input()
    state['role'] = 'spectator_gate'
    state['execution_events'] = {
        'just_finished_intent': intent, 'intent_result': result,
        'recent_intents': (EP.get('hist') or [])[-8:],
        'recent_intent_results': (EP.get('result_hist') or [])[-8:],
        'slow_used': P['slow'], 'max_slow': MAX_SLOW,
    }
    return state, assets


_SPECTATOR_CRITERIA = {
    'defer_to_astra': (
        'Yes — request Codex planning: acting looks stuck, identity doubtful, oscillating, '
        'or current images contradict task_todo progress. Shares slow-think budget.'
    ),
    'continue_acting': (
        'No — let acting Jev keep going; progress looks plausible or distress is mild / already handled.'
    ),
}

def _spectator_jev_should_ask_astra(intent, result, signals, ns, facts, dec):
    """Light spectator Jev call → (defer: bool, detail dict). Updates cooldown on every call."""
    EP['gate_jev_last_dec'] = dec
    EP['gate_jev_calls'] = int(EP.get('gate_jev_calls') or 0) + 1
    state, assets = _spectator_jev_state(intent, result, signals, ns, facts)
    instr = (
        'Spectator gate only: choose defer_to_astra OR continue_acting. '
        'Prefer defer_to_astra when execution_events and current images show failure/repetition without recovery, '
        'or current images contradict task_todo progress. Prefer continue_acting when the last result '
        'shows progress, relook just matched, or budget is nearly spent. One short decisive choice.'
    )
    meta = {**META, 'phase': 'spectator_jev_gate', 'signals': list(signals)}
    try:
        choice, probs, lat = JC.choose(state, instr, _SPECTATOR_CRITERIA, assets=assets, meta=meta)
        defer = choice == 'defer_to_astra'
        detail = {'choice': choice, 'probabilities': probs, 'latency_s': lat, 'signals': list(signals)}
        log(dec=dec, t=T[0], spectator_jev=True, defer=defer, **{k: detail[k] for k in ('choice', 'signals', 'latency_s')})
        return defer, detail
    except Exception as e:
        detail = {'choice': None, 'error': str(e)[:200], 'signals': list(signals), 'defer': False}
        log(dec=dec, t=T[0], spectator_jev=True, defer=False, error=detail['error'], signals=list(signals))
        return False, detail

def do_spectator_astra(intent, result, reason):
    """Spectator said yes → sync Codex path (same budget as ask_astra / hard force)."""
    if not FALLBACK:
        return 'spectator_disabled', None
    if P['slow'] >= MAX_SLOW:
        return 'spectator_cap', None
    sr, so = _sync_slow(intent, result, crit=f'spectator_jev:{reason}')
    if sr in ('slow_cap', 'slow_cap_drop'):
        return 'spectator_cap', so
    if sr == 'intervened':
        pass
    elif sr == 'ok':
        _consume_help_budget(f'spectator_ok:{reason}')
    elif isinstance(sr, str) and sr.startswith('sync_error'):
        return sr, so
    elif so is not None and P['slow'] < MAX_SLOW:
        _consume_help_budget(f'spectator:{sr}:{reason}')
    log(t=T[0], spectator_astra=True, ask_result=sr, reason=reason, slow_used=P['slow'])
    return sr, so


def do_slow_think(intent, result, force=False):
    """Hybrid slow-think after a Jev intent.
    - Most intents: fire async Codex watch on stills; discard if >1 intent stale when ready.
    - Critical (after close / release / no-progress / stage-fail+grip-unchanged): SYNC pause, look, maybe intervene.
    - Hard force (stall / identity-unstable), acting ask_astra, and spectator-gate defer also SYNC; all share MAX_SLOW.
    Mid-episode never uses video; end-of-episode lesson update still watches the full mp4.
    """
    if not FALLBACK: return 'disabled', None
    if DONE[0] or T[0] >= HORIZON: return 'done', None
    if intent in ('finish',) and result == 'finish': return 'skip_finish', None
    crit = _slow_critical(intent, result) if not force else (force if isinstance(force, str) else 'forced')
    if force is True: crit = 'forced'
    # After close with proprio+lift confirm: do not run neighbour undo via sync slow — go lift.
    if crit == 'after_close' and (EP.get('grasp_confirm') or (EP.get('grasp_secured_once') and EP.get('closed'))):
        lift_name = 'lift' if 'lift' in {i['name'] for i in LIB.get('intents', [])} else 'move_above_object'
        EP['force_next_intent'] = lift_name
        out = {'issue': 'ok', 'intervene': False, 'mode': 'grasp_confirm_skip', 'crit': 'after_close',
               'grasp_confirm': True, 'force_next': lift_name}
        try:
            log(t=T[0], slow_think='grasp_confirm_skip', intent=intent, intent_result=result, detail=out)
        except Exception:
            pass
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


def do_fallback(trig):
    """Refresh visual evidence through Astra, then ask Codex for the recovery plan."""
    if P['relook'] >= MAX_RELOOK:
        esc = _escape_identity_deadlock(f'fallback_relook_exhausted:{trig}')
        return f'escape_after_budget:{esc}', {**PATCH, 'escape': True, 'trigger': trig, 'escape_result': esc}
    # Revoke pending advice before changing the target it was based on.
    _abandon_async_slow(reason=f'fallback:{trig}')
    perception = do_relook(f'fallback:{trig}')
    if perception == 'relook_error':
        return 'fallback_perception_error', None
    result, detail = _sync_slow('fallback', perception, f'fallback:{trig}')
    return result, {**PATCH, 'trigger': trig, 'perception': perception, 'slow_detail': detail}


def fb_trigger():
    if not FALLBACK or P['fb'] >= MAX_FB: return None
    # general stage-fail (shared debounce with slow-think via stage_fail_mark)
    sf = _stage_fail_grip_unchanged(consume=True)
    if sf: return sf
    h = EP['hist']; th = EP['tcp_hist']
    if len(h) >= 4 and len(set(h[-4:])) == 1 and h[-1] not in ('finish',) and np.linalg.norm(th[-1] - th[-4]) < .01: return 'no_progress_same_intent'
    if len(h) >= 8 and len(set(h[-8:])) <= 2 and sum(a != b for a, b in zip(h[-8:], h[-7:])) >= 6: return 'intent_oscillation'
    if EP['n_close'] >= 3 and EP['n_close'] % 3 == 0 and EP.get('fb_close_mark') != EP['n_close']:
        EP['fb_close_mark'] = EP['n_close']; return 'repeated_grasp_attempts'
    return None

# ---- episode-start Great Wall global plan (before first approach) ----
dec = 0
START_PLAN_OUT = {'mode': 'codex'}
try:
    START_PLAN_OUT = _run_episode_start_plan()
except Exception as _sp_ex:
    START_PLAN_OUT = {'mode': 'error', 'error': str(_sp_ex)[:300]}
    try:
        log(t=T[0], episode_start_plan_error=True, err=START_PLAN_OUT['error'])
    except Exception:
        pass

t0 = time.time(); res = {'suite': suite, 'task_id': tid, 'init': init_i, 'language': LANG, 'library': LIBP, 'library_version': LIB.get('version'),
                         'fallback': FALLBACK, 'perception0': per0, 'start_plan': START_PLAN_OUT}
end = 'horizon'; jev_calls = 0; fb_events = []; slow_events = []
try:
    if START_PLAN_OUT.get('error'):
        raise RuntimeError(f"episode_start_plan_failed:{START_PLAN_OUT['error']}")
    intents = {i['name']: i for i in LIB['intents']}
    while T[0] < HORIZON and not DONE[0] and dec < MAX_DEC:
        _refresh_task_progress()
        if EP.get('todo_review_due'):
            _review_task_todo(EP.pop('todo_review_due'))
        if EP['closed'] and not any(contacts()):
            EP['grasp_confirm'] = False
            EP['grasp_secured_once'] = False
            PER['seen_held'] = False
        trig = fb_trigger()
        if trig:  # early perception-and-Codex recovery: episode-only re-perception (never edits the library)
            P['fb'] += 1; r, pt = do_fallback(trig); fb_events.append({'dec': dec, 'trigger': trig, 'result': r, 'patch': pt})
            log(dec=dec, t=T[0], fallback=trig, result=r, patch=pt)
            CUR.update(fb=f"{trig} -> {r} " + (f"dz={pt['grasp_dz_objh']:+.2f}objh tilt={pt['tilt_deg']} yaw={pt['yaw_deg']}" if pt else ''), fb_t=T[0])
        ns, preds = namespace(); f = facts(ns, preds); forced = None
        recovery_mutated = False
        # Escape / recovery may have queued a non-close next intent — honor before Jev.
        if EP.get('force_next_intent') and EP['force_next_intent'] in intents:
            forced = EP['force_next_intent']; EP['force_next_intent'] = None
        for rc in LIB.get('recoveries', []):
            try:
                if forced: break
                trig_ok = dsl.evaluate(rc['trigger'], ns)
                uses = EP['uses'].get(rc['id'], 0)
                max_u = int(rc.get('max_uses', 3 if 'force_intent' in rc else 1))
                if not trig_ok or uses >= max_u:
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
                    fb = _apply_stall_feedback(ns, f"recovery:{rc['id']}", action=forced_act)
                    recovery_mutated = True
                    log(dec=dec, t=T[0], recovery=rc['id'], stall_feedback=fb)
                    # Prefer the feedback menu's chosen next intent over a static force_intent.
                    if EP.get('force_next_intent') and EP['force_next_intent'] in intents:
                        forced = EP['force_next_intent']; EP['force_next_intent'] = None
                    elif 'force_intent' in rc and rc['force_intent'] in intents:
                        forced = rc['force_intent']
                    break
                if rc.get('approach_change'):
                    _apply_relative_approach_change(ns, f"recovery:{rc['id']}")
                    recovery_mutated = True
                sh = rc.get('shape') or {}
                if sh:
                    e = lambda k: float(dsl.evaluate(str(sh[k]), ns))
                    for k, rk in (('tilt_deg', 'tilt'), ('tilt_dir_deg', 'tdir'), ('yaw_deg', 'yaw')):
                        if k in sh: RSHAPE[rk] = e(k)
                    if 'grasp_dz_objh' in sh: RSHAPE['dz'] = e('grasp_dz_objh')
                    # Relative yaw delta (dimensionless degrees) added to current command — preferred over absolute yaw_deg.
                    if 'yaw_delta_deg' in sh:
                        cur = _cmd_yaw_deg(); RSHAPE['yaw'] = float(np.clip(cur + e('yaw_delta_deg'), -90, 90))
                        PATCH['yaw_deg'] = RSHAPE['yaw']
                    recovery_mutated = True
                    _note_patch_mutation(f"recovery:{rc['id']}", sh)
                    log(dec=dec, t=T[0], shape_rule=rc['id'], rshape=dict(RSHAPE), patch_yaw=PATCH.get('yaw_deg'))
                if 'force_intent' in rc and rc['force_intent'] in intents:
                    forced = rc['force_intent']; break
            except Exception as ex:
                log(dec=dec, t=T[0], recovery_error=rc.get('id'),
                    error_type=type(ex).__name__, error=str(ex))
                # A rule can fail after moving the arm or partially updating state.
                # Stop this episode; continuing would use an unknown recovery state.
                raise
        # B05: recovery mutates pose/PATCH — recompute ns/facts before selection and execute.
        if recovery_mutated:
            ns, preds = namespace(); f = facts(ns, preds)
            try:
                log(dec=dec, t=T[0], post_recovery_ns_refresh=True, forced=forced,
                    grasp=list(np.asarray(ns['grasp'], float).round(4)),
                    tcp=list(np.asarray(ns['tcp'], float).round(4)))
            except Exception:
                pass
        _refresh_task_progress()
        if forced:
            it, probs = forced, None
        else:
            state, assets = _jev_input(include_execution_history=True)
            menu = _acting_menu(intents, ns)
            it, probs, _ = JC.choose(state, JO.INSTRUCTIONS,
                                     JO.candidate_definitions(menu), assets=assets, meta=META)
            jev_calls += 1
        CUR['intent'] = it; r = execute(it, ns); dec += 1
        _refresh_task_progress(it, r)
        EP['last_result'] = r
        if it in ASK_ASTRA_NAMES or it in ('relook', 'retreat') or r in FAIL_RESULTS or 'empty' in str(r):
            EP['todo_review_due'] = f'{it}:{r}'
        EP['n_same'] = EP['n_same'] + 1 if it == EP['prev'] else 1; EP['prev'] = it; EP['step'] += 1
        EP['hist'].append(it); EP['tcp_hist'].append(eef())
        EP['result_hist'].append(r); EP['grip_hist'].append(bool(EP['closed']))
        contact_state = robot_contacts()
        EP.setdefault('recovery_contact_hist', []).append(
            contact_state['environment_contact_parts'] + contact_state['self_contact_parts'])
        if it == 'descend_to_grasp' and r in ('reached', 'at_target', 'stalled', 'timeout'):
            EP['last_descent_step'] = T[0]
        # Place-phase thrash tracking → maybe queue force open_gripper for next decision
        try:
            _note_place_phase(it, r)
            _maybe_queue_place_force_release()
        except Exception:
            pass
        # grip change clears stage-fail streak debounce so a later stuck phase can fire again
        if len(EP['grip_hist']) >= 2 and EP['grip_hist'][-1] != EP['grip_hist'][-2]:
            EP['stage_fail_mark'] = 0
        log(dec=dec, t=T[0], intent=it, forced=forced, result=r, facts=f, probabilities=probs)
        # Codex planning help: ask_astra already ran sync in execute; else hard-force; else spectator gate; else normal slow-think
        if r != 'finish' and not DONE[0] and T[0] < HORIZON:
            skip = {'rate_limited', 'skip_at_target', 'skip_finish', 'disabled', 'done', 'slow_cap', 'async_busy',
                    'ask_astra_disabled', 'ask_astra_cap', 'spectator_disabled', 'spectator_cap'}
            if it in ASK_ASTRA_NAMES:
                sr, so = f'ask_done:{r}', {'via': 'execute', 'result': r}
                slow_events.append({'dec': dec, 'intent': it, 'intent_result': r, 'result': sr, 'detail': so, 'source': 'ask_astra'})
                log(dec=dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so, source='ask_astra')
            else:
                hard = _hard_astra_trigger(it, r)
                if hard:
                    sr, so = _sync_slow(it, r, crit=hard)
                    if sr == 'ok':
                        _consume_help_budget(hard)
                    elif sr == 'intervened':
                        pass  # already counted
                    slow_events.append({'dec': dec, 'intent': it, 'intent_result': r, 'result': sr, 'detail': so, 'source': hard})
                    log(dec=dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so, source=hard)
                else:
                    # Spectator Jev (旁观者清): additive gate — may request Codex help when acting looks stuck
                    gate_fired = False
                    if _spectator_gate_eligible(it, dec):
                        signals = _spectator_distress_signals(it, r)
                        if signals:
                            # Fresh ns/facts after execute (and any recovery) for spectator inputs (B05).
                            ns_gate, preds_gate = namespace(); f_gate = facts(ns_gate, preds_gate)
                            defer, gdet = _spectator_jev_should_ask_astra(it, r, signals, ns_gate, f_gate, dec)
                            if defer:
                                reason = ','.join(signals)
                                sr, so = do_spectator_astra(it, r, reason)
                                gate_fired = True
                                slow_events.append({'dec': dec, 'intent': it, 'intent_result': r, 'result': sr,
                                                    'detail': so, 'source': 'spectator_jev', 'gate': gdet})
                                log(dec=dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so,
                                    source='spectator_jev', gate=gdet)
                    if not gate_fired:
                        sr, so = do_slow_think(it, r, force=False)
                        if so is not None or sr not in skip:
                            slow_events.append({'dec': dec, 'intent': it, 'intent_result': r, 'result': sr, 'detail': so})
                            log(dec=dec, t=T[0], slow_think=sr, intent=it, intent_result=r, detail=so)
        _refresh_task_progress()
        if r == 'finish': break
    # LIBERO official eval does not stop on a policy 'finish': keep no-op stepping (gripper holds last command) until done or HORIZON
    stop = 'finish' if EP['prev'] == 'finish' else ('decision_cap' if dec >= MAX_DEC else None); t_stop = T[0]
    while not DONE[0] and T[0] < HORIZON:
        step([0, 0, 0, 0, 0, 0, 1 if EP['closed'] else -1])
    if DONE[0]: end = 'success' if stop is None else f'{stop}_then_success'
    else: end = f'{stop}_not_success' if stop else 'horizon'
    _refresh_task_progress()
    res['stopped_at_step'] = t_stop if stop else None
except Exception:
    import traceback; end = 'harness_error'; res['error'] = traceback.format_exc()[-1200:]
_finalize_slow_jobs()
if DONE[0] and EP.get('task_todo'):
    EP['task_todo'] = TODO.official_completion(EP['task_todo'], T[0])
    _save_task_todo('official_success')
# diagnostic: which sim body moved most (never shown to policy)
moved_most = None
try:
    E = env.env; xyz_now = {n: E.sim.data.body_xpos[i].copy() for n, i in E.obj_body_id.items()}
    if hasattr(diag, '_xyz0'):
        moved = sorted(((n, float(np.linalg.norm(xyz_now[n] - diag._xyz0[n]))) for n in diag._xyz0 if n in xyz_now), key=lambda x: -x[1])
        moved_most = [{'name': n, 'disp_m': round(d, 4)} for n, d in moved[:5]]
    else:
        # capture start poses now if DIAG was off during steps
        moved_most = [{'name': n, 'disp_m': None} for n in list(xyz_now)[:5]]
except Exception as _e:
    moved_most = [{'error': str(_e)[:120]}]
res.update(success=bool(DONE[0]), end_reason=end, env_steps=T[0], decisions=dec, jev_calls=jev_calls, astra_calls=P['astra_calls'],
           astra_cost_usd=round(P['astra_cost'], 4), planning_calls=P['planning_calls'],
           planning_cost_usd=round(P['planning_cost'], 4), relooks=P['relook'], fallback_events=fb_events, slow_think_events=slow_events,
           slow_interventions=P['slow'], gate_jev_calls=EP.get('gate_jev_calls', 0), final_patch=PATCH, fallback_lessons=FB_LESSONS, learned_shape=RSHAPE,
           task_todo=EP.get('task_todo'), todo_review_calls=EP.get('todo_calls', 0),
           start_identity=START_ID, identity_events=ID_EVENTS, goal_id=GOAL_ID, goal_target=GOAL_TARGET,
           close_refusals=GOAL_ID.get('close_refusals', 0), moved_most=moved_most, wall_s=round(time.time() - t0, 1),
           motion_diagnosis_used=bool(EP.get('motion_help_used')),
           place_force_release=bool(EP.get('place_force_release')),
           place_force_release_reason=EP.get('place_force_release_reason'),
           place_lower_stall_n=int(EP.get('place_lower_stall_n', 0)),
           place_lift_lower_cycles=int(EP.get('place_lift_lower_cycles', 0)))
if VID:
    try:
        VID.stdin.close()
    except BrokenPipeError:
        pass
    video_rc = VID.wait()
    res['video_path'] = VIDP
    res['video_exit_code'] = video_rc
if FINE_STEPS or EP.get('fine_step_events'):
    res['fine_steps_enabled'] = bool(FINE_STEPS)
    res['fine_step_events'] = list(EP.get('fine_step_events') or [])
json.dump(res, open(os.path.join(ep_dir, 'result.json'), 'w'), indent=1, default=lambda o: np.asarray(o).tolist())
print(json.dumps({k: res[k] for k in ('suite', 'task_id', 'init', 'success', 'end_reason', 'env_steps', 'decisions', 'astra_cost_usd', 'slow_interventions', 'close_refusals', 'wall_s') if k in res}))
try: env.close()  # explicit EGL teardown (avoid __del__ EGLError at interpreter exit)
except Exception: pass
sys.stdout.flush(); logf.close(); os._exit(1 if end == 'harness_error' else 0)
