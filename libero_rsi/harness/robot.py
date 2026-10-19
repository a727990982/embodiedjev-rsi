"""LIBERO environment, stepping, video, tactile/camera sensors and TCP servoing."""

import gc
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot

import motion_feedback as MF

from .config import HORIZON, S
from .state import CUR, DONE, EP, PATCH, PER, RSHAPE, SHAPE, T, log, rt


def make_env():
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    # EGL fix: retry context creation after gc (stale contexts from previous processes / parallel jobs)
    last = None
    for a in range(4):
        try:
            e = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path('bddl_files'), rt.task.problem_folder, rt.task.bddl_file),
                                   camera_heights=S, camera_widths=S, camera_depths=True, horizon=HORIZON)
            e.seed(0); e.reset(); o = e.set_init_state(rt.b.get_task_init_states(rt.tid)[rt.init_i]); return e, o
        except Exception as ex:
            last = ex; gc.collect(); time.sleep(3 + 3 * a)
    raise last


def diag():  # DIAGNOSTIC ONLY (never shown to policy): per-step BDDL goal predicate parts + poses
    E = rt.env.env; osd = E.object_states_dict; out = {'t': T[0], 'check_success': bool(E._check_success()), 'done_flag': DONE[0]}
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
    if rt.DIAGF: rt.DIAGF.write(json.dumps(out) + '\n'); rt.DIAGF.flush()


def _start_video(path):
    import subprocess
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x572', '-r', '20', '-i', '-',
                             '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', '-crf', '26', '-preset', 'medium', '-movflags', '+faststart', path], stdin=subprocess.PIPE)


def vid_frame(a):
    im = Image.new('RGB', (1024, 572)); im.paste(Image.fromarray(rt.obs['agentview_image'][::-1]), (0, 0))
    if 'robot0_eye_in_hand_image' in rt.obs: im.paste(Image.fromarray(rt.obs['robot0_eye_in_hand_image'][::-1]).resize((512, 512)), (512, 0))
    d = ImageDraw.Draw(im); g = 'CLOSED' if a[6] > 0 else 'OPEN'
    op = float(rt.obs['robot0_gripper_qpos'][0] - rt.obs['robot0_gripper_qpos'][1])
    d.text((6, 516), f"step {T[0]}/{HORIZON}   intent: {CUR['intent']}   gripper cmd: {g}  opening {op*100:.1f}cm   {'SUCCESS' if DONE[0] else ''}", fill=(255, 255, 0))
    if T[0] - CUR['fb_t'] < 60: d.text((6, 536), f"FALLBACK: {CUR['fb']}"[:170], fill=(255, 80, 80))
    elif T[0] - CUR['slow_t'] < 60: d.text((6, 536), f"SLOW: {CUR['slow']}"[:170], fill=(80, 200, 255))
    d.text((6, 554), 'left: agentview   right: wrist cam', fill=(180, 180, 180))
    rt.VID.stdin.write(im.tobytes())


def step(a):
    if DONE[0] or T[0] >= HORIZON:
        return
    rt.obs, r, done, info = rt.env.step(np.clip(a, -1, 1)); T[0] += 1
    if rt.VID: vid_frame(a)
    if rt.DIAGF and EP.get('n_close', 0) > 0: diag()
    if rt.env.env._check_success(): DONE[0] = True


def eef():
    return rt.obs['robot0_eef_pos'].copy()


def opening():
    return float(rt.obs['robot0_gripper_qpos'][0] - rt.obs['robot0_gripper_qpos'][1])


def _fixture(g):
    bid = rt.m.geom_bodyid[g]; n = (rt.m.body_id2name(bid) or '').lower()
    return bid == 0 or n.startswith(('robot0', 'gripper0', 'mount0')) or 'table' in n or 'floor' in n


def contacts():
    l = r = False
    for i in range(rt.sim.data.ncon):
        c = rt.sim.data.contact[i]; a, bb = c.geom1, c.geom2
        for f, o in ((a, bb), (bb, a)):
            if o in rt.FIX: continue
            if f in rt.FING['L']: l = True
            if f in rt.FING['R']: r = True
    return l, r


def init_sensors():
    """Tactile proxy and camera intrinsics from the loaded MuJoCo model."""
    m = rt.m = rt.sim.model
    rt.FING = {'L': {m.geom_name2id(n) for n in ('gripper0_finger1_collision', 'gripper0_finger1_pad_collision')},
               'R': {m.geom_name2id(n) for n in ('gripper0_finger2_collision', 'gripper0_finger2_pad_collision')}}
    rt.FIX = {g for g in range(m.ngeom) if _fixture(g)}
    # Robot geometry names describe our own sensor locations, never the touched body.
    rt.ROBOT_PARTS = {}
    for g in range(m.ngeom):
        body = (m.body_id2name(m.geom_bodyid[g]) or '').lower()
        if g in rt.FING['L']: rt.ROBOT_PARTS[g] = 'left_finger'
        elif g in rt.FING['R']: rt.ROBOT_PARTS[g] = 'right_finger'
        elif body.startswith('gripper0'): rt.ROBOT_PARTS[g] = 'palm_or_wrist'
        elif body.startswith('robot0'): rt.ROBOT_PARTS[g] = 'arm'
    rt.CP, rt.RM, rt.F = camera()


def robot_contacts():
    pairs = [(rt.sim.data.contact[i].geom1, rt.sim.data.contact[i].geom2)
             for i in range(rt.sim.data.ncon)]
    return MF.contact_summary(pairs, rt.ROBOT_PARTS)


def camera():
    cid = rt.m.camera_name2id('agentview'); return rt.sim.data.cam_xpos[cid].copy(), rt.sim.data.cam_xmat[cid].reshape(3, 3).copy(), 0.5 * S / np.tan(np.deg2rad(rt.m.cam_fovy[cid]) / 2)


def rgbd():
    from robosuite.utils import camera_utils as CU
    return rt.obs['agentview_image'][::-1].copy(), CU.get_real_depth_map(rt.sim, rt.obs['agentview_depth'])[..., 0][::-1].copy()


def bp(uv, dep):
    u, v = int(np.clip(round(uv[0]), 2, S - 3)), int(np.clip(round(uv[1]), 2, S - 3))
    d = float(np.median(dep[v - 2:v + 3, u - 2:u + 3])); return rt.CP + rt.RM @ np.array([(uv[0] - S / 2) / rt.F, -(uv[1] - S / 2) / rt.F, -1.0]) * d


def servo(target, grip, gain=1.0, max_steps=60, tol=.006, ori_tol=None):
    last = eef(); stall = 0; prev_ori = None
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
            path = os.path.join(rt.ep_dir, f'motion_before_{start_step}_{T[0]}.png')
            Image.fromarray(before_image).save(path)
            EP['motion_before_path'] = path
            EP['failed_motion_before_path'] = path
        log(t=T[0], motion_feedback=EP['last_motion'])
        return result
    pick = lambda pk, sk, d: next((v for v in (PATCH[pk], RSHAPE[sk], SHAPE[sk]) if v is not None), d)
    yw = pick('yaw_deg', 'yaw', PER['yaw']); tl = float(np.clip(pick('tilt_deg', 'tilt', 0), 0, 30)); td = pick('tilt_dir_deg', 'tdir', 0)
    Rt = Rot.from_rotvec(np.deg2rad(tl) * np.array([np.cos(np.deg2rad(td + 90)), np.sin(np.deg2rad(td + 90)), 0])) * Rot.from_euler('z', np.deg2rad(yw)) * rt.Q0
    for _ in range(max_steps):
        if T[0] >= HORIZON or DONE[0]: return finish('horizon_or_done')
        d = target - eef(); a = np.zeros(7); a[:3] = np.clip(d / .05 * 1.6, -1, 1) * float(np.clip(gain, .15, 1.0))
        er = (Rt * Rot.from_quat(rt.obs['robot0_eef_quat']).inv()).as_rotvec(); a[3:6] = np.clip(er / .5 * 1.2, -1, 1)
        a[6] = grip; step(a)  # gripper binary open/close (EmbodiedEvo-style)
        contact_state = robot_contacts()
        during['environment'].update(contact_state['environment_contact_parts'])
        during['self'].update(contact_state['self_contact_parts'])
        ori_err = float(np.linalg.norm((Rt * Rot.from_quat(rt.obs['robot0_eef_quat']).inv()).as_rotvec())) if ori_tol is not None else 0.0
        if np.linalg.norm(target - eef()) < tol and (ori_tol is None or ori_err < ori_tol):
            return finish('reached')
        ori_progress = ori_tol is not None and prev_ori is not None and prev_ori - ori_err > .002
        prev_ori = ori_err
        mv = np.linalg.norm(eef() - last); last = eef()
        stall = stall + 1 if mv < .0005 and not ori_progress else 0
        if stall >= 8: return finish('stalled')
    return finish('timeout')


def grip(g, n):
    for _ in range(n):
        if T[0] >= HORIZON or DONE[0]: break
        step([0, 0, 0, 0, 0, 0, g])


def _combo_rgb():
    rgb, dep = rgbd(); wr = rt.obs.get('robot0_eye_in_hand_image')
    both = np.concatenate([rgb, np.asarray(Image.fromarray(wr[::-1]).resize((S, S)))], 1) if wr is not None else rgb
    return both, rgb, dep
