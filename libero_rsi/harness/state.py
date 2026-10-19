"""Shared episode state.

The harness modules share one episode at a time. Mutable containers are
module-level objects that are mutated in place, never rebound, so every module
that imports them sees the same data. `reset()` restores them for a new run.
Simulator handles and run arguments live on `rt`.
"""
import json
import queue
from contextlib import contextmanager
from copy import deepcopy

import numpy as np


class Runtime:
    """Handles created while starting an episode (simulator, library, output files)."""

    def __init__(self):
        self.suite = self.tid = self.init_i = self.LIBP = self.OUT = None
        self.FALLBACK = True
        self.LIB = self.META = None
        self.b = self.task = self.LANG = None
        self.env = self.sim = self.m = self.obs = None
        self.ep_dir = self.logf = self.DIAGF = None
        self.VID = self.VIDP = None
        self.FING = self.FIX = self.ROBOT_PARTS = None
        self.CP = self.RM = self.F = None  # agentview camera position, rotation, focal length (px)
        self.per0 = self.RGB0 = self.DEP0 = None
        self.Q0 = None  # start end-effector orientation
        self.HOME_EEF = None  # start end-effector position
        self.dec = 0  # completed policy decisions


rt = Runtime()


def log(**k):
    rt.logf.write(json.dumps(k, default=lambda o: np.asarray(o).round(4).tolist()) + '\n'); rt.logf.flush()


NONFATAL = []  # diagnostics/fallback errors that did not change control flow; reported in result.json


def warn(where, exc, once=False, **ctx):
    """Record an error that is safe to continue past (diagnostics, documented fallbacks).

    once=True records only the first occurrence per (where, ctx): for code re-evaluated every step."""
    if once and any(r['where'] == where and all(r.get(k) == v for k, v in ctx.items()) for r in NONFATAL):
        return
    rec = {'where': where, 'error': f'{type(exc).__name__}: {exc}'[:300], 'step': T[0]}
    rec.update(ctx)
    NONFATAL.append(rec)
    if rt.logf is not None and not rt.logf.closed:
        try:
            log(t=T[0], ev='harness_warning', **rec)
        except Exception as log_exc:
            # Keep both failures for result.json; do not retry the failed log sink.
            rec['log_error'] = f'{type(log_exc).__name__}: {log_exc}'[:300]


@contextmanager
def nonfatal(where, **ctx):
    """Wrap diagnostics-only code (logging, image dumps): failures are recorded, never swallowed silently."""
    try:
        yield
    except Exception as exc:
        warn(where, exc, **ctx)


def _empty_patch():
    return {'grasp_dz_objh': 0.0, 'tilt_deg': None, 'tilt_dir_deg': None, 'yaw_deg': None, 'grasp_dxy_objw': [0.0, 0.0]}


def new_episode_state():
    return {
        'closed': False, 'n_same': 0, 'prev': 'none', 'step': 0, 'ever_contact': False, 'n_close': 0, 'released': False,
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
        'place_cycle_armed': False, 'place_force_release': False, 'place_force_release_reason': None,
    }


_INITIAL = {
    'P': {'astra_cost': 0.0, 'astra_calls': 0, 'relook': 0, 'fb': 0, 'slow': 0, 'last_astra_wall': 0.0,
          'planning_calls': 0, 'planning_cost': 0.0, 'last_planning_wall': 0.0},
    # Episode gate: only True counts as confirmed. null/False/unsure => refuse close until a relook matches.
    'GOAL_ID': {'confirmed': False, 'source': None, 'detail': None, 'close_refusals': 0, 'events': []},
    # Episode-only action patches from the early perception-and-Codex recovery (NEVER written to the library);
    # override library shape when set.
    'PATCH': _empty_patch(),
    # Single grasp-target owner: one PER base + one PATCH offset layer. Never two competing goals.
    # Source priority for the effective grasp: relook/keypoint refresh > slow_patch > episode_start.
    'GOAL_TARGET': {'source': 'episode_start', 'base_from': None, 'last_patch_from': None, 'rev': 0},
    'RSHAPE': {'tilt': None, 'tdir': None, 'yaw': None, 'dz': 0.0},  # learned conditional overrides (library recoveries), episode-scoped once triggered
    'SHAPE': {'tilt': 0.0, 'tdir': 0.0, 'yaw': None},  # from the current library intent
    'START_ID': {'forced_relook': False, 'verify': None, 'attempts': []},
    'CUR': {'intent': '-', 'fb': '', 'fb_t': -999, 'slow': '', 'slow_t': -999},  # video caption
    # Per-job ownership: each async request gets a unique job_id + private result_box.
    # Abandoned workers may finish later but MUST NOT write into a newer job's slot (B01).
    'SLOW_JOB': {'thread': None, 'snap_dec': None, 'dep': None, 'both': None, 'intent': None, 'result': None,
                 'mode': None, 'crit': None, 'out': None, 'err': None, 'rec': None, 'parsed': None,
                 'job_id': None, 'result_box': None, 'target_rev': None, 'closed': None,
                 'n_close': None, 'goal_confirmed': None},
}

P = {}
GOAL_ID = {}
PATCH = {}
GOAL_TARGET = {}
RSHAPE = {}
SHAPE = {}
START_ID = {}
CUR = {}
SLOW_JOB = {}
PER = {}  # keypoint base from the latest perception
# Policy decision state. Stays empty until start perception has finished: the
# start-seed keypoint commit must not mark a re-approach.
EP = {}
ID_EVENTS = []  # identity checks at start / relook / fallback (policy-visible reflection evidence)
FB_LESSONS = []  # qualitative fallback reflections (no numbers) -> RSI evidence; quantitative patches stay episode-only
T = [0]  # simulation steps
DONE = [False]  # official success
SLOW_JOB_SEQ = [0]  # monotonic generation for job_id
SLOW_ORPHAN_WAITERS = []
SLOW_ORPHAN_COSTS = queue.SimpleQueue()  # completed abandoned boxes; consumed only on main thread


def reset():
    for name, value in _INITIAL.items():
        target = globals()[name]
        target.clear(); target.update(deepcopy(value))
    for container in (PER, EP):
        container.clear()
    for container in (ID_EVENTS, FB_LESSONS, SLOW_ORPHAN_WAITERS, NONFATAL):
        container.clear()
    T[0] = 0; DONE[0] = False; SLOW_JOB_SEQ[0] = 0
    while not SLOW_ORPHAN_COSTS.empty():
        SLOW_ORPHAN_COSTS.get_nowait()
    rt.__init__()


reset()
