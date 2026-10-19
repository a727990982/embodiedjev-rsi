"""Episode budgets, tolerances and executor result labels.

Tunable values are read from the process environment once, when the harness is
imported; set them before launching `run_ep_fbpatch.py`.
"""
import os

import fine_steps as FS
import recovery_control as RC

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
GATE_JEV = os.environ.get('GATE_JEV', '1') != '0'  # spectator Jev gate (旁观者清); 0 disables
GATE_JEV_COOLDOWN = int(os.environ.get('GATE_JEV_COOLDOWN', '3'))  # min decisions between spectator Jev calls
# After a keypoint refresh that moves the target away from the current TCP, block close until re-approach.
REAPPROACH_XY_M = float(os.environ.get('REAPPROACH_XY_M', '0.03'))
MAX_ESCAPE = int(os.environ.get('MAX_ESCAPE', '3'))
MAX_ESCAPE_LOOK = int(os.environ.get('MAX_ESCAPE_LOOK', '2'))
YAW_EXPLICIT_EPS_DEG = 15.0  # new keypoint yaw counts as explicit if it differs from prior PER yaw by this much
SLOW_STALE_INTENTS = 1  # discard async if more than this many intents completed since snapshot
OPEN_MAX = 0.08  # Panda gripper spec (robot spec, not scene)
RECOVERY_MODE = RC.parse_mode(os.environ.get('JEV_RECOVERY_MENU'))
AUTO_RECOVERY_RULES = RECOVERY_MODE == 'both'
FINE_STEPS = FS.enabled_flag()

TRANSPORT_INTENTS = frozenset({'lift', 'carry_to_place', 'lower_to_place'})
GRASP_FAILED = 'grasp_failed:transport_not_executed'
CLOSE_NOT_READY = 'close_not_executed:not_at_grasp'
MOTION_RECOVERIES = frozenset({'back_off', 'clear_reapproach', 'change_approach_yaw'})
FAIL_RESULTS = frozenset({'timeout', 'stalled', GRASP_FAILED, CLOSE_NOT_READY} |
                         {f'{name}_not_executed:grasp_unconfirmed' for name in TRANSPORT_INTENTS})
ASK_ASTRA_NAMES = frozenset({'ask_astra', 'consult_astra'})

STALL_FB_ACTIONS = (
    'raise_redescend', 'lateral_recenter', 'change_approach_yaw',
    'confirm_contact', 'clear_reapproach', 'push_aside',
)
NEIGHBOUR_STALL_ISSUES = ('blocked_by_neighbour', 'tip_push', 'object_pushed_by_descent')

# Same-family local thrash without progress → force retreat + global front relook + sync slow replan.
LOCAL_LOOP_FAMILY = frozenset({
    'push_aside', 'clear_reapproach', 'raise_redescend', 'lateral_recenter',
})
