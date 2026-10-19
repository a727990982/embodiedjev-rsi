"""Current camera/proprioception boundary for acting and spectator Jev."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import uuid

from PIL import Image
import task_todo
from grasp_experience import build as grasp_experience

FIELDS = ('gripper_xyz_m', 'gripper_orientation_xyzw', 'finger_joint_positions_m',
          'fingers_opening_m', 'gripper_closed_command',
          'left_finger_contact', 'right_finger_contact')
INSTRUCTIONS = (
    'Standard movement intents use two-stage control: you choose the intent here, '
    'then choose XYZ directions and fingers in a separate motor request. Each motor '
    'command runs for at most eight control steps. in_progress means the waypoint '
    'is unfinished, not a failed motion; continue the intent when current observations '
    'support it. Only reached/at_target establishes waypoint arrival. '
    'Choose one immediate intent using task_todo, CURRENT camera images and proprioception. '
    'Todo goals are desired outcomes, not current visual facts or a fixed motor sequence. '
    'Use the active object progress stages and next_stage to distinguish grasp, lift, '
    'transport and placement. A pending final outcome does not mean the grasp failed. '
    'Preserve a currently confirmed target grasp while advancing toward placement. '
    'last_visual_review may predate the latest executor checkpoints. '
    'Use motion_context: requested versus actual motion, robot-part contacts, and failed waypoints. '
    'current_orientation reports the full commanded versus measured rotation error now; '
    'last_orientation_motion preserves the last explicit rotation attempt with its timestamp. '
    'Position reached does not prove orientation reached. Check within_tolerance and '
    'required_for_reached in the motion orientation report before treating reorientation as complete. '
    'A stalled motion is not proof of collision; compare the historical motion_before_image with '
    'CURRENT cameras to distinguish obstruction, poor alignment, and unreachable estimates. '
    'Do not repeat an unchanged blocked motion. Back off or change the approach, or ask for help '
    'when budget remains. Startup obstacle advice is a prediction, not current evidence. '
    'Recovery preparation is not a descent test: after retreat/reorientation and reaching above '
    'the target, inspect current images and test the new approach with descend_to_grasp if clear. '
    'Do not repeat preparation solely because the grasp is still pending. If the new approach '
    'still looks obstructed, choose a different approach or request diagnosis. '
    'Respect candidate applicability conditions. If the target is not visible or alignment '
    'is uncertain, relook or ask_astra; do not move blindly using remembered positions. '
    'A close command does not prove a grasp. Nearly closed fingers without contact cannot '
    'acquire a new grasp by repeating close. Recover from empty, slipped or wrong-object grasps. '
    'The active todo identifies the requested object; do not confuse another task object with it. '
    'Only official_success proves the whole task has succeeded.'
    ' release_finalization records the bounded physical settling wait already performed after '
    'opening and a release-relative retreat height. If official success is still false, '
    'inspect the current placement; do not mistake repeated no-op commands for settling time.'
    ' robot_geometry_advice, when present, is a timestamped Astra visual hypothesis about '
    'collision/occlusion, not current truth, an IK solution or a forced action. Compare it '
    'with current images and actual motion before choosing an available action.'
    ' grasp_experience contains conditional advice matched to current signals, not diagnoses '
    'or prescribed actions. Check it against current images and use its verification criteria '
    'to assess the outcome; you still choose among the offered candidates.'
    ' Candidate execution descriptions separate definite program operations from conditional '
    'effects and unverified physics. Commanded targets are visual-estimate controller goals, '
    'not observed future states. Evaluate the alternatives yourself; no candidate is prescribed.'
)

# Match actual executor behavior; do not name predicates from cached perception.
APPLICABILITY = {
    'move_above_object': 'Use when the active target is visible and needs a new approach. Open the fingers and move above its localized grasp point. Do not release a securely held requested target.',
    'descend_to_grasp': 'Use when current images show the visible active target below an aligned, open gripper. Descend with fingers open. It does not correct an empty closed grasp.',
    'close_gripper': 'Use only after the open fingers reach the grasp position in both XY and height, with the visible target between them. Being above the target is insufficient. The executor checks current position against the visual grasp estimate before closing.',
    'reopen_and_retry': 'Use when the fingers are closed without a secure target grasp, or when holding the wrong object. Open the fingers at the current pose; choose alignment or relook separately afterward.',
    'lift': 'Use when current images, opening and contacts support a secure grasp of the active target, and it needs clearance for transport. Raise while keeping the grip; it cannot acquire an object from below empty fingers.',
    'carry_to_place': 'Use when the active target is securely held and clear of surrounding objects. Carry above its destination while maintaining the grip.',
    'lower_to_place': 'Use when the active target is securely held above its intended destination. Lower to release height while maintaining the grip.',
    'open_gripper': 'Use when the correctly held active target has reached its destination and is ready for release. Open at the current pose. If holding the wrong object or empty, use reopen_and_retry.',
    'retreat': 'Use after release to raise the empty, open gripper clear of the destination and reveal the placed object for review. After a placement release the target is at least 6 cm above the recorded release pose; retries reuse that clearance height.',
    'relook': 'Use when current images do not reliably locate the active target or its grasp alignment. Pause target-directed motion and refresh Astra visual localization.',
    'ask_astra': 'Ask Codex/Astra for localization or recovery help when stuck or the active target is unclear. Jev handles routine todo completion checks.',
    'consult_astra': 'Alias of ask_astra: request localization or recovery help.',
    'finish': 'Use only when official_success is true; placement progress or an all-done todo alone is insufficient.',
    'back_off': 'Move upward a short distance while preserving the current grip, to back away from a failed approach or transport. This does not guarantee clearance; inspect the motion result before proceeding.',
    'clear_reapproach': 'Prepare a new approach when not securely holding the target: raise, slightly recenter XY, and adjust estimated grasp height. The executor then queues move_above_object. This action does NOT descend, retry a grasp, or verify that an obstacle is cleared. Do not keep choosing it after preparation without testing the new approach or obtaining new obstruction evidence.',
    'change_approach_yaw': 'When not securely holding the target, retreat and change the commanded approach yaw before approaching again. Use when the current hand orientation appears blocked. This does not prove a collision-free path.',
}


def build_state(language_goal, todo, proprioception, frames, step, official_success=False):
    if not frames:
        raise ValueError('Current camera images are required')
    proprio = {key: deepcopy(proprioception[key]) for key in FIELDS}
    state = {'task': language_goal, 'task_todo': task_todo.decision_view(todo),
             'observation_step': step, 'proprioception': proprio,
             'official_success': bool(official_success), 'cameras': {},
             'sensor_definitions': 'Lengths in meters; joint finger opening is 0 closed, 0.08 fully open. Command and per-finger contacts are measured at observation_step. Contacts do not identify the touched object.',
             'camera_note': 'All images and proprioception are from the same current simulation step.'}
    assets = {}
    for name, pixels in frames.items():
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format='PNG')
        ref = str(uuid.uuid4())
        state['cameras'][name] = ref
        assets[ref] = {'type': 'image', 'format': 'png',
                       'data_base64': base64.b64encode(stream.getvalue()).decode()}
    json.dumps(state, allow_nan=False)
    return state, assets


def candidate_definitions(intents, execution=None):
    # Unknown/custom intent definitions remain supplied by their own library.
    definitions = {name: APPLICABILITY.get(name, value['definition']) for name, value in intents.items()}
    for name, description in (execution or {}).items():
        if name in definitions:
            definitions[name] += '\nCurrent execution description: ' + json.dumps(
                description, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return definitions


def execution_history(intents, results):
    """Describe completed actions without inferring present scene state."""
    if len(intents) != len(results):
        raise ValueError('Each completed intent needs its execution result')
    same_intent = same_pair = 0
    if intents:
        for intent in reversed(intents):
            if intent != intents[-1]:
                break
            same_intent += 1
        for intent, result in zip(reversed(intents), reversed(results)):
            if (intent, result) != (intents[-1], results[-1]):
                break
            same_pair += 1
    return {
        'recent_actions': [{'intent': intent, 'result': result}
                           for intent, result in zip(intents[-8:], results[-8:])],
        'last_intent': intents[-1] if intents else None,
        'consecutive_same_intent': same_intent,
        'consecutive_same_intent_and_result': same_pair,
        'note': ('Past completed actions and executor return values, not claims about '
                 'the current scene or task success. Counts include the latest completed '
                 'action. A grasp_failed result means no confirmed grasp; *_not_executed '
                 'means the requested motion did not run. Current images and proprioception '
                 'describe the present state.'),
    }
