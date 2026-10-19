"""Current camera/proprioception boundary for acting and spectator Jev."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import uuid

from PIL import Image
import task_todo

FIELDS = ('gripper_xyz_m', 'gripper_orientation_xyzw', 'finger_joint_positions_m',
          'fingers_opening_m', 'gripper_closed_command',
          'left_finger_contact', 'right_finger_contact')
INSTRUCTIONS = (
    'Choose one immediate intent using task_todo, CURRENT camera images and proprioception. '
    'Todo goals are desired outcomes, not current visual facts or a fixed motor sequence. '
    'Use the active object progress stages and next_stage to distinguish grasp, lift, '
    'transport and placement. A pending final outcome does not mean the grasp failed. '
    'Preserve a currently confirmed target grasp while advancing toward placement. '
    'last_visual_review may predate the latest executor checkpoints. '
    'Respect candidate applicability conditions. If the target is not visible or alignment '
    'is uncertain, relook or ask_astra; do not move blindly using remembered positions. '
    'A close command does not prove a grasp. Nearly closed fingers without contact cannot '
    'acquire a new grasp by repeating close. Recover from empty, slipped or wrong-object grasps. '
    'The active todo identifies the requested object; do not confuse another task object with it. '
    'Only official_success proves the whole task has succeeded.'
)

# Match actual executor behavior; do not name predicates from cached perception.
APPLICABILITY = {
    'move_above_object': 'Use when the active target is visible and needs a new approach. Open the fingers and move above its localized grasp point. Do not release a securely held requested target.',
    'descend_to_grasp': 'Use when current images show the visible active target below an aligned, open gripper. Descend with fingers open. It does not correct an empty closed grasp.',
    'close_gripper': 'Use when the fingers are open enough to admit the visible, aligned active target. Close at a fixed end-effector pose. Repeating this command with nearly closed fingers and no contact does not create a new grasp.',
    'reopen_and_retry': 'Use when the fingers are closed without a secure target grasp, or when holding the wrong object. Open the fingers at the current pose; choose alignment or relook separately afterward.',
    'lift': 'Use when current images, opening and contacts support a secure grasp of the active target, and it needs clearance for transport. Raise while keeping the grip; it cannot acquire an object from below empty fingers.',
    'carry_to_place': 'Use when the active target is securely held and clear of surrounding objects. Carry above its destination while maintaining the grip.',
    'lower_to_place': 'Use when the active target is securely held above its intended destination. Lower to release height while maintaining the grip.',
    'open_gripper': 'Use when the correctly held active target has reached its destination and is ready for release. Open at the current pose. If holding the wrong object or empty, use reopen_and_retry.',
    'retreat': 'Use after release to raise the empty, open gripper clear of the destination and reveal the placed object for review.',
    'relook': 'Use when current images do not reliably locate the active target or its grasp alignment. Pause target-directed motion and refresh Astra visual localization.',
    'ask_astra': 'Ask Codex slow-think and Astra todo review for help when stuck, the active goal is unclear, or current images contradict todo progress.',
    'consult_astra': 'Alias of ask_astra: request slow-think and a current-image todo review.',
    'finish': 'Use only when official_success is true; placement progress or an all-done todo alone is insufficient.',
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


def candidate_definitions(intents):
    # Unknown/custom intent definitions remain supplied by their own library.
    return {name: APPLICABILITY.get(name, value['definition']) for name, value in intents.items()}


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
