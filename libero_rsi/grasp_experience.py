"""Conditional grasp advice for acting Jev, selected from its existing observations.

These are curated hypotheses, not learned weights or proven recovery policies.
No simulator access, model calls, candidate filtering or controller mutation.
"""
from copy import deepcopy


LESSONS = {
    'target_identity': {
        'when': 'The active target identity is not currently confirmed.',
        'interpretation': 'An occluded view does not erase an earlier mismatch; contact cannot identify an object.',
        'consider': 'Compare current views with the requested target. Consider relook or available help before target-directed grasping.',
        'verify': 'Require fresh identity confirmation and visible alignment before attempting capture.',
        'source': 'library:hp_start_identity,hp_wrong_object',
    },
    'capture_unconfirmed': {
        'when': 'A close failed confirmation, or the fingers are commanded closed without a supported target grasp.',
        'interpretation': 'Closing is not capture. This can be an empty, slipped or wrong-object grasp; the trigger does not diagnose which.',
        'consider': 'Inspect the target between the fingers. If empty, consider reopening and correcting alignment or relocalizing; repeating close alone cannot acquire it.',
        'verify': 'Check current opening/contact and target motion during a lift; a reached waypoint alone does not prove a grasp.',
        'source': 'library:hp_b,hp_e,hp_contact_not_held',
    },
    'execution_refused': {
        'when': 'The latest requested action was refused or not executed.',
        'interpretation': 'A refused close is not an empty grasp, and a refused motion is not a new collision observation.',
        'consider': 'Read the refusal reason and current images. Address identity, alignment or held-object constraints using available candidates.',
        'verify': 'Distinguish a later executed attempt and its measured outcome from repeated selections of a refused action.',
        'source': 'library:refused_close_is_not_empty_close; executor applicability',
    },
    'test_prepared_approach': {
        'when': 'Recovery preparation has not been followed by a descent test, or preparation/approach cycles show no progress.',
        'interpretation': 'Raising or changing a commanded pose does not establish obstacle clearance or graspability.',
        'consider': 'After reaching above the visible, aligned target, consider descent if current views support clearance; otherwise consider a different approach or available diagnosis.',
        'verify': 'Compare actual descent, remaining error and robot-part contacts. Do not count another successful preparation as grasp progress.',
        'source': 'PR12 init_08_direct recovery loop; PR13 recovery-validation guidance',
    },
    'inspect_stalled_motion': {
        'when': 'The latest action reports stalled or timeout.',
        'interpretation': 'Limited motion can reflect obstruction, misalignment or an unreachable estimate. Palm/wrist contact does not identify a blocker.',
        'consider': 'Compare the failed-motion image with current views and motion feedback. Consider retreat, reorientation or relocalization; push only a visually identified blocker if applicable.',
        'verify': 'A new approach must produce measured progress without unwanted contact. Unchanged failed waypoints are not new evidence.',
        'source': 'library:hp_a,hp_c,contactless_descent_alignment; PR12 init_08_direct',
    },
    'align_before_close': {
        'when': 'The open gripper has not reached the visual grasp estimate in both XY and height.',
        'interpretation': 'Being above the object is insufficient; a clear TCP path also does not guarantee clearance for the whole hand.',
        'consider': 'Inspect front/wrist alignment and nearby objects. Consider approach, descent or a visual refresh according to current clearance.',
        'verify': 'Check measured position and the visible target between the open fingers before closing.',
        'source': 'library:hp_near_approach_tips,hp_e; PR12 close-position check',
    },
    'verify_capture': {
        'when': 'The open gripper is near the visual grasp estimate.',
        'interpretation': 'Position readiness is an estimate, not proof that the requested object lies between the fingers.',
        'consider': 'Use current images to decide whether to close or correct alignment. Account for the full hand and neighbours.',
        'verify': 'After closing, require supported grasp confirmation and inspect the lift outcome before transport.',
        'source': 'library:hp_e,hp_contact_not_held; executor grasp confirmation',
    },
    'preserve_supported_grasp': {
        'when': 'Current progress confirms a target grasp and the closed fingers still have contact support.',
        'interpretation': 'An unfinished placement or older negative visual review does not mean the current grasp failed.',
        'consider': 'Keep the grasp while selecting clearance, transport or lowering actions suited to current views and progress.',
        'verify': 'Check target retention and clearance after each motion; reconsider capture if current evidence contradicts the hold.',
        'source': 'library:hp_contact_not_held,hp_place_target_check',
    },
    'release_at_destination': {
        'when': 'A supported grasp has reached the transport/lowering stage.',
        'interpretation': 'Reaching a transport waypoint does not itself establish a safe release pose.',
        'consider': 'Check the intended destination, height and rim clearance before selecting lowering or release.',
        'verify': 'Inspect the released object after retreat; only official_success establishes whole-task success.',
        'source': 'library:hp_d,hp_place_target_check,hp_wait_official_success',
    },
    'review_after_release': {
        'when': 'The gripper is open and the active object is awaiting placement review.',
        'interpretation': 'Pending completion or occlusion alone is not evidence that another grasp is needed.',
        'consider': 'Consider retreat to reveal the destination. Regrasp only if current evidence supports a placement failure.',
        'verify': 'Check the object at the intended destination and the official result; do not infer success from release alone.',
        'source': 'library:hp_wait_official_success; task todo placement review',
    },
}


def build(state, target_history):
    """Return at most three advisory entries; inputs are already policy-visible.

    target_history contains only actions since the active target was selected.
    Historical failure images and startup predictions alone never trigger advice.
    """
    todo = state['task_todo']
    active_id = todo.get('active_id')
    out = {
        'target_id': active_id, 'observation_step': state['observation_step'],
        'note': 'Curated advice, not scene facts or proven recovery outcomes. Check applicability in CURRENT images. '
                'Choose freely among the offered candidates; advice neither adds actions nor overrides their guards.',
        'lessons': [],
    }
    item = next((i for i in todo.get('items', []) if i['id'] == active_id), None)
    if state['official_success'] or item is None:
        return out
    progress = item.get('progress', {})
    if progress.get('checked_at_step') != state['observation_step']:
        return out  # Do not present stale checkpoint matches as current conditions.
    stages = {s['id']: s['status'] for s in progress.get('stages', [])}
    proprio = state['proprioception']
    motion = state.get('motion_context', {})
    closed = bool(proprio['gripper_closed_command'])
    contact = bool(proprio['left_finger_contact'] or proprio['right_finger_contact'])
    recent = target_history.get('recent_actions', [])
    last = recent[-1] if recent else {}
    result = last.get('result', '')

    def add(key, matched_on):
        out['lessons'].append({'id': key, **deepcopy(LESSONS[key]),
                               'status': 'curated_hypothesis', 'matched_on': deepcopy(matched_on)})

    if stages.get('identify_target') != 'done':
        add('target_identity', {'identity_stage': stages.get('identify_target')})
        return out
    if progress.get('current_grasp_confirmed') and closed and contact:
        add('preserve_supported_grasp', {'current_grasp_confirmed': True, 'closed_with_contact': True})
        if progress.get('next_stage') in ('lower_for_release', 'place_and_clear'):
            add('release_at_destination', {'next_stage': progress['next_stage']})
        return out
    if progress.get('awaiting_placement_review') and not closed:
        add('review_after_release', {'awaiting_placement_review': True, 'gripper_closed_command': False})
        return out
    capture_failed = last.get('intent') == 'close_gripper' and result.startswith('grasp_failed')
    refused = not capture_failed and ('not_executed' in result or result.startswith(('close_refused', 'close_blocked')))
    if refused:
        add('execution_refused', last)
    if closed or capture_failed:
        add('capture_unconfirmed', {'gripper_closed_command': closed, 'last_target_action': last})
        return out
    recovery = motion.get('recovery_validation', {})
    if (recovery.get('no_progress') or recovery.get('preparations_without_descent', 0) > 0
            or recovery.get('descent_tested_since_preparation') is False):
        add('test_prepared_approach', {k: recovery[k] for k in (
            'no_progress', 'preparations_without_descent', 'descent_tested_since_preparation') if k in recovery})
    if result in ('stalled', 'timeout'):
        add('inspect_stalled_motion', last)
    if len(out['lessons']) < 3:
        ready = motion.get('close_position_check', {}).get('ready')
        if ready is not None:
            add('verify_capture' if ready else 'align_before_close', {'visual_grasp_position_ready': ready})
    return out
