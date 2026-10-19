"""Episode task objectives and visually reviewed progress (no simulator truth)."""
from copy import deepcopy


STAGES = (
    ('identify_target', 'Verify the requested target identity'),
    ('secure_grasp', 'Secure the target with a confirmed grasp'),
    ('lift_clear', 'Lift the held target clear for transport'),
    ('carry_to_destination', 'Carry the held target above its destination'),
    ('lower_for_release', 'Lower the held target to the release waypoint'),
    ('place_and_clear', 'Release, retreat clear, and verify the final outcome'),
)


def _new_progress():
    return {'stages': [{'id': key, 'goal': goal, 'status': 'pending',
                       'evidence': '', 'checked_at_step': None} for key, goal in STAGES],
            'next_stage': STAGES[0][0], 'current_grasp_confirmed': False,
            'checked_at_step': None, 'awaiting_placement_review': False}


def _set_stage(progress, key, status, evidence, step):
    stage = next(s for s in progress['stages'] if s['id'] == key)
    stage.update(status=status, evidence=evidence, checked_at_step=step)


def _complete_progress(item, step):
    progress = item['progress']
    # Final placement does not prove every intermediate motion was performed.
    for stage in progress['stages'][:-1]:
        if stage['status'] != 'done':
            stage.update(status='not_required', checked_at_step=step,
                         evidence='Final outcome verified without separate evidence for this checkpoint.')
    _set_stage(progress, 'place_and_clear', 'done', item['evidence'], step)
    progress.update(next_stage=None, current_grasp_confirmed=False,
                    checked_at_step=step, awaiting_placement_review=False)


def active_item(todo):
    return next((item for item in (todo or {}).get('items', [])
                 if item['status'] != 'done'), None)


def initialize(raw, language_goal, step):
    if not isinstance(raw, dict) or not isinstance(raw.get('items'), list) or not raw['items']:
        raise ValueError('task_todo.items must be a nonempty list')
    items, ids = [], set()
    for item in raw['items']:
        fields = {}
        for key in ('id', 'target', 'goal', 'completion_condition'):
            value = item.get(key) if isinstance(item, dict) else None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'task_todo item needs {key}')
            fields[key] = value.strip()
        if fields['id'] in ids:
            raise ValueError('duplicate task_todo id')
        ids.add(fields['id'])
        # Planning is not evidence: discard any planner-supplied done flags.
        items.append({**fields, 'status': 'pending', 'evidence': '', 'checked_at_step': None,
                      'progress': _new_progress()})
    return {'language_goal': language_goal, 'items': items, 'revision': 0,
            'created_at_step': step, 'active_id': items[0]['id']}


def apply_review(todo, raw, step):
    """Apply a complete, current-image review; allow done -> uncertain/pending."""
    updates = raw.get('items') if isinstance(raw, dict) else None
    if not isinstance(updates, list):
        raise ValueError('todo review needs items')
    expected = {item['id'] for item in todo['items']}
    awaiting = {item['id'] for item in todo['items']
                if item['progress']['awaiting_placement_review']}
    by_id = {}
    for item in updates:
        if not isinstance(item, dict) or item.get('id') not in expected or item['id'] in by_id:
            raise ValueError('todo review must cover each known id exactly once')
        if item.get('status') not in ('pending', 'done', 'uncertain', 'unobserved'):
            raise ValueError('invalid todo status')
        if not isinstance(item.get('evidence'), str) or not item['evidence'].strip():
            raise ValueError('todo review needs current-image evidence or uncertainty reason')
        if item['id'] in awaiting and item['status'] in ('pending', 'uncertain'):
            if 'needs_regrasp' not in item or (item['needs_regrasp'] is not None
                                              and type(item['needs_regrasp']) is not bool):
                raise ValueError('post-release review needs boolean/null needs_regrasp')
        by_id[item['id']] = item
    if set(by_id) != expected:
        raise ValueError('incomplete todo review')
    result = deepcopy(todo)
    for item in result['items']:
        update = by_id[item['id']]
        # Occlusion does not create fresh evidence or erase verified history.
        # Contradictory visible evidence must be returned as pending/uncertain.
        if update['status'] == 'unobserved':
            continue
        was_done = item['status'] == 'done'
        item.update(status=update['status'], evidence=update['evidence'], checked_at_step=step)
        if item['status'] == 'done':
            _complete_progress(item, step)
        elif was_done or (item['progress']['awaiting_placement_review']
                          and update.get('needs_regrasp') is True):
            # A visible placement failure reopens the grasp/transport checkpoints.
            item['progress'] = _new_progress()
        elif item['progress']['awaiting_placement_review']:
            _set_stage(item['progress'], 'place_and_clear', 'pending', update['evidence'], step)
    result['revision'] += 1
    active = active_item(result)
    result['active_id'] = active['id'] if active else None
    return result


def refresh_progress(todo, target_id, step, *, identity_confirmed, grasp_ready,
                     gripper_closed, intent=None, result=None, waypoints_changed=False):
    """Refresh the active object's checkpoints from executor feedback and sensors.

    Motion checkmarks are achieved waypoints within the current grasp, not claims
    about the object's current pose. No image-model call or simulator truth here.
    """
    updated = deepcopy(todo)
    item = active_item(updated)
    if item is None or item['id'] != target_id:
        return updated
    progress = item['progress']
    stages = {s['id']: s for s in progress['stages']}
    ready = bool(identity_confirmed and grasp_ready and gripper_closed)
    # Only an actual release after verified transport waits for placement review.
    if (intent == 'open_gripper' and result in ('reached', 'at_target')
            and not gripper_closed and progress['current_grasp_confirmed']
            and stages['carry_to_destination']['status'] == 'done'):
        progress['awaiting_placement_review'] = True
    elif intent is not None and intent not in ('retreat', 'relook', 'ask_astra',
                                               'consult_astra', 'finish'):
        progress['awaiting_placement_review'] = False
    if not identity_confirmed or ready:
        progress['awaiting_placement_review'] = False
    _set_stage(progress, 'identify_target', 'done' if identity_confirmed else 'pending',
               'Target identity verified by the visual identity gate.' if identity_confirmed
               else 'Target identity needs visual verification.', step)
    if ready:
        _set_stage(progress, 'secure_grasp', 'done',
                   'Executor grasp confirmation is supported by current finger contacts/opening.', step)
        if waypoints_changed:
            for stage in progress['stages'][2:5]:
                _set_stage(progress, stage['id'], 'pending',
                           'Visual localization or the executor waypoint changed; recheck this checkpoint.', step)
        motion_stage = {'lift': 'lift_clear', 'carry_to_place': 'carry_to_destination',
                        'lower_to_place': 'lower_for_release'}.get(intent)
        if (motion_stage and result in ('reached', 'at_target')
                and (intent != 'lower_to_place' or stages['carry_to_destination']['status'] == 'done')):
            _set_stage(progress, motion_stage, 'done',
                       f'Executor {intent} returned {result} with a currently supported grasp; '
                       'waypoint uses visual localization, not verified object placement.', step)
            if intent == 'carry_to_place' and stages['lift_clear']['status'] == 'pending':
                _set_stage(progress, 'lift_clear', 'not_required',
                           'Destination reached without a separately recorded lift waypoint.', step)
            if intent in ('lift', 'carry_to_place'):
                _set_stage(progress, 'lower_for_release', 'pending',
                           'Release height needs checking after this motion.', step)
    elif not progress['awaiting_placement_review']:
        for stage in progress['stages'][1:5]:
            _set_stage(progress, stage['id'], 'pending',
                       'No currently supported target grasp; re-establish grasp before transport.', step)
    _set_stage(progress, 'place_and_clear', 'pending',
               'Released after transport; retreat and obtain current-image placement review.'
               if progress['awaiting_placement_review'] else 'Final outcome has not been verified.', step)
    progress.update(current_grasp_confirmed=ready, checked_at_step=step)
    progress['next_stage'] = ('place_and_clear' if progress['awaiting_placement_review'] else
                              next(s['id'] for s in progress['stages'] if s['status'] == 'pending'))
    if updated == todo:
        return updated
    updated['revision'] += 1
    return updated


def decision_view(todo):
    result = deepcopy(todo)
    for item in result['items']:
        # Keep old scene descriptions explicitly historical, separate from live progress.
        item['last_visual_review'] = {'status': item['status'],
                                     'evidence': item.pop('evidence'),
                                     'checked_at_step': item.pop('checked_at_step')}
    result['note'] = (
        'Each object has fine-grained progress stages. next_stage is the next unmet subgoal, '
        'not a forced action sequence. Motion checkmarks record achieved executor waypoints '
        'within the supported grasp; not_required does not claim a motion occurred. '
        'current_grasp_confirmed is refreshed from executor confirmation and current sensors. '
        'When grasp is confirmed and lift_clear is done, preserve the grasp and progress toward '
        'the destination; do not reopen merely because the overall outcome is pending. '
        'last_visual_review is historical and can predate the current grasp/lift. '
        'Current images and proprioception take priority. '
        'Do not count a command as completion. If a completed goal is contradicted, ask Astra '
        'to review it. Empty, slipped or wrong-object grasps require recovery within the active goal. '
        'Only official_success establishes full task success.'
    )
    return result


def official_completion(todo, step):
    """Called only after the environment's official success predicate is true."""
    result = deepcopy(todo)
    for item in result['items']:
        if item['status'] != 'done':
            item.update(status='done', evidence='Official full language-goal success check passed.',
                        checked_at_step=step)
        _complete_progress(item, step)
    result.update(active_id=None, revision=result['revision'] + 1)
    return result
