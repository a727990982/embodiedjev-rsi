"""Episode task objectives and visually reviewed progress (no simulator truth)."""
from copy import deepcopy


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
        items.append({**fields, 'status': 'pending', 'evidence': '', 'checked_at_step': None})
    return {'language_goal': language_goal, 'items': items, 'revision': 0,
            'created_at_step': step, 'active_id': items[0]['id']}


def apply_review(todo, raw, step):
    """Apply a complete, current-image review; allow done -> uncertain/pending."""
    updates = raw.get('items') if isinstance(raw, dict) else None
    if not isinstance(updates, list):
        raise ValueError('todo review needs items')
    expected = {item['id'] for item in todo['items']}
    by_id = {}
    for item in updates:
        if not isinstance(item, dict) or item.get('id') not in expected or item['id'] in by_id:
            raise ValueError('todo review must cover each known id exactly once')
        if item.get('status') not in ('pending', 'done', 'uncertain', 'unobserved'):
            raise ValueError('invalid todo status')
        if not isinstance(item.get('evidence'), str) or not item['evidence'].strip():
            raise ValueError('todo review needs current-image evidence or uncertainty reason')
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
        item.update(status=update['status'], evidence=update['evidence'], checked_at_step=step)
    result['revision'] += 1
    active = active_item(result)
    result['active_id'] = active['id'] if active else None
    return result


def decision_view(todo):
    result = deepcopy(todo)
    result['note'] = (
        'Goals describe desired outcomes. Status/evidence are observations at checked_at_step, '
        'not assertions about the current image. Current images and proprioception take priority. '
        'Do not count a command as completion. Jev reviews completion from fresh observations '
        'after actions; the host updates progress before the next decision. '
        'Empty, slipped or wrong-object grasps require recovery within the active goal. '
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
    result.update(active_id=None, revision=result['revision'] + 1)
    return result
