"""Jev judges observable todo completion; the host commits progress before acting."""
from copy import deepcopy
import json

import jev_choice as JC
import task_todo as TODO


INSTRUCTIONS = '''Judge ONLY the specified object's completion condition using the
CURRENT camera images and same-frame proprioception. This is a progress check,
not a next-action question. Assess every clause of the completion condition.
Distinguish the requested object from its neighbours and the destination interior
from its rim or the space above it. For "butter inside the basket and released",
done requires the BUTTER inside the basket and no longer held by the gripper;
butter still held, hovering above the basket, or outside it is pending. Apply
the supplied condition to the actual target, not the example's object.
A commanded release, open fingers, contacts, an intended action or an earlier
done status alone never proves completion. Use uncertain for visible but
ambiguous evidence, including uncertain object identity. Use unobserved when
occlusion or the camera view prevents checking the required object/region.
Visible evidence that a previously completed goal is now false means pending.
Evaluate each item independently: official_success=false or another unfinished
item does not mean this particular item is incomplete.
Do not predict what the next action will accomplish. Only the environment's
official success signal establishes full task success.'''


def review(state, *, assets, reason):
    """One multi-question request, followed by an atomic host-side todo update."""
    todo = deepcopy(state['task_todo'])
    todo.pop('note', None)
    step = state['observation_step']
    questions = {}
    for index, item in enumerate(todo['items']):
        subject = json.dumps({k: item[k] for k in
                              ('target', 'goal', 'completion_condition')}, ensure_ascii=False)
        condition = item['completion_condition']
        questions[f'todo_{index}'] = {
            'type': 'choice',
            'instructions': INSTRUCTIONS + '\nJudge this item:\n' + subject,
            'criteria': {
                'done': f'Current evidence establishes ALL of: {condition}',
                'pending': f'Current evidence shows at least one requirement is NOT satisfied: {condition}',
                'uncertain': f'The relevant scene is visible, but identity or satisfaction is ambiguous: {condition}',
                'unobserved': f'The required object/region is occluded or outside the views, so this cannot be checked: {condition}',
            },
        }
    # Hide old verdicts from the classifier. Keep them locally for occlusion
    # preservation and rollback; the host never infers success from an action.
    review_state = deepcopy(state)
    review_state['task_todo'] = {'language_goal': todo['language_goal'], 'items': [
        {k: item[k] for k in ('id', 'target', 'goal', 'completion_condition')}
        for item in todo['items']]}
    answers, latency = JC.choose_questions(review_state, questions, assets=assets)
    rows = []
    for index, item in enumerate(todo['items']):
        answer = answers[f'todo_{index}']
        rows.append({'id': item['id'], 'status': answer['choice'],
                     # Choice answers do not include a free-text visual rationale.
                     'evidence': f"Jev {JC.MODEL} classified {answer['choice']} at step {step}; "
                                 'see logged current frames, completion question and probabilities.'})
    updated = TODO.apply_review(todo, {'items': rows}, step)
    return updated, {'model': JC.MODEL, 'observation_step': step,
                     'reason': reason, 'latency_s': latency,
                     'questions': questions, 'answers': answers,
                     'item_ids': [item['id'] for item in todo['items']]}
