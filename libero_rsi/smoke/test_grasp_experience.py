"""Offline evidence boundaries and selection regressions for acting Jev advice."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_experience as GE
import jev_observation as JO
import task_todo as TODO
from test_transport_feedback import fixture


def observation():
    todo = TODO.initialize({'items': [{'id': 'target', 'target': 'requested object',
        'goal': 'place in destination', 'completion_condition': 'object in destination'}]}, 'task', 20)
    todo = TODO.refresh_progress(todo, 'target', 20, identity_confirmed=True,
                                 grasp_ready=False, gripper_closed=False)
    return {'task_todo': TODO.decision_view(todo), 'observation_step': 20, 'official_success': False,
            'proprioception': {'gripper_closed_command': False, 'left_finger_contact': False,
                              'right_finger_contact': False},
            'motion_context': {'close_position_check': {'ready': False}, 'recovery_validation': {}}}


def advice(state, intent=None, result=None):
    history = JO.execution_history([intent] if intent else [], [result] if intent else [])
    return GE.build(state, history)


def ids(value):
    return [x['id'] for x in value['lessons']]


class GraspExperienceTests(unittest.TestCase):
    def test_alignment_and_estimated_readiness_do_not_claim_capture(self):
        state = observation()
        self.assertEqual(ids(advice(state)), ['align_before_close'])
        state['motion_context']['close_position_check']['ready'] = True
        self.assertEqual(ids(advice(state)), ['verify_capture'])
        self.assertEqual(advice(state)['lessons'][0]['status'], 'curated_hypothesis')

    def test_refused_close_is_distinct_from_failed_capture(self):
        state = observation()
        refused = advice(state, 'close_gripper', 'close_not_executed:not_at_grasp')
        self.assertIn('execution_refused', ids(refused))
        self.assertNotIn('capture_unconfirmed', ids(refused))
        failed = advice(state, 'close_gripper', 'grasp_failed:transport_not_executed')
        self.assertEqual(ids(failed), ['capture_unconfirmed'])
        # Closing happened; the transport_not_executed suffix must not relabel it as a refused close.

    def test_recovery_preparation_supersedes_historical_stall(self):
        state = observation()
        motion = state['motion_context']
        motion['last_failed_motion'] = {'result': 'stalled', 'end_step': 10}
        motion['recovery_validation'] = {'preparations_without_descent': 2, 'no_progress': True,
                                        'descent_tested_since_preparation': False}
        result = advice(state, 'move_above_object', 'reached')
        self.assertIn('test_prepared_approach', ids(result))
        self.assertNotIn('inspect_stalled_motion', ids(result))
        motion['recovery_validation'] = {'preparations_without_descent': 0, 'no_progress': False,
                                        'descent_tested_since_preparation': True}
        result = advice(state, 'descend_to_grasp', 'stalled')
        self.assertIn('inspect_stalled_motion', ids(result))
        self.assertNotIn('test_prepared_approach', ids(result))

    def test_supported_grasp_and_release_review_override_old_failure_history(self):
        state = observation()
        progress = state['task_todo']['items'][0]['progress']
        progress.update(current_grasp_confirmed=True, next_stage='lower_for_release')
        state['proprioception'].update(gripper_closed_command=True, left_finger_contact=True)
        result = advice(state, 'descend_to_grasp', 'stalled')
        self.assertEqual(ids(result), ['preserve_supported_grasp', 'release_at_destination'])
        state['proprioception'].update(gripper_closed_command=False, left_finger_contact=False)
        progress.update(current_grasp_confirmed=False, awaiting_placement_review=True)
        self.assertEqual(ids(advice(state)), ['review_after_release'])

    def test_missing_contact_does_not_inherit_a_historical_hold(self):
        state = observation()
        state['task_todo']['items'][0]['progress']['current_grasp_confirmed'] = True
        state['proprioception']['gripper_closed_command'] = True
        self.assertEqual(ids(advice(state)), ['capture_unconfirmed'])

    def test_identity_uncertainty_takes_priority_and_stale_progress_is_not_current(self):
        state = observation()
        progress = state['task_todo']['items'][0]['progress']
        progress['stages'][0]['status'] = 'pending'
        self.assertEqual(ids(advice(state)), ['target_identity'])
        progress['checked_at_step'] = 19
        self.assertEqual(ids(advice(state)), [])
        progress['checked_at_step'] = 20
        state['official_success'] = True
        self.assertEqual(ids(advice(state)), [])

    def test_advice_is_bounded_detached_and_ignores_privileged_or_predicted_inputs(self):
        state = observation()
        state['motion_context']['recovery_validation'] = {'no_progress': True}
        state.update(sim_object_pose='secret ground truth', startup_obstacle_advice={'diagnosis': 'blocked'})
        before = deepcopy(state)
        result = advice(state, 'descend_to_grasp', 'stalled')
        self.assertLessEqual(len(result['lessons']), 3)
        self.assertLess(len(json.dumps(result)), 3200)
        self.assertNotIn('secret ground truth', json.dumps(result))
        self.assertEqual(state, before)
        for lesson in result['lessons']:
            self.assertTrue(all(lesson[k] for k in ('when', 'interpretation', 'consider', 'verify', 'source', 'matched_on')))
        result['lessons'][0]['consider'] = 'mutated output'
        self.assertNotIn('mutated output', json.dumps(GE.LESSONS))

    def test_real_jev_input_scopes_history_to_target_and_leaves_candidates_unchanged(self):
        scope, ep, _, commands = fixture()
        todo = TODO.initialize({'items': [{'id': 'new_target', 'target': 'new object', 'goal': 'place',
            'completion_condition': 'placed'}]}, 'task', 100)
        todo = TODO.refresh_progress(todo, 'new_target', 100, identity_confirmed=True,
                                     grasp_ready=False, gripper_closed=False)
        ep.update(task_todo=todo, hist=['close_gripper'], result_hist=['grasp_failed:transport_not_executed'],
                  recovery_history_start=1)
        proprio = {key: 0 for key in JO.FIELDS}
        scope['_current_observation'] = lambda: (proprio, {'front': np.zeros((2,2,3), dtype='uint8')})
        scope['_motion_context'] = lambda: {'close_position_check': {'ready': False}}
        intents = {i['name']: i for i in scope['LIB']['intents']}
        menu_before = deepcopy(scope['_acting_menu'](intents))
        with tempfile.TemporaryDirectory() as temp:
            scope['ep_dir'] = temp
            state, _ = scope['_jev_input'](include_execution_history=True)
            review_state, _ = scope['_jev_input']()
        self.assertEqual(state['grasp_experience']['target_id'], 'new_target')
        self.assertEqual(ids(state['grasp_experience']), ['align_before_close'])
        self.assertNotIn('grasp_experience', review_state)
        self.assertEqual(scope['_acting_menu'](intents), menu_before)
        self.assertEqual(list(scope['_acting_menu'](intents)), list(menu_before))
        self.assertEqual(commands, [])
        self.assertNotIn('force_next_intent', ep)


if __name__ == '__main__':
    unittest.main()
