"""Production vision/todo contracts: no simulator and no live model calls."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS
import jev_choice as JC
import jev_observation as JO
import jev_todo as JT
import task_todo as TODO
from test_review_regressions import functions


def todo():
    return TODO.initialize({'items': [
        {'id': name, 'target': name, 'goal': f'put {name} in basket',
         'completion_condition': f'{name} visible inside basket and released', 'status': 'done'}
        for name in ('cheese', 'butter')]}, 'put cheese and butter in basket', 0)


def proprio():
    return {'gripper_xyz_m': [0, 0, .5], 'gripper_orientation_xyzw': [0, 0, 0, 1],
            'finger_joint_positions_m': [.001, -.001], 'fingers_opening_m': .002,
            'gripper_closed_command': True, 'left_finger_contact': False,
            'right_finger_contact': False}


class TodoTests(unittest.TestCase):
    def test_planning_cannot_claim_completion(self):
        t = todo()
        self.assertEqual([i['status'] for i in t['items']], ['pending', 'pending'])
        self.assertEqual(t['active_id'], 'cheese')

    def test_official_completion_has_explicit_official_evidence(self):
        t = todo()
        result = TODO.official_completion(t, 99)
        self.assertIsNone(result['active_id'])
        self.assertTrue(all(item['status'] == 'done' for item in result['items']))
        self.assertIn('Official', result['items'][0]['evidence'])
        self.assertEqual(t['active_id'], 'cheese')

    def test_placement_advances_and_new_evidence_can_reopen_goal(self):
        old = todo()
        raw = {'items': [{'id': 'cheese', 'status': 'done', 'evidence': 'cheese inside basket, fingers open'},
                         {'id': 'butter', 'status': 'pending', 'evidence': 'butter on table'}]}
        updated = TODO.apply_review(old, raw, 10)
        self.assertEqual(updated['active_id'], 'butter')
        self.assertEqual(old['active_id'], 'cheese')
        raw['items'][0].update(status='uncertain', evidence='cheese visible on rim, unstable placement')
        reverted = TODO.apply_review(updated, raw, 12)
        self.assertEqual(reverted['active_id'], 'cheese')
        self.assertEqual(reverted['items'][0]['checked_at_step'], 12)

    def test_occlusion_preserves_verified_history_without_new_timestamp(self):
        t = todo()
        t['items'][0].update(status='done', evidence='inside basket', checked_at_step=10)
        raw = {'items': [{'id': 'cheese', 'status': 'unobserved', 'evidence': 'occluded'},
                         {'id': 'butter', 'status': 'pending', 'evidence': 'on table'}]}
        updated = TODO.apply_review(t, raw, 20)
        self.assertEqual(updated['active_id'], 'butter')
        self.assertEqual(updated['items'][0], t['items'][0])

    def test_incomplete_or_unsupported_review_cannot_update_progress(self):
        t = todo()
        cases = [[], [{'id': 'cheese', 'status': 'done', 'evidence': 'released'}],
                 [{'id': 'other', 'status': 'done', 'evidence': 'x'}],
                 [{'id': 'cheese', 'status': 'done', 'evidence': ''}],
                 [{'id': 'cheese', 'status': 'complete', 'evidence': 'x'}]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                TODO.apply_review(t, {'items': rows}, 1)
        self.assertEqual(t['revision'], 0)

    def test_planner_requests_observable_target_specific_conditions(self):
        self.assertIn('Is the butter inside the basket and no longer held', CAS.START_PLAN_PROMPT)


class VisionTests(unittest.TestCase):
    def test_history_keeps_full_streak_with_bounded_recent_window(self):
        intents = ['close_gripper'] + ['lift'] * 63
        results = ['closed'] + ['at_target'] * 63
        history = JO.execution_history(intents, results)
        self.assertEqual(len(history['recent_actions']), 8)
        self.assertEqual(history['consecutive_same_intent'], 63)
        self.assertEqual(history['consecutive_same_intent_and_result'], 63)
        results[-1] = 'stalled'
        self.assertEqual(history['recent_actions'][-1]['result'], 'at_target')
        changed = JO.execution_history(intents, results)
        self.assertEqual(changed['consecutive_same_intent'], 63)
        self.assertEqual(changed['consecutive_same_intent_and_result'], 1)

    def test_history_requires_paired_completed_actions(self):
        self.assertEqual(JO.execution_history([], [])['consecutive_same_intent'], 0)
        with self.assertRaises(ValueError):
            JO.execution_history(['lift'], [])

    def test_acting_history_is_logged_and_does_not_mutate_old_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            ep = {'task_todo':todo(), 'vision_serial':0,
                  'hist':['lift','lift'], 'result_hist':['reached','at_target']}
            logged = []
            scope = functions('_jev_input', FINE_STEPS=False, _motion_context=lambda:{}, _startup_obstacle_advice=lambda:{}, JO=JO, EP=ep, LANG='goal', T=[706],
                DONE=[False], Image=Image, ep_dir=temp, log=lambda **kw:logged.append(kw),
                _current_observation=lambda:(proprio(),{'front':np.zeros((2,2,3),dtype='uint8')}))
            state, _ = scope['_jev_input'](include_execution_history=True)
            self.assertEqual(state['execution_history']['recent_actions'][-1],
                             {'intent':'lift','result':'at_target'})
            self.assertEqual(logged[-1]['state'], state)
            ep['hist'].append('carry_to_place'); ep['result_hist'].append('reached')
            self.assertEqual(state['execution_history']['last_intent'], 'lift')
            spectator_base, _ = scope['_jev_input']()
            self.assertNotIn('execution_history', spectator_base)

    def test_same_frame_pixels_complete_proprio_and_no_cached_facts(self):
        measured = {**proprio(), 'astra_says_object_held': True, 'at_grasp': True,
                    'place_point_estimate_m': [9, 9, 9]}
        for step in (3, 8):
            pixels = np.full((8, 8, 3), step, dtype=np.uint8)
            state, assets = JO.build_state('goal', todo(), measured, {'front': pixels}, step)
            encoded = assets[state['cameras']['front']]['data_base64']
            self.assertTrue(np.array_equal(np.asarray(Image.open(BytesIO(base64.b64decode(encoded)))), pixels))
            self.assertEqual(state['proprioception'], proprio())
            self.assertEqual(state['observation_step'], step)
            for name in ('observed_facts', 'experience_brief', 'at_grasp', 'astra_says_object_held', 'place_point_estimate_m'):
                self.assertNotIn(name, json.dumps(state))
            state['task_todo']['items'][0]['goal'] = 'mutated'
            self.assertNotEqual(todo()['items'][0]['goal'], 'mutated')

    def test_missing_images_or_contact_is_not_silently_filled(self):
        with self.assertRaises(ValueError):
            JO.build_state('goal', todo(), proprio(), {}, 1)
        p = proprio(); del p['left_finger_contact']
        with self.assertRaises(KeyError):
            JO.build_state('goal', todo(), p, {'front': np.zeros((2, 2, 3), dtype=np.uint8)}, 1)

    def test_transport_sends_assets_and_pins_actual_model(self):
        state, assets = JO.build_state('goal', todo(), proprio(),
                                      {'front': np.zeros((2, 2, 3), dtype=np.uint8)}, 1)
        reply = {'model': JC.MODEL, 'answers': {'next_intent': {
            'choice': 'reopen_and_retry', 'probabilities': {'close_gripper': .1, 'reopen_and_retry': .9}}}}
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-placeholder'}), patch.object(JC, 'urlopen') as call:
            call.return_value.__enter__.return_value.read.return_value = json.dumps(reply).encode()
            choice, probs, _ = JC.choose(state, JO.INSTRUCTIONS,
                {n: JO.APPLICABILITY[n] for n in ('close_gripper', 'reopen_and_retry')}, assets=assets)
            request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body['model'], 'jev-1.14.0')
        self.assertEqual(body['assets'], assets)
        self.assertEqual(body['state']['task_todo'], state['task_todo'])
        self.assertEqual(request.get_header('Typesafe-danger-zone-images'), 'true')
        self.assertEqual(choice, 'reopen_and_retry')
        self.assertEqual(probs['reopen_and_retry'], .9)
        self.assertNotIn('test-placeholder', json.dumps(body))

    def test_transport_rejects_wrong_model_or_unknown_action(self):
        for model, choice in [('d1-ultra', 'close_gripper'), (JC.MODEL, 'unknown')]:
            reply = {'model': model, 'answers': {'next_intent': {
                'choice': choice, 'probabilities': {'close_gripper': 1.0}}}}
            with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-placeholder'}), \
                    patch.object(JC, 'urlopen') as call:
                call.return_value.__enter__.return_value.read.return_value = json.dumps(reply).encode()
                with self.assertRaises(ValueError):
                    JC.choose({}, 'choose', {'close_gripper': 'close'}, assets={'image': {}})


class HarnessTests(unittest.TestCase):
    def test_handoff_relocalizes_and_does_not_bypass_identity(self):
        t = todo(); t['items'][0]['status'] = 'done'; t['active_id'] = 'butter'
        ep = {'task_todo': t, 'todo_target_id': 'cheese', 'grasp_confirm': True,
              'grasp_secured_once': True, 'force_next_intent': 'close_gripper'}
        events, identities = [], []
        scope = functions('_retarget_todo', '_active_language_goal', TODO=TODO, EP=ep,
            LANG='both objects', PER={}, GOAL_TARGET={'rev': 3}, T=[5],
            RSHAPE={'yaw': 90}, _reset_patch=lambda reason: None,
            _abandon_async_slow=lambda reason: events.append('abandon'),
            set_goal_id=lambda value, *a: identities.append(value),
            perceive=lambda **kw: ({'matched_language_goal': True}, None, None),
            _commit_keypoints=lambda *a, **kw: events.append('localize'),
            verify_grasp_identity=lambda *a, **kw: ({'is_language_goal_object': None}, None),
            identity_ok=lambda p: True, log=lambda **kw: None)
        scope['_retarget_todo']()
        self.assertEqual(events, ['abandon', 'localize'])
        self.assertEqual(identities, [False, False])
        self.assertFalse(ep['grasp_secured_once'])
        self.assertEqual(ep['force_next_intent'], 'relook')
        self.assertIn('Current TODO target: butter', scope['_active_language_goal']())

    def test_reopen_is_stationary_and_finish_needs_official_success(self):
        lib = json.loads((ROOT / 'libs' / 'ee_pg_easy_v2_lib.json').read_text())
        commands = []
        ep = {'closed': True, 'grasp_confirm': True, 'grasp_secured_once': True}
        scope = functions('execute', _action_block_reason=lambda *args:None, MOTION_RECOVERIES=set(), LIB=lib, ASK_ASTRA_NAMES={'ask_astra', 'consult_astra'},
            TRANSPORT_INTENTS={'lift', 'carry_to_place', 'lower_to_place'},
            EP=ep, PER={'seen_held': True}, DONE=[False],
            grip=lambda *args: commands.append(args))
        self.assertEqual(scope['execute']('reopen_and_retry', {}), 'reopened')
        self.assertEqual(commands, [(-1, 15)])
        self.assertFalse(ep['grasp_secured_once'])
        self.assertFalse(scope['PER']['seen_held'])
        self.assertEqual(scope['execute']('finish', {}), 'skip_finish')
        scope['DONE'][0] = True
        self.assertEqual(scope['execute']('finish', {}), 'finish')


if __name__ == '__main__':
    unittest.main()
