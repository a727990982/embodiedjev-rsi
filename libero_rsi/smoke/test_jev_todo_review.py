"""Todo routing, image evidence, rollback and atomic Choice response handling."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS
import jev_choice as JC
import jev_observation as JO
import jev_todo as JT
import task_todo as TODO
from test_review_regressions import functions
from test_vision_todo_chain import todo, proprio


def response(statuses):
    return {'model': JC.MODEL, 'answers': {
        f'todo_{i}': {'choice': status, 'probabilities': {
            k: float(k == status) for k in ('done', 'pending', 'uncertain', 'unobserved')}}
        for i, status in enumerate(statuses)}}


class JevTodoTests(unittest.TestCase):
    def observation(self, t=None, step=27):
        self.canonical_todo = deepcopy(t or todo())
        return JO.build_state('put cheese and butter in basket', t or todo(), proprio(),
            {'front': np.full((8, 8, 3), 70, dtype=np.uint8),
             'wrist': np.full((8, 8, 3), 90, dtype=np.uint8)}, step)

    def review(self, state, assets, reply):
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-placeholder'}), \
                patch.object(JC, 'urlopen') as transport, \
                patch.object(CAS, 'call_codex', side_effect=AssertionError('Must use Jev')):
            transport.return_value.__enter__.return_value.read.return_value = json.dumps(reply).encode()
            updated, rec = JT.review(state, todo=self.canonical_todo, assets=assets, reason='open_gripper:opened')
            self.request = json.loads(transport.call_args.args[0].data)
        return updated, rec

    def test_one_request_checks_exact_conditions_without_prior_verdicts(self):
        state, assets = self.observation()
        before = deepcopy(state)
        updated, rec = self.review(state, assets, response(['done', 'pending']))
        body = self.request
        self.assertEqual(body['model'], JC.MODEL)
        self.assertEqual(body['assets'], assets)
        self.assertEqual(body['state']['proprioception'], state['proprioception'])
        self.assertEqual(body['state']['observation_step'], 27)
        self.assertEqual(body['state']['cameras'], state['cameras'])
        self.assertEqual(set(body['questions']), {'todo_0', 'todo_1'})
        for index, item in enumerate(before['task_todo']['items']):
            question = body['questions'][f'todo_{index}']
            self.assertIn(item['completion_condition'], question['instructions'])
            self.assertIn(item['target'], question['instructions'])
            self.assertNotIn('status', body['state']['task_todo']['items'][index])
            self.assertNotIn('evidence', body['state']['task_todo']['items'][index])
        self.assertIn('hovering above the basket', body['questions']['todo_1']['instructions'])
        self.assertEqual(updated['active_id'], 'butter')
        self.assertEqual(updated['items'][0]['checked_at_step'], 27)
        self.assertIn('classified done', updated['items'][0]['evidence'])
        self.assertNotIn('note', updated)
        self.assertEqual(rec['answers'], response(['done', 'pending'])['answers'])
        self.assertEqual(rec['item_ids'], ['cheese', 'butter'])
        self.assertEqual(state, before)

    def test_occlusion_keeps_old_timestamp_but_visible_contradiction_reopens(self):
        prior = todo()
        prior['items'][0].update(status='done', evidence='earlier classification', checked_at_step=10)
        prior['active_id'] = 'butter'
        state, assets = self.observation(prior)
        updated, _ = self.review(state, assets, response(['unobserved', 'uncertain']))
        self.assertEqual(updated['items'][0], prior['items'][0])
        self.assertEqual(updated['items'][1]['status'], 'uncertain')
        self.assertEqual(updated['active_id'], 'butter')
        state, assets = self.observation(updated, step=28)
        reopened, _ = self.review(state, assets, response(['pending', 'pending']))
        self.assertEqual(reopened['active_id'], 'cheese')
        self.assertEqual(reopened['items'][0]['checked_at_step'], 28)

    def test_post_release_review_preserves_or_reopens_stages(self):
        for choice, expected in [('yes',False),('no',True),('uncertain',True)]:
            prior=todo()
            prior['items'][0]['progress']['awaiting_placement_review']=True
            state,assets=self.observation(prior)
            reply=response(['pending','pending'])
            reply['answers']['regrasp_0']={'choice':choice,'probabilities':{
                key:float(key==choice) for key in ('yes','no','uncertain')}}
            updated,_=self.review(state,assets,reply)
            self.assertEqual(updated['items'][0]['progress']['awaiting_placement_review'],expected)
            self.assertIn('last_visual_review',TODO.decision_view(updated)['items'][0])
            self.assertIn('regrasp_0',self.request['questions'])
            del reply['answers']['regrasp_0']
            with self.assertRaises(ValueError):self.review(state,assets,reply)
            self.assertTrue(prior['items'][0]['progress']['awaiting_placement_review'])

    def test_partial_or_invalid_service_answers_cannot_commit_any_completion(self):
        replies = []
        missing = response(['done', 'pending']); del missing['answers']['todo_1']; replies.append(missing)
        extra = response(['done', 'pending']); extra['answers']['other'] = extra['answers']['todo_0']; replies.append(extra)
        bad = response(['done', 'pending']); bad['answers']['todo_1']['choice'] = 'complete'; replies.append(bad)
        bad = response(['done', 'pending']); bad['answers']['todo_1']['probabilities']['done'] = True; replies.append(bad)
        bad = response(['done', 'pending']); bad['answers']['todo_1']['probabilities']['pending'] = float('nan'); replies.append(bad)
        bad = response(['done', 'pending']); bad['answers']['todo_1'] = None; replies.append(bad)
        for reply in replies:
            state, assets = self.observation()
            before = deepcopy(state)
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                self.review(state, assets, reply)
            self.assertEqual(state, before)

    def test_host_commit_invalidates_changed_progress_before_retarget(self):
        for statuses in (['pending', 'pending'], ['done', 'pending']):
            state, assets = self.observation()
            updated, rec = self.review(state, assets, response(statuses))
            ep = {'task_todo': todo(), 'todo_calls': 0, 'vision_serial': 1}
            events = []
            scope = functions('_review_task_todo', JT=JT, EP=ep, T=[27],
                _jev_input=lambda: (state, assets),
                _abandon_async_slow=lambda reason: events.append('abandon'),
                _save_task_todo=lambda reason: events.append(('save', ep['task_todo']['active_id'])),
                _retarget_todo=lambda: events.append(('retarget', ep['task_todo']['active_id'])),
                log=lambda **kw: None)
            with patch.object(JT, 'review', return_value=(updated, rec)):
                scope['_review_task_todo']('after_action')
            active = updated['active_id']
            expected = (['abandon'] if statuses[0] == 'done' else []) + [('save', active), ('retarget', active)]
            self.assertEqual(events, expected)
            self.assertEqual(ep['todo_calls'], 1)
            events.clear()
            with patch.object(JT, 'review', return_value=(updated, rec)):
                scope['_review_task_todo']('terminal', retarget=False)
            self.assertEqual(events, [('save', active)])


if __name__ == '__main__':
    unittest.main()
