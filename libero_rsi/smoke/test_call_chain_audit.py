"""Regression probes for concrete call-chain failures; no live models or simulator."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from test_review_regressions import functions
import coding_agent_slow as CAS


class CallChainAuditTests(unittest.TestCase):
    def test_nonzero_cli_exit_rejects_even_valid_json(self):
        with patch.object(CAS, '_resolve_codex_bin', return_value='/fake/codex'), \
             patch.object(CAS, '_proxy_env', return_value={}), \
             patch.object(CAS.subprocess, 'run', return_value=SimpleNamespace(
                 returncode=1, stdout='{"issue":"ok","intervene":false,"plan":[]}', stderr='request failed')):
            with self.assertRaisesRegex(RuntimeError, 'codex_exit_1'):
                CAS.call_monitor({}, mode='codex')

    def test_missing_requested_image_does_not_start_cli(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(CAS, '_resolve_codex_bin', return_value='/fake/codex'), \
             patch.object(CAS.subprocess, 'run') as run:
            with self.assertRaisesRegex(FileNotFoundError, 'codex_image_missing'):
                CAS.call_codex('look at image', [str(Path(temp) / 'missing.jpg')])
        run.assert_not_called()

    def test_invalid_center_and_nonfinite_geometry_are_rejected(self):
        for key, value in [('object_center_px', ['bad', 10]), ('object_center_px', [float('nan'), 1]),
                           ('object_height_m', float('nan')), ('grasp_depth_m', float('inf'))]:
            with self.subTest(key=key, value=value):
                raw = {'grasp_px': [200, 300], 'object_center_px': [200, 300],
                       'place_px': [400, 240], 'matched_language_goal': True,
                       'object_height_m': .03, 'grasp_depth_m': .02}
                raw[key] = value
                self.assertFalse(CAS.validate_keypoints(raw)[0])

    def test_invalid_adjust_is_not_accepted_by_start_plan(self):
        for value in ('not-a-number', float('nan')):
            raw = dict(issue='other', intervene=True, matched_language_goal=True,
                       plan=[{'step': 'adjust', 'tilt_deg': value}])
            self.assertFalse(CAS.validate_start_plan(raw)[0])

    def test_requested_relook_performs_perception(self):
        calls = []
        state = {'confirmed': False}
        scope = functions('_apply_episode_plan', 'identity_fields', np=np, AUTO_RECOVERY_RULES=True,
            EP={'closed': False, 'last_result': 'reached'}, LIB={'intents': []},
            PATCH={}, GOAL_TARGET={}, NEIGHBOUR_STALL_ISSUES=set(), T=[1],
            _reset_patch=lambda **kw: None,
            set_goal_id=lambda confirmed, *a: state.update(confirmed=confirmed),
            goal_id_confirmed=lambda: state['confirmed'],
            _should_open_on_intervention=lambda **kw: False,
            do_relook=lambda why: calls.append(why) or 'relooked_matched',
            log=lambda **kw: None)
        result = scope['_apply_episode_plan'](
            {'issue': 'wrong_object', 'matched_language_goal': None, 'plan': [{'step': 'relook'}]}, None)
        self.assertEqual(len(calls), 1)
        self.assertTrue(any('relooked_matched' in x for x in result))

    def test_slow_reply_does_not_double_charge_perception_budget(self):
        budget = {'slow': 0, 'relook': 0}
        def apply_plan(*args, **kwargs):
            budget['relook'] += 1  # actual perception owns this budget
            return ['relook:relooked_matched']
        scope = functions('_finish_slow', 'identity_fields', AUTO_RECOVERY_RULES=True, P=budget, EP={'closed': False},
            MAX_SLOW=4, MAX_RELOOK=4, FB_LESSONS=[], ID_EVENTS=[], PATCH={}, CUR={}, T=[1], dec=0,
            _account_slow_rec=lambda rec: None, _apply_episode_plan=apply_plan,
            Image=SimpleNamespace(fromarray=lambda image: SimpleNamespace(save=lambda path: None)),
            ep_dir='/unused')
        result, detail = scope['_finish_slow'](
            {'issue': 'other', 'intervene': True, 'matched_language_goal': True,
             'plan': [{'step': 'relook'}], 'grasp_px': [10, 10]},
            {'cost_usd': 0}, None, None, 'descend', 'stalled', 'sync', None, 0)
        self.assertEqual(result, 'intervened')
        self.assertEqual(budget['relook'], 1)

    def test_async_result_is_discarded_after_target_refresh(self):
        applied, accounted = [], []
        box = {'job_id': 4, 'err': None, 'parsed': {'issue': 'ok'}, 'rec': {'cost_usd': 0}}
        job = dict(thread=SimpleNamespace(is_alive=lambda: False), result_box=box, job_id=4,
                   snap_dec=3, target_rev=1, closed=False, n_close=0, goal_confirmed=True,
                   intent='relook', result='reached', both=None, dep=None, mode='async', crit=None)
        scope = functions('_poll_async_slow', SLOW_JOB=job, dec=3, SLOW_STALE_INTENTS=1,
            GOAL_TARGET={'rev': 2}, EP={'closed': False}, goal_id_confirmed=lambda: True,
            _drain_orphan_slow_costs=lambda: None, _log_slow_result=lambda box: None,
            _clear_slow_job=lambda: None, _account_slow_rec=lambda r: accounted.append(r),
            _finish_slow=lambda *a: applied.append(a) or ('ok', {}))
        result = scope['_poll_async_slow']()
        self.assertEqual(applied, [])
        self.assertEqual(result[0], 'async_state_changed_discard')
        self.assertEqual(len(accounted), 1)

    def test_monitor_brief_keeps_start_plan_and_episode_lessons(self):
        with tempfile.TemporaryDirectory() as temp:
            received = []
            scope = functions('_prepare_slow_snapshot',
                astra_geometry=lambda *a, **kw: ({'status': 'unavailable'}, []),
                JX=SimpleNamespace(build_jev_experience_brief=lambda *a, **kw: received.append(kw) or {}),
                LIB={}, dsl=None, EP={'uses': {}, 'start_plan': {'phase_order': ['approach', 'place', 'retreat']}},
                FB_LESSONS=[{'lesson': 'raise before retry'}], T=[5], PATCH={}, GOAL_TARGET={},
                namespace=lambda: ({'opening': .08}, {}), facts=lambda *a: {'held': False},
                LANG='goal', goal_id_confirmed=lambda: True, ep_dir=temp,
                Image=SimpleNamespace(fromarray=lambda im: SimpleNamespace(save=lambda path: Path(path).write_bytes(b'fixture'))))
            scope['_prepare_slow_snapshot']('{}', None, 'async', None, 'sample')
            self.assertEqual(received[0].get('episode_start_plan'), scope['EP']['start_plan'])
            self.assertEqual(received[0].get('episode_lessons'), scope['FB_LESSONS'])

    def test_prompt_keeps_current_state_after_long_history(self):
        prompt = CAS.build_prompt({'recent_diag': ['x' * 13000],
            'slow_ctx': {'grasp_confirm': True, 'current_marker': 'latest_observation'},
            'experience_brief': {'episode_start_plan': {'phase_order': ['place', 'retreat']}}})
        self.assertIn('latest_observation', prompt)
        self.assertIn('episode_start_plan', prompt)

    def test_shutdown_harvests_worker_without_applying_plan(self):
        events = []
        scope = functions('_finalize_slow_jobs',
            SLOW_JOB={'thread': SimpleNamespace(join=lambda: events.append('join')),
                      'job_id': 1, 'result_box': {'rec': {'cost_usd': 0}, 'parsed': {'intervene': True}}},
            SLOW_ORPHAN_WAITERS=[], T=[10],
            _log_slow_result=lambda box: events.append('log'),
            _account_slow_rec=lambda rec: events.append('account'),
            _clear_slow_job=lambda: events.append('clear'),
            _drain_orphan_slow_costs=lambda: events.append('drain'), log=lambda **kw: None,
            _finish_slow=lambda *a: self.fail('shutdown must not move robot'))
        scope['_finalize_slow_jobs']()
        self.assertEqual(events, ['join', 'log', 'account', 'clear', 'drain'])


if __name__ == '__main__':
    unittest.main()
