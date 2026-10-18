"""Execute production functions with model/environment doubles; no simulator or network."""
import ast
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS
import jev_experience as JX
import validate

TREE = ast.parse((ROOT / 'run_ep_fbpatch.py').read_text())


def functions(*names, **bindings):
    nodes = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    scope = {'json': json, 'os': __import__('os'), **bindings}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'run_ep_fbpatch.py'), 'exec'), scope)
    return scope


class RegressionTests(unittest.TestCase):
    def test_start_plan_cannot_confirm_identity(self):
        for match in (True, None, False):
            with self.subTest(match=match), tempfile.TemporaryDirectory() as temp:
                plan = dict(matched_language_goal=match, issue='blocked_by_neighbour', intervene=True,
                            step1_must='push_aside', plan=[{'step': 'push_aside'}], phase_order=['clear_neighbour'])
                gid = {'confirmed': False, 'events': []}
                ep = {}
                applied = []
                scope = functions('_run_episode_start_plan', 'set_goal_id', 'goal_id_confirmed', 'identity_fields',
                    JX=JX, LIB={'intents': [], 'lessons': []}, dsl=None,
                    EP=ep, GOAL_ID=gid, T=[37], START_ID={'final': {}, 'verify': {}}, per0={},
                    LANG='put object in basket', ep_dir=temp, log=lambda **kw: None,
                    _apply_episode_plan=lambda *a, **kw: applied.append(a), DEP0=None)
                with patch.object(CAS, 'harness_call_start_plan', return_value=(plan, {'backend': 'codex_start'})) as planner:
                    result = scope['_run_episode_start_plan']()
                    self.assertEqual(planner.call_args.kwargs['mode'], 'codex')
                self.assertIsNone(result['error'])
                self.assertFalse(gid['confirmed'])
                self.assertFalse(result['applied'])
                self.assertEqual(applied, [])
                self.assertEqual(ep['force_next_intent'], 'relook')
                artifact = json.loads((Path(temp) / 'start_plan.json').read_text())
                self.assertEqual(artifact['plan']['plan'], [{'step': 'relook'}])
                self.assertEqual(artifact['proposed_plan']['plan'], [{'step': 'push_aside'}])

    def test_start_plan_failure_does_not_switch_to_astra(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = functions('_run_episode_start_plan',
                JX=JX, LIB={'intents': [], 'lessons': []}, dsl=None, EP={}, T=[0],
                START_ID={'final': {}, 'verify': {}}, per0={}, LANG='goal', ep_dir=temp,
                goal_id_confirmed=lambda: True, log=lambda **kw: None)
            with patch.object(CAS, 'harness_call_start_plan', side_effect=RuntimeError('codex unavailable')):
                result = scope['_run_episode_start_plan']()
            self.assertEqual(result['mode'], 'codex')
            self.assertEqual(result['error'], 'codex unavailable')
            self.assertFalse(result['applied'])

    def test_goal_event_has_simulation_step(self):
        scope = functions('set_goal_id', T=[29], GOAL_ID={'events': []})
        scope['set_goal_id'](True, 'relook')
        self.assertEqual(scope['GOAL_ID']['events'][0]['t'], 29)

    def test_removed_backends_are_rejected_without_calls(self):
        for mode in ('mock', 'dual', 'astra', 'off', '0', 'auto', '1', 'on'):
            with self.subTest(mode=mode), patch.object(CAS, 'call_codex') as codex:
                with self.assertRaisesRegex(ValueError, 'unsupported_planning_backend'):
                    CAS.call_monitor({}, mode=mode)
                with self.assertRaisesRegex(ValueError, 'unsupported_planning_backend'):
                    CAS.call_start_plan({}, mode=mode)
                with self.assertRaisesRegex(ValueError, 'unsupported_planning_backend'):
                    CAS.call_perceive({}, mode=mode)
                codex.assert_not_called()

    def test_async_uses_snapshot_and_main_thread_logging(self):
        with tempfile.TemporaryDirectory() as temp:
            # This separate diagnostic output is deliberately privileged and must never enter context.
            (Path(temp) / 'episode_diag.jsonl').write_text('{"obj_xyz":[123,456,789]}\n')
            (Path(temp) / 'decisions.jsonl').write_text('{"intent":"lift"}\n')
            patch_state = {'offset': [1]}
            target = {'base': {'pixel': [2]}}
            events = []
            scope = functions('_prepare_slow_snapshot', '_slow_worker', '_log_slow_result',
                T=[5],
                JX=SimpleNamespace(build_jev_experience_brief=lambda *a, **kw: {'lessons': ['prior']}),
                LIB={}, dsl=None, EP={'uses': {}}, LANG='goal', PATCH=patch_state, GOAL_TARGET=target,
                goal_id_confirmed=lambda: True, ep_dir=temp,
                Image=SimpleNamespace(fromarray=lambda image: SimpleNamespace(
                    save=lambda path: Path(path).write_bytes(b'test fixture'))),
                log=lambda **kw: events.append((threading.current_thread(), kw)))
            snapshot = scope['_prepare_slow_snapshot']('{}', None, 'async', None, 'async_1')
            self.assertNotIn('obj_xyz', json.dumps(snapshot))
            self.assertEqual(snapshot['coding_context']['recent_diag'], [{'intent': 'lift'}])
            entered, release = threading.Event(), threading.Event()
            seen = []
            def monitor(ctx, mode):
                self.assertEqual(mode, 'codex')
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('test synchronization failed')
                seen.append(ctx)
                return {'issue': 'ok', 'plan': [], 'intervene': False}, {'backend': mode, 'cost_usd': 0}
            box = {}
            with patch.object(CAS, 'call_monitor', side_effect=monitor):
                thread = threading.Thread(target=scope['_slow_worker'], args=(1, box, snapshot))
                thread.start()
                try:
                    self.assertTrue(entered.wait(2))
                    patch_state['offset'][0] = 99
                    target['base']['pixel'][0] = 99
                    scope['T'][0] = 99
                    self.assertEqual(events, [])
                finally:
                    release.set()
                    thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertNotIn('err', box)
            self.assertEqual(seen[0]['patch_goal']['patch']['offset'], [1])
            self.assertEqual(seen[0]['patch_goal']['goal_id']['target']['base']['pixel'], [2])
            self.assertEqual(events, [])
            scope['_log_slow_result'](box)
            self.assertEqual(events[0][0], threading.current_thread())
            self.assertEqual(events[0][1]['t'], 5)

    def test_async_launcher_wires_worker_with_fixed_codex_role(self):
        scope = functions('_launch_async_slow', '_slow_worker', '_next_slow_job_id', '_new_slow_result_box',
            threading=threading, time=SimpleNamespace(time=lambda: 20),
            SLOW_JOB={}, SLOW_JOB_SEQ=[0], SLOW_MIN_GAP_S=5,
            P={'slow': 0, 'last_astra_wall': 0}, MAX_SLOW=4, dec=2,
            _drain_orphan_slow_costs=lambda: None, _combo_rgb=lambda: ([], [], []),
            _slow_ctx=lambda *a: '{}',
            _prepare_slow_snapshot=lambda *a: {'t': 10, 'coding_context': {'captured': True}})
        with patch.object(CAS, 'call_monitor', return_value=(
                {'issue': 'ok', 'plan': []}, {'backend': 'codex', 'cost_usd': 0})) as monitor:
            result, _ = scope['_launch_async_slow']('descend', 'stalled')
            self.assertEqual(result, 'async_launched')
            scope['SLOW_JOB']['thread'].join(2)
            self.assertFalse(scope['SLOW_JOB']['thread'].is_alive())
            monitor.assert_called_once_with({'captured': True}, mode='codex')
        box = scope['SLOW_JOB']['result_box']
        self.assertIsNone(box['err'])
        self.assertEqual(box['rec']['backend'], 'codex')

    def test_codex_failure_has_no_astra_fallback(self):
        astra = SimpleNamespace(call=lambda *a, **kw: self.fail('Astra must not plan'))
        scope = functions('_slow_worker', AC=astra)
        box = {}
        with patch.object(CAS, 'call_monitor', side_effect=RuntimeError('codex unavailable')):
            scope['_slow_worker'](1, box, {'t': 1, 'coding_context': {}})
        self.assertTrue(box['done'])
        self.assertIn('codex unavailable', str(box['err']))
        self.assertNotIn('parsed', box)
        self.assertTrue(box['events'][0]['coding_agent_slow_error'])

    def test_sync_slow_uses_codex_and_reports_failure(self):
        events, applied = [], []
        scope = functions('_sync_slow', SLOW_JOB={}, P={'slow': 0}, MAX_SLOW=4, T=[10], dec=2,
            _drain_orphan_slow_costs=lambda: None, _combo_rgb=lambda: ('image', None, 'depth'),
            _slow_ctx=lambda *a: '{}', _next_slow_job_id=lambda: 1,
            _prepare_slow_snapshot=lambda *a: {'coding_context': {'captured': True}},
            _finish_slow=lambda *a: applied.append(a) or ('ok', {}),
            log=lambda **kw: events.append(kw),
            AC=SimpleNamespace(CapReached=type('CapReached', (Exception,), {}),
                               call=lambda *a, **kw: self.fail('Astra must not plan')))
        with patch.object(CAS, 'call_monitor', return_value=({'issue': 'ok'}, {'backend': 'codex'})) as monitor:
            self.assertEqual(scope['_sync_slow']('descend', 'stalled', 'hard_stall')[0], 'ok')
            monitor.assert_called_once_with({'captured': True}, mode='codex')
        self.assertEqual(len(applied), 1)
        with patch.object(CAS, 'call_monitor', side_effect=RuntimeError('unavailable')):
            result, detail = scope['_sync_slow']('descend', 'stalled', 'hard_stall')
        self.assertTrue(result.startswith('sync_error:coding_agent:'))
        self.assertIsNone(detail)
        self.assertEqual(len(applied), 1)
        self.assertTrue(events[-1]['coding_agent_slow_error'])

    def test_fallback_refreshes_astra_then_plans_with_codex(self):
        events = []
        scope = functions('do_fallback', P={'relook': 0}, MAX_RELOOK=4, PATCH={'offset': 1},
            _abandon_async_slow=lambda **kw: events.append('abandon'),
            do_relook=lambda reason: events.append('astra_perception') or 'relooked_matched',
            _sync_slow=lambda *a: events.append('codex_plan') or ('intervened', {'plan_done': ['adjust']}))
        result, detail = scope['do_fallback']('stall')
        self.assertEqual(events, ['abandon', 'astra_perception', 'codex_plan'])
        self.assertEqual(result, 'intervened')
        self.assertEqual(detail['perception'], 'relooked_matched')
        scope['do_relook'] = lambda *a: 'relook_error'
        events.clear()
        self.assertEqual(scope['do_fallback']('stall'), ('fallback_perception_error', None))
        self.assertNotIn('codex_plan', events)

    def test_recovery_error_is_logged_and_propagated(self):
        loop = next(n for n in ast.walk(TREE) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == 'rc')
        events = []
        def fail(*args):
            raise ValueError('bad trigger')
        scope = dict(LIB={'recoveries': [{'id': 'broken', 'trigger': 'bad'}]},
                     forced=None, dsl=SimpleNamespace(evaluate=fail), ns={}, dec=3, T=[7],
                     log=lambda **kw: events.append(kw))
        with self.assertRaisesRegex(ValueError, 'bad trigger'):
            exec(compile(ast.Module(body=[loop], type_ignores=[]), '<recovery>', 'exec'), scope)
        self.assertEqual(events[0]['recovery_error'], 'broken')

    def test_all_shipped_libraries_validate(self):
        for path in (ROOT / 'libs').rglob('*.json'):
            with self.subTest(path=path.name):
                self.assertTrue(validate.validate(json.loads(path.read_text())))


if __name__ == '__main__':
    unittest.main()
