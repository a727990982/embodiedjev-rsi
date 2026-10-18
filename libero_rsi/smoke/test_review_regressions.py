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
                    EPISODE_START_PLAN='codex', JX=JX, LIB={'intents': [], 'lessons': []}, dsl=None,
                    EP=ep, GOAL_ID=gid, T=[37], START_ID={'final': {}, 'verify': {}}, per0={},
                    LANG='put object in basket', ep_dir=temp, log=lambda **kw: None,
                    _apply_episode_plan=lambda *a, **kw: applied.append(a), DEP0=None)
                with patch.object(CAS, 'harness_call_start_plan', return_value=(plan, {'backend': 'codex_start'})):
                    result = scope['_run_episode_start_plan']()
                self.assertIsNone(result['error'])
                self.assertFalse(gid['confirmed'])
                self.assertFalse(result['applied'])
                self.assertEqual(applied, [])
                self.assertEqual(ep['force_next_intent'], 'relook')
                artifact = json.loads((Path(temp) / 'start_plan.json').read_text())
                self.assertEqual(artifact['plan']['plan'], [{'step': 'relook'}])
                self.assertEqual(artifact['proposed_plan']['plan'], [{'step': 'push_aside'}])

    def test_goal_event_has_simulation_step(self):
        scope = functions('set_goal_id', T=[29], GOAL_ID={'events': []})
        scope['set_goal_id'](True, 'relook')
        self.assertEqual(scope['GOAL_ID']['events'][0]['t'], 29)

    def test_unsupported_backend_values_raise(self):
        for resolver in (CAS.backend_mode, CAS.perceive_backend_mode, CAS.start_plan_backend_mode):
            for value in ('mock', '1', 'on'):
                with self.subTest(resolver=resolver.__name__, value=value):
                    with self.assertRaises(ValueError):
                        resolver(value)

    def test_dual_calls_codex(self):
        with patch.object(CAS, 'call_codex', return_value=(
            {'issue': 'ok', 'intervene': False, 'plan': []}, {'backend': 'codex'})) as codex:
            _, rec = CAS.call_monitor({}, mode='dual')
        codex.assert_called_once()
        self.assertEqual(rec['backend'], 'codex')

    def test_async_uses_snapshot_and_main_thread_logging(self):
        with tempfile.TemporaryDirectory() as temp:
            # This separate diagnostic output is deliberately privileged and must never enter context.
            (Path(temp) / 'episode_diag.jsonl').write_text('{"obj_xyz":[123,456,789]}\n')
            (Path(temp) / 'decisions.jsonl').write_text('{"intent":"lift"}\n')
            patch_state = {'offset': [1]}
            target = {'base': {'pixel': [2]}}
            events = []
            scope = functions('_prepare_slow_snapshot', '_slow_worker', '_log_slow_result',
                CODING_AGENT_SLOW='codex', T=[5], ASTRA_EFFORT='xhigh',
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
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('test synchronization failed')
                seen.append(ctx)
                return {'issue': 'ok', 'plan': [], 'intervene': False}, {'backend': mode, 'cost_usd': 0}
            box = {}
            with patch.object(CAS, 'call_monitor', side_effect=monitor):
                thread = threading.Thread(target=scope['_slow_worker'], args=(b'', '', {}, 1, box, snapshot))
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

    def test_dual_failure_still_runs_astra_and_records_error(self):
        astra = SimpleNamespace(call=lambda *a, **kw: ({'issue': 'ok'}, {'cost_usd': 0.1}))
        scope = functions('_slow_worker', AC=astra)
        box = {}
        with patch.object(CAS, 'call_monitor', side_effect=RuntimeError('codex unavailable')):
            scope['_slow_worker'](b'', '', {}, 1, box,
                {'mode': 'dual', 't': 1, 'effort': 'high', 'coding_context': {}})
        self.assertNotIn('err', box)
        self.assertEqual(box['parsed']['issue'], 'ok')
        self.assertIn('codex unavailable', box['rec']['coding_dual']['rec']['err'])

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
