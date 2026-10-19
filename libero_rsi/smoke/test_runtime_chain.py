"""Run the complete top-level harness with fake simulator, rotation, Jev and Codex.

This verifies wiring and result/error handling, NOT LIBERO physics or task success.
"""
import json
import os
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS


def fake_run(destination, fail_start=False, fine=False):
    events = []
    def module(name, **attrs):
        obj = types.ModuleType(name)
        obj.__dict__.update(attrs)
        sys.modules[name] = obj
        return obj
    class Rotation:
        @classmethod
        def from_quat(cls, value): return cls()
        @classmethod
        def from_euler(cls, *args): return cls()
        @classmethod
        def from_rotvec(cls, value): return cls()
        def as_euler(self, axes): return np.zeros(3)
        def as_rotvec(self): return np.zeros(3)
        def inv(self): return self
        def __mul__(self, other): return self
    class FakeEnv:
        def __init__(self, **kwargs):
            self.env = self
            self.count = 0
            names = ['gripper0_finger1_collision', 'gripper0_finger1_pad_collision',
                     'gripper0_finger2_collision', 'gripper0_finger2_pad_collision']
            model = types.SimpleNamespace(ngeom=4, geom_bodyid=np.zeros(4, dtype=int),
                geom_name2id=names.index, body_id2name=lambda i: 'world',
                camera_name2id=lambda name: 0, cam_fovy=np.array([45.]))
            data = types.SimpleNamespace(ncon=0, contact=[], cam_xpos=np.array([[0., 0., 1.]]),
                cam_xmat=np.eye(3).reshape(1,9), body_xpos=np.zeros((0,3)))
            self.sim = types.SimpleNamespace(model=model, data=data)
            self.pos = np.array([0., 0., .6])
            self.object_states_dict = {}; self.obj_body_id = {}; self.object_sites_dict = {}
            self.parsed_problem = {'goal_state': []}
        def seed(self, seed): pass
        def reset(self): pass
        def _check_success(self): return self.count >= 18
        def observation(self):
            return {'agentview_image': np.zeros((512,512,3), dtype=np.uint8),
                    'agentview_depth': np.ones((512,512,1)) * .5,
                    'robot0_eef_pos': self.pos.copy(),
                    'robot0_eef_quat': np.array([0.,0.,0.,1.]),
                    'robot0_gripper_qpos': np.array([.04,-.04])}
        def set_init_state(self, state): return self.observation()
        def step(self, action):
            self.count += 1
            # Only the fine-step scenario integrates, so a 1 cm command can reach.
            # Baseline actions stay put and the original call counts do not change.
            if fine:
                a = np.asarray(action, float).reshape(-1)
                self.pos = self.pos + np.clip(a[:3], -1, 1) * 0.05
            return self.observation(), 0, self._check_success(), {}
        def close(self): events.append('env_closed')
    task = types.SimpleNamespace(language='put butter in basket', problem_folder='fake', bddl_file='fake.bddl')
    suite = types.SimpleNamespace(get_task=lambda i: task, get_task_init_states=lambda i: [None])
    module('libero')
    module('libero.libero', benchmark=types.SimpleNamespace(get_benchmark_dict=lambda: {'fake_suite': lambda: suite}),
           get_libero_path=lambda *a: '/fake')
    module('libero.libero.envs', OffScreenRenderEnv=FakeEnv)
    module('robosuite'); module('robosuite.utils', camera_utils=types.SimpleNamespace(get_real_depth_map=lambda sim, depth: depth))
    module('scipy'); module('scipy.spatial'); module('scipy.spatial.transform', Rotation=Rotation)
    planned = ['fine_x_plus_1cm', 'finish'] if fine else ['ask_astra', 'relook', 'finish']
    seen = []
    def choose(state, instruction, criteria, **kwargs):
        chosen = planned[len(seen)] if len(seen) < len(planned) else 'finish'
        assert chosen in criteria, chosen
        facts = state.get('observed_facts') or {}
        seen.append({
            'choice': chosen,
            'has_fine': 'fine_x_plus_1cm' in criteria,
            'has_move': 'move_above_object' in criteria,
            'n_criteria': len(criteria),
            'instruction_has_fine': 'Fine-step menu' in (instruction or ''),
            'last_fine_step': facts['last_fine_step'] if 'last_fine_step' in facts else 'MISSING',
            'last_fine_step_text': facts.get('last_fine_step_text'),
        })
        events.append('jev:' + chosen)
        return chosen, {}, 0
    module('jev_choice', choose=choose)
    def codex(prompt, images, **kwargs):
        assert all(Path(p).exists() for p in images)
        if 'RED circle+cross' in prompt:
            events.append('identity')
            assert kwargs['model'] == 'gpt-6-astra'
            raw = {'is_language_goal_object': True, 'reason': 'test fixture'}
        elif 'perception module' in prompt:
            events.append('perception')
            assert kwargs['model'] == 'gpt-6-astra'
            raw = {'grasp_px': [200, 300], 'object_center_px': [200, 300],
                   'place_px': [400, 240], 'matched_language_goal': True,
                   'goal_object_named': 'butter', 'place_region_named': 'basket'}
        elif 'episode-START' in prompt:
            events.append('start_plan')
            if fail_start: raise RuntimeError('injected start failure')
            raw = {'matched_language_goal': True, 'issue': 'ok', 'intervene': False,
                   'plan': [], 'step1_must': 'approach', 'phase_order': ['approach','grasp','lift','place','retreat']}
        else:
            events.append('slow_plan')
            raw = {'matched_language_goal': True, 'is_goal_object': None,
                   'issue': 'ok', 'intervene': False, 'plan': []}
        return raw, {'cost_usd': 0., 'backend': 'codex', 'model': kwargs.get('model', 'test-planner')}
    lib = json.loads((ROOT/'libs'/'lib_seed_shape_prior.json').read_text())
    lib['recoveries'] = []
    library = Path(destination)/'library.json'
    library.write_text(json.dumps(lib))
    def exit_process(code): raise SystemExit(code)
    env = {'HORIZON': '40' if fine else '20', 'MAX_DEC': '5', 'GATE_JEV': '0', 'SLOW_MIN_GAP_S': '0'}
    if fine:
        env['FINE_STEPS'] = '1'
    with patch.dict(os.environ, env, clear=True), patch.object(CAS, 'call_codex', side_effect=codex), \
         patch.object(sys, 'argv', ['run_ep_fbpatch.py','fake_suite','0','0',str(library),destination]), \
         patch.object(os, '_exit', side_effect=exit_process):
        try:
            runpy.run_path(str(ROOT/'run_ep_fbpatch.py'), run_name='__main__')
        except SystemExit as ex:
            code = ex.code
    (Path(destination)/'calls.json').write_text(json.dumps(events))
    (Path(destination)/'seen.json').write_text(json.dumps(seen))
    return code


class RuntimeChainTests(unittest.TestCase):
    def run_scenario(self, fail_start, fine=False):
        with tempfile.TemporaryDirectory() as temp:
            args = [sys.executable, __file__, '--fake-run', temp]
            if fail_start: args.append('--fail-start')
            if fine: args.append('--fine')
            run = subprocess.run(args, capture_output=True, text=True, timeout=20)
            result_path = Path(temp)/'fake_suite_t0_i0'/'result.json'
            self.assertTrue(result_path.exists(), run.stdout + run.stderr)
            seen_path = Path(temp)/'seen.json'
            seen = json.loads(seen_path.read_text()) if seen_path.exists() else []
            return run, json.loads(result_path.read_text()), json.loads((Path(temp)/'calls.json').read_text()), seen

    def test_entire_entrypoint_with_fake_services(self):
        run, result, events, seen = self.run_scenario(False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotEqual(result['end_reason'], 'harness_error', result.get('error'))
        self.assertEqual(events[:3], ['perception', 'identity', 'start_plan'])
        self.assertIn('jev:ask_astra', events)
        self.assertIn('slow_plan', events)
        self.assertIn('jev:relook', events)
        self.assertEqual(events[-1], 'env_closed')
        self.assertEqual(result['decisions'], 3)
        self.assertEqual(result['astra_calls'], 3)
        self.assertEqual(result['planning_calls'], 3)
        self.assertNotIn('fine_steps_enabled', result)
        self.assertNotIn('fine_step_events', result)
        self.assertTrue(seen)
        self.assertTrue(all(not row['has_fine'] and row['has_move'] for row in seen))
        self.assertTrue(all(row['last_fine_step'] == 'MISSING' and not row['instruction_has_fine'] for row in seen))
        after_finish = events[events.index('jev:finish') + 1:]
        self.assertNotIn('perception', after_finish)

    def test_start_plan_failure_does_not_enter_jev_loop(self):
        run, result, events, seen = self.run_scenario(True)
        self.assertEqual(run.returncode, 1)
        self.assertEqual(result['end_reason'], 'harness_error')
        self.assertIn('episode_start_plan_failed', result['error'])
        self.assertFalse(any(e.startswith('jev:') for e in events))
        self.assertEqual(events[-1], 'env_closed')
        self.assertEqual(seen, [])

    def test_fine_steps_report_new_position_to_jev(self):
        run, result, events, seen = self.run_scenario(False, fine=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotEqual(result['end_reason'], 'harness_error', result.get('error'))
        self.assertTrue(result['fine_steps_enabled'])
        self.assertGreaterEqual(len(seen), 2, seen)
        first, second = seen[0], seen[1]
        self.assertEqual(first['choice'], 'fine_x_plus_1cm')
        self.assertTrue(first['has_fine'] and first['has_move'] and first['instruction_has_fine'])
        self.assertIsNone(first['last_fine_step'])
        self.assertGreater(first['n_criteria'], 18)
        event = result['fine_step_events'][0]
        self.assertEqual(event['intent'], 'fine_x_plus_1cm')
        self.assertEqual(event['result'], 'reached')
        self.assertEqual(event['commanded_delta_m'], [0.01, 0.0, 0.0])
        self.assertFalse(event['gripper_closed'])
        self.assertAlmostEqual(event['ee_xyz_m'][0], 0.01, delta=0.003)
        self.assertAlmostEqual(event['ee_xyz_m'][2], 0.6, places=3)
        self.assertIn('fine_x_plus_1cm reached', event['text'])
        self.assertIn('gripper open', event['text'])
        self.assertEqual(second['last_fine_step_text'], event['text'])
        self.assertEqual(second['last_fine_step']['ee_xyz_m'], event['ee_xyz_m'])
        self.assertIn('jev:fine_x_plus_1cm', events)
        self.assertIn('jev:finish', events)


if __name__ == '__main__':
    if '--fake-run' in sys.argv:
        raise SystemExit(fake_run(sys.argv[2], '--fail-start' in sys.argv, '--fine' in sys.argv))
    unittest.main()
