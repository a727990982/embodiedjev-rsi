"""Pose menu: default-off injection, yaw/align/home geometry, servo path. No simulator."""
import ast
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import validate
from test_review_regressions import functions

TREE = ast.parse((ROOT / 'run_ep_fbpatch.py').read_text())
LIBS = ROOT / 'libs'
FROZEN = LIBS / 'ee_pg_lib_frozen.json'
EASY = LIBS / 'ee_pg_easy_v2_lib.json'
POSE = LIBS / 'ee_pg_pose_menu_lib.json'

POSE_FUNCS = (
    '_pose_menu_enabled', '_pose_menu_kinds', '_pose_kind_of', '_pose_menu_intent_specs',
    '_pose_menu_tip', '_pose_menu_visible', '_install_pose_menu',
    '_wrap_deg', '_yaw_step', '_yaw_toward', '_unit_xy', '_object_side_dirs',
    '_pose_body_radius', '_yaw_face_point', '_pose_clearance_m', '_align_target',
    '_pose_grip', '_finite3', '_pose_note', '_execute_pose_menu', 'facts',
)


def load(path):
    return json.loads(path.read_text())


def pose_scope(**bindings):
    return functions(*POSE_FUNCS, np=np, **bindings)


class FakeRot:
    def __init__(self, err=None):
        self.err = err

    @classmethod
    def from_quat(cls, value):
        return cls(err=float(WORLD['err']))

    @classmethod
    def from_euler(cls, *args):
        return cls()

    @classmethod
    def from_rotvec(cls, value):
        return cls()

    def inv(self):
        return self

    def __mul__(self, other):
        err = self.err if self.err is not None else getattr(other, 'err', None)
        return FakeRot(err=err)

    def as_rotvec(self):
        err = 0.0 if self.err is None else float(self.err)
        return np.array([err, 0.0, 0.0])


WORLD = {'err': 1.0, 'steps': 0, 'mode': 'hold', 'goal': np.zeros(3)}


class PoseMenuTests(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict(os.environ, {'POSE_MENU': '0', 'POSE_MENU_KINDS': ''}, clear=False)
        self._env.start()
        for key in ('POSE_CLEAR_SMALL_M', 'POSE_CLEAR_MEDIUM_M', 'POSE_CLEAR_LARGE_M'):
            os.environ.pop(key, None)

    def tearDown(self):
        self._env.stop()

    def test_shipped_pose_library_extends_easy_v2_only(self):
        easy, pose, frozen = load(EASY), load(POSE), load(FROZEN)
        validate.validate(pose)
        scope = pose_scope()
        specs = {s['name']: s for s in scope['_pose_menu_intent_specs']()}
        self.assertEqual(set(specs), {
            'rotate_yaw_plus_15', 'rotate_yaw_minus_15', 'rotate_yaw_plus_45', 'rotate_yaw_minus_45',
            'rotate_yaw_toward_astra', 'return_home',
            *(f'align_{side}_{clear}' for side in ('top', 'left', 'right', 'front', 'back')
              for clear in ('small', 'medium', 'large')),
        })
        pose_by = {i['name']: i for i in pose['intents']}
        for name, spec in specs.items():
            self.assertEqual(pose_by[name]['definition'], spec['definition'])
            self.assertEqual(pose_by[name]['action'], 'pose_menu')
            self.assertEqual(pose_by[name]['fingers'], 'keep')
        easy_by = {i['name']: i for i in easy['intents']}
        for name, intent in easy_by.items():
            self.assertEqual(pose_by[name]['definition'], intent['definition'])
            self.assertEqual(pose_by[name].get('refpoint'), intent.get('refpoint'))
        self.assertEqual(pose['predicates'], easy['predicates'])
        self.assertEqual(pose['recoveries'], easy['recoveries'])
        self.assertEqual(pose['lessons'], easy['lessons'])
        self.assertEqual(pose['intent_instructions'], easy['intent_instructions'] + scope['_pose_menu_tip'](pose))
        self.assertNotIn('pose_menu', json.dumps(frozen))
        self.assertNotIn('rotate_yaw_plus_15', json.dumps(easy))
        self.assertNotIn('return_home', json.dumps(frozen))

    def test_default_env_does_not_change_a_loaded_library(self):
        lib = load(FROZEN)
        before = json.dumps(lib)
        scope = pose_scope(LIB=lib)
        self.assertFalse(scope['_pose_menu_enabled']())
        self.assertIsNone(scope['_pose_menu_kinds']())
        self.assertFalse(scope['_install_pose_menu'](lib))
        self.assertEqual(json.dumps(lib), before)
        self.assertNotIn('pose_action_outcome', self._facts(scope, lib))

    def test_flag_injects_and_kinds_filter(self):
        scope = pose_scope(LIB=load(EASY))
        with patch.dict(os.environ, {'POSE_MENU': '1', 'POSE_MENU_KINDS': ''}):
            lib = load(FROZEN)
            self.assertTrue(scope['_install_pose_menu'](lib))
            validate.validate(lib)
            names = [i['name'] for i in lib['intents']]
            self.assertIn('rotate_yaw_plus_15', names)
            self.assertIn('align_left_medium', names)
            self.assertIn('return_home', names)
            self.assertEqual(names[-1], 'finish')
            self.assertIn('Pose menu:', lib['intent_instructions'])
            self.assertEqual(len(names), len(load(FROZEN)['intents']) + 21)
        with patch.dict(os.environ, {'POSE_MENU': '1', 'POSE_MENU_KINDS': 'home'}):
            lib = load(FROZEN)
            scope['_install_pose_menu'](lib)
            added = [i['name'] for i in lib['intents'] if i.get('action') == 'pose_menu']
            self.assertEqual(added, ['return_home'])
        with patch.dict(os.environ, {'POSE_MENU': '0', 'POSE_MENU_KINDS': 'rotate'}):
            lib = load(POSE)
            scope['_install_pose_menu'](lib)
            kinds = {scope['_pose_kind_of'](i) for i in lib['intents'] if i.get('action') == 'pose_menu'}
            self.assertEqual(kinds, {'rotate'})
            self.assertIn('move_above_object', {i['name'] for i in lib['intents']})

    def test_yaw_steps_hold_the_short_arc(self):
        scope = pose_scope()
        self.assertEqual(scope['_yaw_step'](10, 15), (25.0, 15.0))
        self.assertEqual(scope['_yaw_step'](10, -45), (-35.0, -45.0))
        new, step = scope['_yaw_step'](170, 45)
        self.assertAlmostEqual(step, 45.0)
        self.assertAlmostEqual(new, -145.0)
        self.assertEqual(scope['_yaw_toward'](0, 40, 15), (15.0, 15.0))
        self.assertEqual(scope['_yaw_toward'](0, -10, 15), (-10.0, -10.0))
        self.assertEqual(scope['_yaw_toward'](12, 12, 15)[1], 0.0)
        turned, delta = scope['_yaw_toward'](170, -170, 15)
        self.assertAlmostEqual(delta, 15.0)
        self.assertAlmostEqual(turned, -175.0)

    def test_align_uses_astra_estimates_and_faces_the_object(self):
        scope = pose_scope()
        self.assertEqual(scope['_pose_clearance_m']('small'), 0.05)
        self.assertEqual(scope['_pose_clearance_m']('medium'), 0.08)
        self.assertEqual(scope['_pose_clearance_m']('large'), 0.15)
        objc = np.array([0.0, 0.0, 0.90])
        grasp = np.array([0.10, 0.0, 0.50])
        tcp = np.array([0.20, 0.30, 0.80])
        home = np.array([0.0, -1.0])
        radius = scope['_pose_body_radius'](grasp, objc, 0.02)
        self.assertAlmostEqual(radius, 0.10)
        top, top_yaw = scope['_align_target']('top', 0.08, tcp, grasp, objc, 0.02, home)
        self.assertIsNone(top_yaw)
        np.testing.assert_allclose(top, [0.0, 0.0, 0.98])
        left, left_yaw = scope['_align_target']('left', 0.05, tcp, grasp, objc, 0.02, home)
        right, _ = scope['_align_target']('right', 0.05, tcp, grasp, objc, 0.02, home)
        front, _ = scope['_align_target']('front', 0.05, tcp, grasp, objc, 0.02, home)
        back, _ = scope['_align_target']('back', 0.05, tcp, grasp, objc, 0.02, home)
        self.assertAlmostEqual(left[2], tcp[2])
        self.assertAlmostEqual(right[2], tcp[2])
        np.testing.assert_allclose(left[:2] - objc[:2], -(right[:2] - objc[:2]))
        np.testing.assert_allclose(front[:2] - objc[:2], -(back[:2] - objc[:2]))
        np.testing.assert_allclose(np.linalg.norm(left[:2] - objc[:2]), radius + 0.05)
        # Front is toward the home gripper, which sits on -Y.
        self.assertLess(front[1], 0.0)
        self.assertGreater(back[1], 0.0)
        forward = np.array([np.cos(np.deg2rad(left_yaw)), np.sin(np.deg2rad(left_yaw))])
        toward = objc[:2] - left[:2]
        toward = toward / np.linalg.norm(toward)
        dirs = scope['_object_side_dirs'](np.zeros(2), np.zeros(2), None)
        np.testing.assert_allclose(dirs['front'], [0.0, -1.0])
        self.assertGreater(float(np.dot(forward, toward)), 0.99)
        opening = np.array([-np.sin(np.deg2rad(left_yaw)), np.cos(np.deg2rad(left_yaw))])
        self.assertLess(abs(float(np.dot(opening, toward))), 0.05)

    def test_execute_routes_each_family_through_servo(self):
        calls = []
        tcp = np.array([0.2, 0.3, 0.8])

        ep = {'closed': True, 'pose_menu_note': None, 'pose_menu_counts': {}}
        patch_state = {'yaw_deg': None, 'tilt_deg': None, 'tilt_dir_deg': None}
        rshape = {'yaw': 20.0, 'tilt': 5.0, 'tdir': 10.0, 'dz': 0.1}
        per = {'yaw': 40.0, 'grasp': np.array([0.1, 0.0, 0.5]), 'objc': np.array([0.0, 0.0, 0.9]), 'obj_h': 0.02}
        seen = []

        def fake_servo(target, grip, gain=1.0, max_steps=60, tol=0.006, ori_tol=None):
            calls.append({'target': np.asarray(target, float).copy(), 'grip': grip, 'ori_tol': ori_tol})
            seen.append(patch_state.get('yaw_deg'))
            return 'reached'

        scope = pose_scope(
            EP=ep, PATCH=patch_state, PER=per, RSHAPE=rshape, servo=fake_servo, eef=lambda: tcp.copy(),
            _cmd_yaw_deg=lambda: 10.0, log=lambda **kw: None, T=[3],
            HOME_EEF=np.array([0.4, 0.1, 0.7]), CP=np.array([0.0, -0.5, 1.2]),
        )
        spec = {s['name']: s for s in scope['_pose_menu_intent_specs']()}
        result = scope['_execute_pose_menu'](spec['rotate_yaw_plus_15'], {'tcp': tcp})
        self.assertEqual(result, 'reached')
        self.assertEqual(calls[0]['grip'], 1.0)
        self.assertEqual(calls[0]['ori_tol'], 0.08)
        np.testing.assert_allclose(calls[0]['target'], tcp)
        self.assertAlmostEqual(patch_state['yaw_deg'], 25.0)
        self.assertIn('position held', ep['pose_menu_note'])
        self.assertIn('10.0 -> 25.0', ep['pose_menu_note'])

        calls.clear()
        patch_state['yaw_deg'] = 40.0
        scope['_cmd_yaw_deg'] = lambda: 40.0
        result = scope['_execute_pose_menu'](spec['rotate_yaw_toward_astra'], {'tcp': tcp})
        self.assertEqual(result, 'at_target')
        self.assertEqual(calls, [])
        self.assertIn('already aligned', ep['pose_menu_note'])

        calls.clear()
        scope['_cmd_yaw_deg'] = lambda: 0.0
        patch_state['yaw_deg'] = None
        scope['_execute_pose_menu'](spec['align_left_small'], {'tcp': tcp})
        self.assertAlmostEqual(calls[0]['target'][2], tcp[2])
        self.assertIsNotNone(calls[0]['ori_tol'])
        self.assertIn('height held', ep['pose_menu_note'])
        self.assertIn('0.050 m', ep['pose_menu_note'])

        calls.clear()
        yaw_before = patch_state['yaw_deg']
        scope['_execute_pose_menu'](spec['align_top_medium'], {'tcp': tcp})
        np.testing.assert_allclose(calls[0]['target'], [0.0, 0.0, 0.98])
        self.assertIsNone(calls[0]['ori_tol'])
        self.assertEqual(patch_state['yaw_deg'], yaw_before)
        self.assertIn('object top centre', ep['pose_menu_note'])

        calls.clear()
        per['grasp'] = None
        self.assertEqual(scope['_execute_pose_menu'](spec['align_front_large'], {'tcp': tcp}), 'pose_menu_no_estimate')
        self.assertEqual(calls, [])
        self.assertIn('missing grasp point', ep['pose_menu_note'])

        calls.clear()
        scope['_execute_pose_menu'](spec['return_home'], {'tcp': tcp})
        np.testing.assert_allclose(calls[-1]['target'], [0.4, 0.1, 0.7])
        self.assertEqual(seen[-1], 0.0)
        self.assertIsNone(patch_state['yaw_deg'])
        self.assertIsNone(patch_state['tilt_deg'])
        self.assertIsNone(rshape['yaw'])
        self.assertEqual(rshape['dz'], 0.1)
        self.assertIn('fingers unchanged', ep['pose_menu_note'])
        self.assertIn('overrides cleared', ep['pose_menu_note'])
        self.assertEqual(ep['pose_menu_counts']['rotate'], 2)
        self.assertEqual(ep['pose_menu_counts']['align'], 3)
        self.assertEqual(ep['pose_menu_counts']['home'], 1)

    def test_outcome_appears_in_jev_facts_only_when_the_menu_is_on(self):
        lib = {'intents': [{'name': 'finish', 'action': 'finish'}]}
        scope = pose_scope(LIB=lib, EP={'pose_menu_note': 'return home: servo reached'})
        self.assertNotIn('pose_action_outcome', self._facts(scope, lib))
        with patch.dict(os.environ, {'POSE_MENU': '1'}):
            facts = self._facts(scope, lib)
        self.assertEqual(facts['pose_action_outcome'], 'return home: servo reached')
        lib['intents'].append({'name': 'return_home', 'action': 'pose_menu', 'pose': 'home'})
        os.environ['POSE_MENU'] = '0'
        self.assertIn('pose_action_outcome', self._facts(scope, lib))

    def test_servo_keeps_position_success_unless_orientation_is_requested(self):
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'servo')
        self.assertEqual(fn.args.args[-1].arg, 'ori_tol')
        self.assertIsNone(fn.args.defaults[-1].value)
        body = ast.get_source_segment((ROOT / 'run_ep_fbpatch.py').read_text(), fn)
        execute = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
        execute_src = ast.get_source_segment((ROOT / 'run_ep_fbpatch.py').read_text(), execute)
        self.assertLess(execute_src.find("act == 'pose_menu'"), execute_src.find("it['refpoint']"))
        menu = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_pose_menu')
        menu_src = ast.get_source_segment((ROOT / 'run_ep_fbpatch.py').read_text(), menu)
        self.assertIn('servo(', menu_src)
        self.assertNotIn('perceive(', menu_src)
        self.assertNotIn('segment', menu_src.lower())

        pos = np.zeros(3)
        T = [0]
        WORLD.update(err=1.0, steps=0, mode='hold', goal=np.array([0.05, 0.0, 0.0]))

        def eef():
            return pos.copy()

        def step(action):
            WORLD['steps'] += 1
            T[0] += 1
            if WORLD['mode'] == 'translate':
                pos[:] = pos + 0.5 * (WORLD['goal'] - pos)
            elif WORLD['mode'] == 'spin':
                WORLD['err'] = max(0.0, WORLD['err'] - 0.25)

        scope = functions(
            'servo', np=np, eef=eef, step=step, T=T, DONE=[False], HORIZON=1000, Rot=FakeRot, Q0=FakeRot(),
            obs={'robot0_eef_quat': np.array([0.0, 0.0, 0.0, 1.0])},
            PATCH={'yaw_deg': 0.0, 'tilt_deg': 0.0, 'tilt_dir_deg': 0.0},
            RSHAPE={'yaw': None, 'tilt': None, 'tdir': None},
            SHAPE={'yaw': None, 'tilt': 0.0, 'tdir': 0.0},
            PER={'yaw': 0.0},
        )
        WORLD['mode'] = 'translate'
        WORLD['steps'] = 0
        self.assertEqual(scope['servo'](WORLD['goal'].copy(), -1), 'reached')
        self.assertGreater(WORLD['steps'], 1)

        pos[:] = WORLD['goal']
        WORLD.update(mode='hold', err=1.0, steps=0)
        self.assertEqual(scope['servo'](WORLD['goal'].copy(), -1), 'reached')
        self.assertEqual(WORLD['steps'], 1)

        WORLD.update(mode='hold', err=1.0, steps=0)
        self.assertEqual(scope['servo'](WORLD['goal'].copy(), -1, ori_tol=0.05), 'stalled')
        self.assertEqual(WORLD['steps'], 8)

        WORLD.update(mode='spin', err=1.0, steps=0)
        self.assertEqual(scope['servo'](WORLD['goal'].copy(), -1, ori_tol=0.05), 'reached')
        self.assertGreater(WORLD['steps'], 1)
        self.assertLess(WORLD['steps'], 8)
        self.assertIn('ori_tol', body)

    def _facts(self, scope, lib):
        scope['LIB'] = lib
        tcp = np.zeros(3)
        ns = {
            'tcp': tcp, 'opening': 0.04, 'closed_cmd': False, 'left_contact': False, 'right_contact': False,
            'contact': False, 'opening_at_close': 0.04, 'tcp_at_close': tcp, 'n_close': 0, 'released': False,
            'seen_held': False, 'grasp': tcp, 'objc': tcp, 'place': tcp, 'obj_h': 0.05, 'prev': 'none',
            'n_same': 0, 'last_result': 'none',
        }
        return scope['facts'](ns, {})


if __name__ == '__main__':
    unittest.main()
