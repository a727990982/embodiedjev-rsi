"""Fine-step menu: default off, clamped, servoed, reported. No simulator."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dsl_libero as dsl
import fine_steps as FS
import validate as V
from test_review_regressions import functions

V2 = ROOT / 'libs' / 'ee_pg_easy_v2_lib.json'
FROZEN = ROOT / 'libs' / 'ee_pg_lib_frozen.json'
OVERLAY = ROOT / 'libs' / 'ee_pg_fine_steps.json'


def load_funcs(*names, **bindings):
    return functions(*names, **{'np': np, **bindings})


def _ns(tcp):
    return {'tcp': np.asarray(tcp, float), 'fine_unit': FS.FINE_UNIT_M}


class FineStepTests(unittest.TestCase):
    def test_overlay_matches_builder_and_validates(self):
        overlay = json.loads(OVERLAY.read_text())
        self.assertEqual(overlay, FS.build_overlay())
        self.assertTrue(V.validate(overlay))
        names = [i['name'] for i in overlay['intents'] if i['action'] == 'fine_step']
        expect = [f'fine_{ax}_{sg}_{sz}'
                  for ax in 'xyz' for sg in ('plus', 'minus') for sz in ('5mm', '1cm', '3cm')]
        self.assertEqual(names, expect)
        self.assertTrue(all(i['fingers'] == 'keep' and i['gain'] == '1' for i in overlay['intents'] if i['action'] == 'fine_step'))
        tcp = np.zeros(3)
        got = set()
        for it in overlay['intents']:
            if it['action'] != 'fine_step':
                continue
            tgt = np.asarray(dsl.evaluate(it['refpoint'], _ns(tcp)), float)
            got.add(tuple(round(float(x), 5) for x in (tgt - tcp)))
        expected = set()
        for axis in range(3):
            for mag in (0.005, 0.01, 0.03):
                for sign in (1, -1):
                    delta = [0.0, 0.0, 0.0]
                    delta[axis] = sign * mag
                    expected.add(tuple(delta))
        self.assertEqual(got, expected)

    def test_bare_length_and_gripper_change_are_rejected(self):
        bad = FS.build_overlay()
        bad['intents'][0]['refpoint'] = 'vec(tcp[0] + 0.01, tcp[1], tcp[2])'
        with self.assertRaises(Exception):
            V.validate(bad)
        lib = {'version': 't', 'intents': [{'name': 'finish', 'action': 'finish', 'fingers': 'keep', 'definition': 'Stop.'}],
               'intent_instructions': 'Select the next intent.'}
        ov = FS.build_overlay()
        ov['intents'][0]['fingers'] = 'close'
        with self.assertRaises(ValueError):
            FS.merge_fine_steps(lib, ov)

    def test_flag_off_does_not_merge_or_read_overlay(self):
        self.assertFalse(FS.enabled_flag('0'))
        self.assertTrue(FS.enabled_flag('1'))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(FS.enabled_flag())
        base = json.loads(V2.read_text())
        snap = json.dumps(base)

        def boom(path=None):
            raise AssertionError('overlay was read')

        orig = FS.load_fine_library
        FS.load_fine_library = boom
        try:
            FS.apply_fine_steps(base, False)
        finally:
            FS.load_fine_library = orig
        self.assertEqual(json.dumps(base), snap)
        self.assertNotIn('fine_step', V2.read_text())
        self.assertNotIn('fine_unit', V2.read_text())
        self.assertNotIn('fine_step', FROZEN.read_text())

    def test_flag_on_merges_without_replacing_the_base_menu(self):
        base = json.loads(V2.read_text())
        before = [(i['name'], i.get('refpoint'), i.get('fingers'), i.get('action')) for i in base['intents']]
        version, note = base['version'], base.get('note')
        FS.apply_fine_steps(base, True, FS.build_overlay())
        self.assertEqual(base['version'], version)
        self.assertEqual(base.get('note'), note)
        for name, ref, fingers, action in before:
            got = next(i for i in base['intents'] if i['name'] == name)
            self.assertEqual(got.get('refpoint'), ref)
            self.assertEqual(got.get('fingers'), fingers)
            self.assertEqual(got.get('action', 'move'), action or 'move')
        names = [i['name'] for i in base['intents']]
        fin = names.index('finish')
        self.assertEqual(names[fin - 18:fin], [f'fine_{ax}_{sg}_{sz}'
                         for ax in 'xyz' for sg in ('plus', 'minus') for sz in ('5mm', '1cm', '3cm')])
        self.assertEqual(names.count('finish'), 1)
        self.assertEqual(base['intent_instructions'].count(FS.MERGE_TIP), 1)
        self.assertTrue(base['intent_instructions'].startswith(json.loads(V2.read_text())['intent_instructions']))
        n = len(base['intents'])
        FS.apply_fine_steps(base, True, FS.build_overlay())
        self.assertEqual(len(base['intents']), n)
        self.assertEqual(base['intent_instructions'].count(FS.MERGE_TIP), 1)
        self.assertTrue(V.validate(base))

    def test_existing_predicates_ignore_fine_unit(self):
        lib = json.loads(V2.read_text())
        ns = {
            'tcp': np.array([0.098, -0.096, 0.09]), 'grasp': np.array([0.082, -0.098, 0.053]),
            'objc': np.array([0.082, -0.098, 0.1]), 'place': np.zeros(3),
            'opening_at_close': 0.08, 'obj_h': 0.12, 'open_max': 0.08,
            'n_same': 2, 'n_close': 0, 'prev': 'descend_to_grasp', 'contact': False,
            'left_contact': False, 'right_contact': False, 'closed_cmd': False, 'last_result': 'stalled',
        }

        def preds(extra):
            env = dict(ns, **extra)
            out = {}
            for p in lib['predicates']:
                out[p['name']] = bool(dsl.evaluate(p['expr'], {**env, **out}))
            return out

        self.assertEqual(preds({}), preds({'fine_unit': FS.FINE_UNIT_M}))

    def test_clamp_and_text_report(self):
        self.assertEqual(FS.FINE_UNIT_M, 0.01)
        self.assertEqual(FS.FINE_STEP_MAX_M, 0.03)
        self.assertLess(FS.FINE_STEP_TOL_M, 0.005)
        self.assertLess(FS.FINE_STEP_TOL_M, 0.006)
        tcp = np.array([0.1, -0.02, 0.3])
        tgt, delta, status = FS.clamp_fine_target(tcp, tcp + np.array([0.10, -0.05, 0.04]))
        self.assertEqual(status, 'ok')
        self.assertTrue(np.allclose(delta, [0.03, -0.03, 0.03]))
        self.assertTrue(np.allclose(tgt, tcp + delta))
        same, delta5, status5 = FS.clamp_fine_target(tcp, tcp + np.array([0.005, 0.0, 0.0]))
        self.assertEqual(status5, 'ok')
        self.assertTrue(np.allclose(delta5, [0.005, 0.0, 0.0]))
        self.assertTrue(np.allclose(same, tcp + delta5))
        _, _, bad = FS.clamp_fine_target(tcp, np.array([np.nan, 0.0, 0.0]))
        self.assertEqual(bad, 'rejected')
        rep = FS.fine_step_report('fine_x_plus_5mm', 'reached', [0.105, 0.0, 0.2], [0.005, 0.0, 0.0], False)
        self.assertIn('fine_x_plus_5mm reached', rep['text'])
        self.assertIn('0.1050', rep['text'])
        self.assertIn('0.0050', rep['text'])
        self.assertIn('gripper open', rep['text'])
        self.assertEqual(FS.fine_step_fact(False, None), {})
        self.assertEqual(FS.fine_step_fact(True, None), {'last_fine_step': None})
        shown = FS.fine_step_fact(True, rep)
        self.assertEqual(shown['last_fine_step_text'], rep['text'])
        self.assertIn('end-effector now at', shown['last_fine_step_text'])

    def test_execute_path_servos_clamped_target_and_keeps_gripper(self):
        tcp0 = np.array([0.10, -0.02, 0.30])
        state = {'pos': tcp0.copy()}
        calls = []
        cleared = []

        def eef():
            return state['pos'].copy()

        def servo(target, grip, gain=1.0, max_steps=60, tol=0.006):
            calls.append({'target': np.asarray(target, float).copy(), 'grip': grip, 'gain': gain,
                          'max_steps': max_steps, 'tol': tol})
            state['pos'] = np.asarray(target, float).copy()
            return 'reached'

        ep = {'closed': True}
        shape = {'tilt': 12.0, 'tdir': 4.0, 'yaw': 25.0}
        scope = load_funcs('_remember_fine_step', '_execute_fine_step',
                           dsl=dsl, FS=FS, EP=ep, SHAPE=shape, eef=eef, servo=servo,
                           _clear_reapproach_if_aligned=lambda reason: cleared.append(reason))
        it = next(i for i in FS.build_overlay()['intents'] if i['name'] == 'fine_x_plus_5mm')
        result = scope['_execute_fine_step'](it, _ns(tcp0))
        self.assertEqual(result, 'reached')
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.allclose(calls[0]['target'], tcp0 + np.array([0.005, 0.0, 0.0])))
        self.assertEqual(calls[0]['grip'], 1)
        self.assertEqual(calls[0]['tol'], FS.FINE_STEP_TOL_M)
        self.assertEqual(calls[0]['max_steps'], FS.FINE_STEP_MAX_STEPS)
        self.assertTrue(ep['closed'])
        self.assertEqual(shape, {'tilt':12.0,'tdir':4.0,'yaw':25.0})
        self.assertEqual(cleared, ['fine_step'])
        rep = ep['last_fine_step']
        self.assertEqual(rep['result'], 'reached')
        self.assertEqual(rep['ee_xyz_m'], [0.105, -0.02, 0.3])
        self.assertIn('fine_x_plus_5mm reached', rep['text'])
        self.assertIn('0.1050', rep['text'])
        self.assertIn('gripper closed', rep['text'])
        self.assertTrue(rep['gripper_closed'])

        calls.clear()
        ep['closed'] = False
        state['pos'] = tcp0.copy()
        big = dict(it, refpoint='vec(tcp[0] + 10*fine_unit, tcp[1], tcp[2])')
        result = scope['_execute_fine_step'](big, _ns(tcp0))
        self.assertEqual(result, 'reached')
        self.assertTrue(np.allclose(calls[0]['target'], tcp0 + np.array([0.03, 0.0, 0.0])))
        self.assertEqual(calls[0]['grip'], -1)
        self.assertFalse(ep['closed'])
        self.assertIn('gripper open', ep['last_fine_step']['text'])

        calls.clear()
        shape['yaw'] = 25.0
        state['pos'] = tcp0.copy()
        tiny = dict(it, refpoint='vec(tcp[0] + 0.01*fine_unit, tcp[1], tcp[2])')
        result = scope['_execute_fine_step'](tiny, _ns(tcp0))
        self.assertEqual(result, 'at_target')
        self.assertEqual(calls, [])
        self.assertEqual(shape, {'tilt':12.0,'tdir':4.0,'yaw':25.0})
        self.assertEqual(ep['last_fine_step']['result'], 'at_target')
        self.assertEqual(ep['last_fine_step']['ee_xyz_m'], [0.1, -0.02, 0.3])

        def stall_servo(target, grip, gain=1.0, max_steps=60, tol=0.006):
            calls.append({'tol': tol})
            state['pos'] = state['pos'] + np.array([0.001, 0.0, 0.0])
            return 'stalled'

        scope['servo'] = stall_servo
        calls.clear()
        state['pos'] = tcp0.copy()
        result = scope['_execute_fine_step'](it, _ns(tcp0))
        self.assertEqual(result, 'stalled')
        self.assertEqual(ep['last_fine_step']['ee_xyz_m'], [0.101, -0.02, 0.3])
        self.assertIn('stalled', ep['last_fine_step']['text'])
        self.assertIn('0.1010', ep['last_fine_step']['text'])
        self.assertEqual(ep['last_fine_step']['commanded_delta_m'], [0.005, 0.0, 0.0])

        class NanDsl:
            def evaluate(self, expr, ns):
                return np.array([np.nan, np.nan, np.nan])

        class Boom:
            def evaluate(self, expr, ns):
                raise ValueError('bad refpoint')

        calls.clear()
        scope['dsl'] = NanDsl()
        self.assertEqual(scope['_execute_fine_step'](it, _ns(tcp0)), 'rejected')
        self.assertEqual(calls, [])
        self.assertEqual(ep['last_fine_step']['result'], 'rejected')
        scope['dsl'] = Boom()
        self.assertEqual(scope['_execute_fine_step'](it, _ns(tcp0)), 'refpoint_error')
        self.assertEqual(calls, [])

    def test_feedback_reaches_current_image_payload(self):
        import tempfile
        import jev_observation as JO
        from test_transport_feedback import fixture
        scope,ep,_,_=fixture()
        ep['last_fine_step']=FS.fine_step_report('fine_x_plus_5mm','reached',[.005,0,0],[.005,0,0],False)
        scope.update(FINE_STEPS=True,FS=FS)
        scope['_current_observation']=lambda:({key:0 for key in JO.FIELDS},
                                              {'front':np.zeros((2,2,3),dtype='uint8')})
        with tempfile.TemporaryDirectory() as temp:
            scope['ep_dir']=temp
            state,assets=scope['_jev_input']()
        self.assertEqual(state['motion_context']['last_fine_step'],ep['last_fine_step'])
        self.assertNotIn('observed_facts',state)
        self.assertTrue(assets)


if __name__ == '__main__':
    unittest.main()
