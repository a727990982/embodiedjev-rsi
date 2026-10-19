"""Motor transport/execution contracts. Deterministic doubles, not physical success."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cartesian_policy as CP
import motion_feedback as MF
from test_review_regressions import functions


class CartesianTests(unittest.TestCase):
    def test_axis_questions_do_not_require_a_single_axis_or_ground_truth(self):
        context, questions = CP.motor_request('carry', 'carry', [0, 0, .2], [.1, -.1, .2], 'close')
        self.assertEqual(context['target_relative_to_gripper'], {'x': 'positive', 'y': 'negative', 'z': 'aligned'})
        self.assertEqual(set(questions), {'x', 'y', 'z', 'fingers'})
        self.assertEqual(context['requested_fingers'], 'close')
        self.assertNotIn('cube_xyz', context)
        motor = CP.motor_choices({a: {'choice': c} for a, c in
                                  dict(x='positive', y='negative', z='stay', fingers='close').items()})
        np.testing.assert_allclose(CP.signed_delta([.1, -.1, .1], motor, .14, .05), [.007, -.007, 0])

    def test_wrong_direction_and_endpoint_clamp(self):
        motor = dict(x='positive', y='negative', z='positive', fingers='stay')
        np.testing.assert_allclose(CP.signed_delta([-.1, -.001, .1], motor, .14, .05), [0, -.001, .007])
        np.testing.assert_allclose(CP.signed_delta([0, 0, 0], motor, .14, .05), [0, 0, 0])
        with self.assertRaises(ValueError):
            CP.motor_choices({'x': {'choice': 'positive'}})
        with self.assertRaises(ValueError):
            CP.motor_request('carry', '', [0, 0, 0], [0, float('nan'), 0], 'close')

    def servo_scope(self, *, locked=False, done=False, horizon=100):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        pos = np.array([0., 0., .2]); t = [0]; actions = []
        contact = lambda: {'environment_contact_parts': [], 'self_contact_parts': []}
        def step(action):
            actions.append(action.copy()); t[0] += 1
            if not locked: pos[:] += np.asarray(action[:3]) * .05
        scope = functions('servo', np=np, CP=CP, MF=MF, Rot=Rot, Image=Image,
            EP={}, T=t, DONE=[done], HORIZON=horizon, ep_dir=temp.name,
            env=SimpleNamespace(env=SimpleNamespace(control_timestep=.05)),
            obs={'robot0_eef_quat': [0, 0, 0, 1]}, eef=lambda: pos.copy(), step=step,
            commanded_rotation=lambda: Rot.identity(),
            orientation_feedback=lambda *a: {'within_tolerance': True}, robot_contacts=contact,
            _combo_rgb=lambda: (np.zeros((4, 4, 3), dtype=np.uint8), None, None), log=lambda **kw: None)
        return scope, pos, actions

    def test_partial_block_does_not_claim_waypoint_arrival(self):
        scope, pos, actions = self.servo_scope()
        motor = dict(x='positive', y='stay', z='stay', fingers='close')
        result = scope['servo'](np.array([.2, 0, .2]), 1, max_steps=8, motor=motor)
        self.assertEqual(result, 'in_progress')
        self.assertEqual(len(actions), 8)
        self.assertAlmostEqual(pos[0], .056)
        self.assertFalse(scope['EP']['last_motion']['position_reached'])
        self.assertTrue(all(a[6] == 1 and a[1] == 0 and a[2] == 0 for a in actions))

    def test_reached_stall_and_terminal_are_distinct(self):
        motor = dict(x='positive', y='stay', z='stay', fingers='open')
        scope, pos, actions = self.servo_scope()
        self.assertEqual(scope['servo'](np.array([.01, 0, .2]), -1, max_steps=8, motor=motor), 'reached')
        self.assertLessEqual(pos[0], .01)
        scope, _, _ = self.servo_scope(locked=True)
        self.assertEqual(scope['servo'](np.array([.2, 0, .2]), -1, max_steps=8, motor=motor), 'stalled')
        for options in ({'done': True}, {'horizon': 0}):
            scope, _, actions = self.servo_scope(**options)
            self.assertEqual(scope['servo'](np.array([.2, 0, .2]), -1, motor=motor), 'horizon_or_done')
            self.assertEqual(actions, [])

    def policy_scope(self, *, fingers='close', blocked=None):
        ns = {'tcp': np.array([0, 0, .2])}
        def choose(state, questions, *, assets):
            self.assertEqual(assets, {'frame': 'current'})
            self.assertEqual(state['observation_step'], 17)
            self.assertEqual(state['motor_context']['reference_target_xyz_m'], [.1, 0, .2])
            return {k: {'choice': v} for k, v in
                    dict(x='positive', y='stay', z='stay', fingers=fingers).items()}, .1
        transport = Mock(side_effect=choose)
        execute = Mock(return_value='in_progress')
        scope = functions('execute_policy', np=np, CP=CP, JC=SimpleNamespace(choose_questions=transport),
            dsl=SimpleNamespace(evaluate=lambda *a: [.1, 0, .2]),
            LIB={'intents': [{'name': 'carry_to_place', 'action': 'move', 'fingers': 'close',
                              'refpoint': 'place', 'definition': 'Carry above destination'}]},
            EP={'vision_serial': 2}, T=[17], DONE=[False],
            MOTION_RECOVERIES=set(), TRANSPORT_INTENTS={'carry_to_place'},
            namespace=lambda: (ns, {}), _action_block_reason=lambda *a: blocked,
            _transport_grasp_ready=lambda: True, eef=lambda: ns['tcp'], execute=execute,
            _jev_input=lambda **kw: ({'observation_step': 17}, {'frame': 'current'}), log=lambda **kw: None)
        return scope, transport, execute

    def test_motor_call_reaches_executor_and_is_accounted(self):
        scope, transport, execute = self.policy_scope()
        self.assertEqual(scope['execute_policy']('carry_to_place', {}), 'in_progress')
        transport.assert_called_once()
        self.assertEqual(scope['EP']['motor_calls'], 1)
        self.assertEqual(execute.call_args.kwargs['motor']['x'], 'positive')
        np.testing.assert_allclose(execute.call_args.kwargs['reference'], [.1, 0, .2])

    def test_motor_cannot_open_a_transport_grasp(self):
        scope, transport, execute = self.policy_scope(fingers='open')
        self.assertEqual(scope['execute_policy']('carry_to_place', {}), 'motor_not_executed:gripper_intent_conflict')
        execute.assert_not_called()

    def test_stay_preserves_an_already_closed_transport_grasp(self):
        scope, _, execute = self.policy_scope(fingers='stay')
        scope['EP']['closed'] = True
        self.assertEqual(scope['execute_policy']('carry_to_place', {}), 'in_progress')
        execute.assert_called_once()

    def test_reopen_does_not_advertise_the_old_library_approach_motion(self):
        scope, transport, execute = self.policy_scope()
        scope['LIB']['intents'][0].update(name='reopen_and_retry', fingers='open')
        def choose(state, questions, *, assets):
            self.assertEqual(state['motor_context']['reference_target_xyz_m'], [0, 0, .2])
            self.assertEqual(set(state['motor_context']['target_relative_to_gripper'].values()), {'aligned'})
            return {a: {'choice': 'open' if a == 'fingers' else 'stay'} for a in questions}, .1
        transport.side_effect = choose
        scope['execute_policy']('reopen_and_retry', {})
        np.testing.assert_allclose(execute.call_args.kwargs['reference'], [0, 0, .2])

    def test_existing_refusal_and_network_failure_do_not_fallback_to_motor_motion(self):
        scope, transport, execute = self.policy_scope(blocked='motion_not_executed:unchanged_failed_waypoint')
        scope['execute_policy']('carry_to_place', {})
        transport.assert_not_called(); execute.assert_called_once_with('carry_to_place', scope['namespace']()[0])
        scope, transport, execute = self.policy_scope()
        transport.side_effect = RuntimeError('offline service failure')
        with self.assertRaisesRegex(RuntimeError, 'offline service failure'):
            scope['execute_policy']('carry_to_place', {})
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
