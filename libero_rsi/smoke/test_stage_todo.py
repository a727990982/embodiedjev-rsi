"""Evidence-backed stage progress, including the init6 premature-release regression."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task_todo as TODO
from test_review_regressions import functions


def todo():
    return TODO.initialize({'items': [
        {'id': target, 'target': target, 'goal': f'put {target} in basket',
         'completion_condition': f'{target} inside basket, released and gripper clear',
         'progress': {'stages': [{'status': 'done'}]}}
        for target in ('cheese', 'butter')]}, 'put cheese and butter in basket', 10)


def refresh(t, step, intent=None, result=None, *, held=True, identity=True):
    return TODO.refresh_progress(t, 'cheese', step, identity_confirmed=identity,
                                 grasp_ready=held, gripper_closed=held,
                                 intent=intent, result=result)


def progress(t):
    return t['items'][0]['progress']


def statuses(t):
    return {s['id']: s['status'] for s in progress(t)['stages']}


def review(t, status, step=210, needs_regrasp=True):
    return TODO.apply_review(t, {'items': [
        {'id': 'cheese', 'status': status, 'evidence': 'current-image review',
         'needs_regrasp': needs_regrasp},
        {'id': 'butter', 'status': 'pending', 'evidence': 'butter on table'}]}, step)


class StageTests(unittest.TestCase):
    def test_init6_grasp_lift_then_premature_reopen(self):
        t = todo()
        self.assertTrue(all(s == 'pending' for s in statuses(t).values()))
        t['items'][0].update(evidence='On tabletop between open fingers', checked_at_step=123)
        t = refresh(t, 166, 'close_gripper', 'closed')
        t = refresh(t, 179, 'lift', 'reached')
        self.assertEqual(progress(t)['next_stage'], 'carry_to_destination')
        self.assertTrue(progress(t)['current_grasp_confirmed'])
        self.assertEqual([s['status'] for s in progress(t)['stages']],
                         ['done', 'done', 'done', 'pending', 'pending', 'pending'])
        self.assertEqual(t['active_id'], 'cheese')
        self.assertEqual(t['items'][0]['status'], 'pending')
        state = TODO.decision_view(t)
        self.assertNotIn('evidence', state['items'][0])
        self.assertEqual(state['items'][0]['last_visual_review']['checked_at_step'], 123)
        self.assertEqual(state['items'][0]['progress']['checked_at_step'], 179)
        dropped = refresh(t, 194, 'reopen_and_retry', 'reopened', held=False)
        self.assertEqual(progress(dropped)['next_stage'], 'secure_grasp')
        self.assertFalse(progress(dropped)['current_grasp_confirmed'])
        self.assertTrue(all(s['status'] == 'pending' for s in progress(dropped)['stages'][1:]))
        self.assertEqual(progress(t)['next_stage'], 'carry_to_destination')

    def test_grasp_loss_or_wrong_identity_revokes_transport_progress(self):
        t = refresh(refresh(todo(), 179, 'lift', 'reached'), 190, 'carry_to_place', 'reached')
        for held, identity, expected in [(False, True, 'secure_grasp'), (True, False, 'identify_target')]:
            with self.subTest(held=held, identity=identity):
                lost = refresh(t, 191, held=held, identity=identity)
                self.assertEqual(progress(lost)['next_stage'], expected)
                self.assertFalse(progress(lost)['current_grasp_confirmed'])
                self.assertEqual(statuses(lost)['carry_to_destination'], 'pending')

    def test_command_or_failed_motion_cannot_check_a_stage(self):
        for result in ('stalled', 'timeout', 'lift_not_executed:grasp_unconfirmed', None):
            t = refresh(todo(), 179, 'lift', result)
            self.assertEqual(statuses(t)['lift_clear'], 'pending')
        t = refresh(todo(), 179, 'lift', 'reached', held=False)
        self.assertEqual(statuses(t)['lift_clear'], 'pending')
        t = refresh(todo(), 179, 'lower_to_place', 'reached')
        self.assertEqual(statuses(t)['lower_for_release'], 'pending')

    def test_at_target_is_valid_feedback_but_does_not_mean_task_done(self):
        t = refresh(todo(), 179, 'lift', 'at_target')
        self.assertEqual(progress(t)['next_stage'], 'carry_to_destination')
        self.assertEqual(t['items'][0]['status'], 'pending')
        same = refresh(t, 179, 'lift', 'at_target')
        self.assertEqual(same, t)

    def test_expected_release_waits_for_review_and_visible_failure_resets(self):
        t = refresh(todo(), 179, 'lift', 'reached')
        t = refresh(t, 190, 'carry_to_place', 'reached')
        t = refresh(t, 200, 'lower_to_place', 'reached')
        self.assertEqual(progress(t)['next_stage'], 'place_and_clear')
        t = refresh(t, 205, 'open_gripper', 'at_target', held=False)
        self.assertTrue(progress(t)['awaiting_placement_review'])
        self.assertEqual(t['items'][0]['status'], 'pending')
        self.assertEqual(statuses(t)['place_and_clear'], 'pending')
        hidden = refresh(review(t, 'unobserved'), 211, 'retreat', 'reached', held=False)
        self.assertTrue(progress(hidden)['awaiting_placement_review'])
        self.assertEqual(progress(hidden)['next_stage'], 'place_and_clear')
        failed = refresh(review(t, 'pending'), 211, held=False)
        self.assertEqual(progress(failed)['next_stage'], 'secure_grasp')
        self.assertFalse(progress(failed)['awaiting_placement_review'])
        done = review(t, 'done')
        self.assertEqual(done['active_id'], 'butter')
        self.assertEqual(statuses(done)['place_and_clear'], 'done')
        self.assertEqual(done['items'][1]['progress'], todo()['items'][1]['progress'])

    def test_placement_waiting_for_clearance_does_not_request_regrasp(self):
        t = refresh(refresh(todo(), 179, 'lift', 'reached'), 190, 'carry_to_place', 'reached')
        t = refresh(t, 205, 'open_gripper', 'at_target', held=False)
        for needs_regrasp in (False, None):
            checked = review(t, 'pending', needs_regrasp=needs_regrasp)
            checked = refresh(checked, 211, held=False)
            self.assertTrue(progress(checked)['awaiting_placement_review'])
            self.assertEqual(progress(checked)['next_stage'], 'place_and_clear')
        for invalid in ('false', 0, []):
            with self.assertRaises(ValueError):
                review(t, 'pending', needs_regrasp=invalid)
        with self.assertRaises(ValueError):
            TODO.apply_review(t, {'items': [
                {'id': name, 'status': 'pending', 'evidence': 'on table'}
                for name in ('cheese', 'butter')]}, 211)

    def test_early_release_does_not_claim_expected_placement(self):
        t = refresh(todo(), 179, 'lift', 'reached')
        t = refresh(t, 194, 'open_gripper', 'at_target', held=False)
        self.assertFalse(progress(t)['awaiting_placement_review'])
        self.assertEqual(progress(t)['next_stage'], 'secure_grasp')

    def test_final_success_does_not_fabricate_unobserved_intermediate_motions(self):
        t = TODO.official_completion(todo(), 99)
        self.assertEqual(statuses(t)['lift_clear'], 'not_required')
        self.assertEqual(statuses(t)['place_and_clear'], 'done')
        self.assertIsNone(progress(t)['next_stage'])
        self.assertIsNone(t['active_id'])
        reopened = review(t, 'uncertain')
        self.assertTrue(all(s == 'pending' for s in statuses(reopened).values()))

    def test_sensor_progress_stays_with_its_object(self):
        t = todo()
        unchanged = TODO.refresh_progress(t, 'butter', 166, identity_confirmed=True,
                                          grasp_ready=True, gripper_closed=True)
        self.assertEqual(unchanged, t)

    def test_new_localization_invalidates_old_motion_waypoints(self):
        t = refresh(refresh(todo(), 179, 'lift', 'reached'), 190, 'carry_to_place', 'reached')
        changed = TODO.refresh_progress(t, 'cheese', 191, identity_confirmed=True,
            grasp_ready=True, gripper_closed=True, waypoints_changed=True)
        self.assertEqual(progress(changed)['next_stage'], 'lift_clear')
        self.assertTrue(progress(changed)['current_grasp_confirmed'])
        self.assertEqual(statuses(changed)['carry_to_destination'], 'pending')

    def test_runtime_refresh_uses_live_support_without_clearing_grasp_or_retargeting(self):
        ep = {'task_todo': todo(), 'todo_target_id': 'cheese', 'closed': True,
              'grasp_confirm': True, 'grasp_secured_once': True}
        sensors = {'contacts': (True, True)}
        saved = []
        scope = functions('_refresh_task_progress', '_transport_grasp_ready',
            EP=ep, TODO=TODO, T=[166], GOAL_TARGET={'rev': 1}, OPEN_MAX=.08,
            goal_id_confirmed=lambda: True,
            contacts=lambda: sensors['contacts'], opening=lambda: .0555,
            _save_task_todo=lambda reason: saved.append(deepcopy(ep['task_todo'])))
        scope['_refresh_task_progress']('close_gripper', 'closed')
        scope['T'][0] = 179
        scope['_refresh_task_progress']('lift', 'reached')
        self.assertEqual(progress(ep['task_todo'])['next_stage'], 'carry_to_destination')
        self.assertTrue(ep['grasp_confirm'])
        self.assertEqual(ep['todo_target_id'], 'cheese')
        sensors['contacts'] = (False, False)
        scope['T'][0] = 180
        scope['_refresh_task_progress']()
        self.assertEqual(progress(ep['task_todo'])['next_stage'], 'secure_grasp')
        self.assertEqual(saved[-1], ep['task_todo'])


if __name__ == '__main__':
    unittest.main()
