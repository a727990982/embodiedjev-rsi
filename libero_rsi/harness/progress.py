"""Task todo persistence, checkpoint refresh and Jev todo review."""

import json
import os

import jev_todo as JT
import task_todo as TODO

from . import observation, planning, vision
from .state import EP, GOAL_TARGET, PER, RSHAPE, T, log, rt


def _save_task_todo(reason):
    with open(os.path.join(rt.ep_dir, 'task_todo.json'), 'w') as output:
        json.dump(EP['task_todo'], output, indent=2)
    log(t=T[0], task_todo=EP['task_todo'], todo_reason=reason)


def _refresh_task_progress(intent=None, result=None):
    """Main-thread sensor refresh; checkpoints never retarget or call a model."""
    target_rev = GOAL_TARGET['rev']
    updated = TODO.refresh_progress(
        EP['task_todo'], EP['todo_target_id'], T[0],
        identity_confirmed=vision.goal_id_confirmed(), grasp_ready=observation._transport_grasp_ready(),
        gripper_closed=EP['closed'], intent=intent, result=result,
        waypoints_changed=EP.get('todo_progress_target_rev', target_rev) != target_rev,
        preserve_placement_review=any(i['name'] == intent and i.get('action') in ('fine_step', 'pose_menu')
                                      for i in rt.LIB['intents']))
    EP['todo_progress_target_rev'] = target_rev
    if updated != EP['task_todo']:
        EP['task_todo'] = updated
        _save_task_todo('stage_progress')


def _retarget_todo():
    """Invalidate old grasp state; localize and verify the active todo object."""
    item = TODO.active_item(EP['task_todo'])
    new_id = item['id'] if item else None
    if EP.get('todo_target_id') == new_id:
        return
    planning._abandon_async_slow('todo_target_changed')
    old_id = EP.get('todo_target_id')
    EP['todo_target_id'] = new_id
    EP.pop('robot_geometry_advice', None)
    EP['recovery_history_start'] = len(EP.get('hist', []))
    EP['last_recovery_preparation'] = None
    EP['last_failed_motion'] = None
    EP['failed_motion_before_path'] = None
    EP['motion_before_path'] = None
    EP.update(force_next_intent=None, grasp_confirm=False, grasp_secured_once=False,
              ever_contact=False, place_carry_reached_once=False, place_lower_stall_n=0,
              place_lift_lower_cycles=0, place_cycle_armed=False, place_force_release=False,
              place_force_release_reason=None, uses={})
    vision._reset_patch('todo_target_changed')
    RSHAPE.update(tilt=None, tdir=None, yaw=None, dz=0.0)
    EP.update(tcp_at_close=None, opening_at_close=None, n_same=0,
              last_stall_issue=None, local_fb_family_hist=[], push_aside_approach_n=0)
    PER['seen_held'] = False
    vision.set_goal_id(False, 'todo_target_changed', {'old': old_id, 'new': new_id})
    GOAL_TARGET['rev'] += 1
    if item is not None:
        p, rgb, dep = vision.perceive(phase='todo_target')
        vision._commit_keypoints(p, dep, 'todo_target', held_now=False)
        verification, _ = vision.verify_grasp_identity(p, rgb, phase='todo_target')
        matched = vision.identity_ok(p) and verification.get('is_language_goal_object') is True
        vision.set_goal_id(matched, 'todo_target_verified', verification)
        if not matched:
            EP['force_next_intent'] = 'relook'
    log(t=T[0], todo_target_switch=True, old_id=old_id, new_id=new_id)


def _review_task_todo(reason, *, retarget=True):
    """Jev classifies current outcomes synchronously; only the host commits them."""
    state, assets = observation._jev_input()
    updated, rec = JT.review(state, todo=EP['task_todo'], assets=assets, reason=reason)
    EP['todo_calls'] += 1
    # Repeated checks with unchanged statuses must not starve asynchronous help.
    if [i['status'] for i in updated['items']] != [i['status'] for i in EP['task_todo']['items']]:
        planning._abandon_async_slow('todo_progress_changed')
    EP['task_todo'] = updated
    log(t=T[0], jev_todo_review=rec, observation_serial=EP['vision_serial'])
    _save_task_todo(reason)
    if retarget:
        _retarget_todo()
