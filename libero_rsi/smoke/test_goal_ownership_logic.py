"""Logic smoke: single grasp-target ownership (no MuJoCo / Astra).
Priority: latest relook keypoints > slow-think patch on current target > episode-start.
Relook overwrites PER and clears PATCH; slow-think mutates the same PATCH; relook requests execute perception.
"""
import ast
from pathlib import Path
import numpy as np
from test_review_regressions import functions

HARNESS = Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py'


def empty_patch():
    return {'grasp_dz_objh': 0.0, 'tilt_deg': None, 'tilt_dir_deg': None, 'yaw_deg': None, 'grasp_dxy_objw': [0.0, 0.0]}


def effective_grasp(per, patch):
    dxy = patch.get('grasp_dxy_objw') or [0.0, 0.0]
    return per['grasp'] + np.array([float(dxy[0]) * per['obj_w'], float(dxy[1]) * per['obj_w'],
                                    float(patch.get('grasp_dz_objh') or 0.0) * per['obj_h']])


def commit_keypoints(per, patch, goal, new_grasp, source):
    """Mirror of harness _commit_keypoints ownership rules."""
    per['grasp'] = np.asarray(new_grasp, float).copy()
    src = str(source)
    is_start = src in ('episode_start', 'perceive_start')
    if not is_start:
        patch.clear(); patch.update(empty_patch())
        goal['source'] = 'relook'
    else:
        goal['source'] = 'episode_start'
    goal['base_from'] = src
    goal['rev'] = int(goal.get('rev', 0)) + 1
    return goal


def note_patch_mutation(patch, goal, updates, source):
    patch.update(updates)
    goal['source'] = 'slow_patch'
    goal['last_patch_from'] = source
    goal['rev'] = int(goal.get('rev', 0)) + 1
    return goal


def plan_target_update(id_ok, stallish, has_kp, plan_steps):
    calls = []
    scope = functions('_apply_episode_plan', 'identity_fields', np=np,
        EP={'closed': False, 'last_result': 'stalled' if stallish else 'reached'},
        LIB={'intents': []}, PATCH={}, NEIGHBOUR_STALL_ISSUES=set(),
        goal_id_confirmed=lambda: id_ok, _should_open_on_intervention=lambda **kw: False,
        do_relook=lambda why: calls.append('perceive') or 'relooked_matched',
        namespace=lambda: ({}, {}),
        _apply_stall_feedback=lambda *a, **kw: calls.append('feedback') or {'action': kw.get('action', 'raise_redescend')})
    plan = {'issue': 'blocked_by_neighbour' if stallish else 'ok', 'matched_language_goal': True,
            'plan': [{'step': k} for k in plan_steps]}
    if has_kp: plan['grasp_px'] = [10, 10]
    return scope['_apply_episode_plan'](plan, None), bool(calls)


def main():
    src = HARNESS.read_text()
    # harness wiring present
    for needle in ('_commit_keypoints', '_reset_patch', '_note_patch_mutation', 'GOAL_TARGET',
                   'forced_target_update', "do_relook(f'{source}:plan')",
                   "ASTRA_EFFORT = os.environ.get('ASTRA_EFFORT', 'xhigh')"):
        assert needle in src, needle
    # syntax still valid
    ast.parse(src)

    per = {'grasp': np.array([0.1, 0.0, 0.05]), 'obj_h': 0.1, 'obj_w': 0.05}
    patch = empty_patch()
    goal = {'source': 'episode_start', 'base_from': None, 'last_patch_from': None, 'rev': 0}
    g0 = effective_grasp(per, patch).copy()

    # 1) episode-start seeds base, does not invent a second goal
    commit_keypoints(per, patch, goal, [0.1, 0.0, 0.05], 'episode_start')
    assert goal['source'] == 'episode_start'
    assert patch['grasp_dz_objh'] == 0.0

    # 2) slow-think mutates SAME patch on current base
    note_patch_mutation(patch, goal, {'grasp_dz_objh': 0.2, 'yaw_deg': 45.0}, 'slow:adjust')
    assert goal['source'] == 'slow_patch'
    g1 = effective_grasp(per, patch)
    assert abs(g1[2] - (0.05 + 0.2 * 0.1)) < 1e-9
    assert patch['yaw_deg'] == 45.0

    # 3) relook overwrites base AND clears stale patch (priority: relook > slow_patch)
    old_yaw = patch['yaw_deg']
    commit_keypoints(per, patch, goal, [0.12, 0.01, 0.06], 'relook')
    assert goal['source'] == 'relook'
    assert patch == empty_patch(), patch
    assert patch.get('yaw_deg') is None and old_yaw == 45.0
    g2 = effective_grasp(per, patch)
    assert np.allclose(g2, [0.12, 0.01, 0.06])

    # 4) perception refresh after a slow request clears previous offsets
    note_patch_mutation(patch, goal, {'grasp_dz_objh': 0.3, 'grasp_dxy_objw': [0.5, 0.0]}, 'stall')
    commit_keypoints(per, patch, goal, [0.11, 0.0, 0.055], 'relook')
    assert goal['source'] == 'relook'
    assert patch['grasp_dz_objh'] == 0.0
    assert patch['grasp_dxy_objw'] == [0.0, 0.0]

    # 5) A requested relook on a confirmed stall runs real perception.
    done, upd = plan_target_update(id_ok=True, stallish=True, has_kp=False, plan_steps=['relook'])
    assert done == ['relook:relooked_matched'] and upd

    # 6) Planner-supplied pixels request perception instead of overwriting its base.
    done2, upd2 = plan_target_update(id_ok=True, stallish=True, has_kp=True, plan_steps=['relook'])
    assert done2 == ['relook:relooked_matched'] and upd2

    # 7) Relative motion feedback remains executable.
    done3, upd3 = plan_target_update(id_ok=True, stallish=True, has_kp=False, plan_steps=['raise_redescend'])
    assert upd3 and done3 == ['raise_redescend:raise_redescend']

    # 8) one shared effective goal formula (PER + PATCH), not two stores
    note_patch_mutation(patch, goal, {'grasp_dz_objh': 0.1}, 'slow')
    g = effective_grasp(per, patch)
    assert np.allclose(g, per['grasp'] + np.array([0, 0, 0.1 * per['obj_h']]))

    print('OK goal_ownership_logic')


if __name__ == '__main__':
    main()
