"""Logic smoke: single grasp-target ownership (no MuJoCo / Astra).
Priority: latest relook keypoints > slow-think patch on current target > episode-start.
Relook overwrites PER and clears PATCH; slow-think mutates the same PATCH; empty relook-only on confirmed stall forbidden.
"""
import ast
from pathlib import Path
import numpy as np

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
    """Mirror empty-relook forbid + forced target update decision."""
    done, target_updated = [], False
    if has_kp:
        target_updated = True
    for k in plan_steps:
        if k == 'relook':
            if id_ok and stallish and not has_kp:
                done.append('relook_skipped_empty_confirmed_stall')
            else:
                done.append('relook')
                if has_kp:
                    target_updated = True
        elif k in ('adjust', 'raise_redescend', 'lateral_recenter', 'change_approach_yaw',
                   'confirm_contact', 'clear_reapproach', 'push_aside'):
            done.append(k); target_updated = True
    if id_ok and stallish and not target_updated:
        done.append('forced_stall_feedback:raise_redescend')
        target_updated = True
    return done, target_updated


def main():
    src = HARNESS.read_text()
    # harness wiring present
    for needle in ('_commit_keypoints', '_reset_patch', '_note_patch_mutation', 'GOAL_TARGET',
                   'forced_target_update', 'relook_skipped_empty_confirmed_stall',
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

    # 4) slow keypoints (= relook priority) also clear previous offsets
    note_patch_mutation(patch, goal, {'grasp_dz_objh': 0.3, 'grasp_dxy_objw': [0.5, 0.0]}, 'stall')
    commit_keypoints(per, patch, goal, [0.11, 0.0, 0.055], 'slow_keypoints')
    assert goal['source'] == 'relook'
    assert patch['grasp_dz_objh'] == 0.0
    assert patch['grasp_dxy_objw'] == [0.0, 0.0]

    # 5) empty relook-only on confirmed stall → forced target update
    done, upd = plan_target_update(id_ok=True, stallish=True, has_kp=False, plan_steps=['relook'])
    assert 'relook_skipped_empty_confirmed_stall' in done
    assert any(s.startswith('forced_stall_feedback') for s in done)
    assert upd is True

    # 6) relook-refresh with keypoints on stall counts as target update (allowed)
    done2, upd2 = plan_target_update(id_ok=True, stallish=True, has_kp=True, plan_steps=['relook'])
    assert 'relook_skipped_empty_confirmed_stall' not in done2
    assert not any(s.startswith('forced_stall_feedback') for s in done2)
    assert upd2 is True

    # 7) raise/shift without relook also updates target
    done3, upd3 = plan_target_update(id_ok=True, stallish=True, has_kp=False, plan_steps=['raise_redescend'])
    assert upd3 is True and 'raise_redescend' in done3

    # 8) one shared effective goal formula (PER + PATCH), not two stores
    note_patch_mutation(patch, goal, {'grasp_dz_objh': 0.1}, 'slow')
    g = effective_grasp(per, patch)
    assert np.allclose(g, per['grasp'] + np.array([0, 0, 0.1 * per['obj_h']]))

    print('OK goal_ownership_logic')


if __name__ == '__main__':
    main()
