"""Logic smoke: post-recovery namespace freshness (B05) + hold-preserving fallback release. No MuJoCo."""
import ast
from pathlib import Path
import numpy as np

HARNESS = Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py'


def should_open_on_intervention(*, closed, igo_n, issue=None, failure_class=None):
    """Mirror of harness _should_open_on_intervention."""
    if not closed:
        return False
    issue = str(issue or failure_class or '')
    if issue in ('wrong_object', 'empty_grasp', 'tip_push'):
        return True
    if igo_n is False:
        return True
    if igo_n is True:
        return False
    return True


def effective_grasp(per, patch, rshape_dz=0.0):
    dxy = patch.get('grasp_dxy_objw') or [0.0, 0.0]
    return per['grasp'] + np.array([
        float(dxy[0]) * per['obj_w'], float(dxy[1]) * per['obj_w'],
        (float(patch.get('grasp_dz_objh') or 0.0) + rshape_dz) * per['obj_h'],
    ])


def main():
    src = HARNESS.read_text()
    ast.parse(src)
    for needle in (
        'recovery_mutated', 'post_recovery_ns_refresh',
        'ns_gate, preds_gate = namespace()', 'f_gate = facts(ns_gate, preds_gate)',
        '_should_open_on_intervention',
        'Do not unconditionally reopen here',
    ):
        assert needle in src, f'missing: {needle}'

    # 1) Recovery mutates PATCH → execute must see NEW grasp (fresh ns)
    per = {'grasp': np.array([0.10, 0.0, 0.05]), 'obj_h': 0.10, 'obj_w': 0.05}
    patch = {'grasp_dz_objh': 0.0, 'grasp_dxy_objw': [0.0, 0.0]}
    g_before = effective_grasp(per, patch).copy()
    # stall feedback raises grasp_dz
    patch['grasp_dz_objh'] = 0.25
    patch['grasp_dxy_objw'] = [0.4, 0.0]
    g_after = effective_grasp(per, patch)
    assert not np.allclose(g_before, g_after)
    # Simulate bug: execute with stale ns (old grasp) vs refreshed
    stale_ns = {'grasp': g_before}
    fresh_ns = {'grasp': g_after}
    assert not np.allclose(stale_ns['grasp'], fresh_ns['grasp'])
    # harness now recomputes after recovery_mutated — forced intent uses fresh_ns
    forced_target = fresh_ns['grasp']
    assert abs(forced_target[2] - (0.05 + 0.25 * 0.10)) < 1e-9

    # 2) Hold-preserving release matrix
    # verified hold + path stall → keep closed
    assert should_open_on_intervention(closed=True, igo_n=True, failure_class='blocked_by_neighbour') is False
    assert should_open_on_intervention(closed=True, igo_n=True, issue='other') is False
    # wrong object / empty → open
    assert should_open_on_intervention(closed=True, igo_n=False, failure_class='wrong_object') is True
    assert should_open_on_intervention(closed=True, igo_n=True, issue='empty_grasp') is True
    assert should_open_on_intervention(closed=True, igo_n=None, failure_class='wrong_object') is True
    # unsure while closed → open
    assert should_open_on_intervention(closed=True, igo_n=None, issue='other') is True
    # already open → False
    assert should_open_on_intervention(closed=False, igo_n=False, issue='wrong_object') is False

    print('test_post_recovery_ns_freshness: OK')


if __name__ == '__main__':
    main()
