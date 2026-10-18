"""Logic smoke for identity×relook escape + relative yaw (no MuJoCo / Astra)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate as V
import dsl_libero as dsl

def relative_yaw_delta_deg(ns, approach_changes):
    l = bool(ns.get('left_contact')); r = bool(ns.get('right_contact'))
    if l and not r: sign = 1.0
    elif r and not l: sign = -1.0
    else: sign = 1.0 if (approach_changes % 2 == 0) else -1.0
    return sign * 45.0

def apply_yaw(cur, delta):
    new_yaw = float(np.clip(cur + delta, -90.0, 90.0))
    if abs(new_yaw - cur) < 1.0:
        new_yaw = float(np.clip(cur - delta, -90.0, 90.0))
    return new_yaw

def decide_close_path(goal_ok, relooks, max_relook, close_refusals_after):
    if goal_ok: return 'close'
    if relooks >= max_relook: return 'escape'
    rr = 'relooked_unmatched' if relooks + 1 < max_relook else 'relook_budget_exhausted'
    new_relooks = min(relooks + 1, max_relook)
    if rr == 'relook_budget_exhausted' or new_relooks >= max_relook or close_refusals_after >= 3:
        return 'escape'
    return f'refuse:{rr}'

def main():
    lib = json.load(open(Path(__file__).resolve().parents[1] / 'libs' / 'ee_pg_easy_v2_lib.json'))
    V.validate(lib)
    assert any(p['name'] == 'descent_stalled' for p in lib['predicates'])
    assert any(p['name'] == 'descent_servo_stalled' for p in lib['predicates'])
    assert any(r.get('stall_feedback') or r.get('approach_change') for r in lib['recoveries'])
    assert any(r['id'] == 'hp_descent_stalled_feedback' for r in lib['recoveries'])
    assert relative_yaw_delta_deg({'left_contact': True, 'right_contact': False}, 0) == 45.0
    assert apply_yaw(90, 45) == 45
    assert decide_close_path(False, 4, 4, 1) == 'escape'
    assert decide_close_path(False, 0, 4, 1).startswith('refuse:')
    ns = {'tcp': np.array([0, 0, 0.05]), 'grasp': np.array([0, 0, 0.02]), 'objc': np.zeros(3), 'place': np.zeros(3),
          'opening': 0.08, 'opening_at_close': 0.08, 'tcp_at_close': np.zeros(3), 'obj_h': 0.05, 'obj_w': 0.03,
          'clear': 0.03, 'open_max': 0.08, 'n_same': 2, 'step': 10, 'n_close': 0, 'n_relook': 0,
          'prev': 'descend_to_grasp', 'contact': False, 'left_contact': False, 'right_contact': False,
          'ever_contact': False, 'closed_cmd': False, 'released': False, 'seen_held': False,
          'last_result': 'reached'}
    preds = {}
    for p in lib['predicates']:
        preds[p['name']] = bool(dsl.evaluate(p['expr'], {**ns, **preds}))
    assert preds['descent_stalled'] is True
    print('OK')

if __name__ == '__main__':
    main()
