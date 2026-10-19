import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsl_libero as dsl
SHAPE_DEFAULTS = {'tilt_deg': '0', 'tilt_dir_deg': '0'}
FROZEN_REFPOINTS = {'descend_to_grasp': 'grasp'}
def validate(lib):
    preds = []
    for p in lib.get('predicates', []):
        d = dsl.check_dims(p['expr'], preds); assert d == 'B', f"predicate {p['name']} not bool ({d})"; dsl.check_text(p.get('meaning', ''), p['name']); preds.append(p['name'])
    names = [i['name'] for i in lib['intents']]; assert 'finish' in names
    for i in lib['intents']:
        # fine_step is a bounded relative move: same dimensional checks as move; the harness clamps it.
        if i.get('action', 'move') in ('move', 'fine_step'):
            d = dsl.check_dims(i['refpoint'], preds); assert d == 'L', f"refpoint {i['name']} dim {d}"
            g = dsl.check_dims(str(i.get('gain', '1')), preds); assert g in ('C', 'N'), f"gain {i['name']} must be dimensionless"
            for k in ('tilt_deg', 'tilt_dir_deg', 'yaw_deg', 'grasp_dz_objh'):
                if k in i: d = dsl.check_dims(str(i[k]), preds); assert d in ('C', 'N', 'Z'), f"{k} {i['name']} must be dimensionless (degrees)"
        assert i.get('fingers', 'keep') in ('open', 'close', 'keep')
        dsl.check_text(i['definition'], i['name'])
    for i in lib['intents']:  # action-shape DEFAULTS are frozen: seed values only; RSI adapts via conditional recoveries
        assert 'open_width' not in i, 'gripper is binary; no width parameter'
        for k, v in SHAPE_DEFAULTS.items():
            assert str(i.get(k, v)) == v, f"intent {i['name']}: default {k} is frozen ({v}); use a conditional recovery 'shape' override"
        for k in ('yaw_deg', 'grasp_dz_objh'): assert k not in i, f"intent {i['name']}: {k} default is frozen; use a recovery 'shape' override"
        if i['name'] in FROZEN_REFPOINTS: assert i['refpoint'] == FROZEN_REFPOINTS[i['name']], f"{i['name']} refpoint (grasp height default) is frozen"
    for r in lib.get('recoveries', []):
        assert dsl.check_dims(r['trigger'], preds) == 'B'
        has_ctrl = ('force_intent' in r) or ('shape' in r) or r.get('approach_change') or r.get('stall_feedback')
        assert has_ctrl, f"recovery {r.get('id')} needs force_intent, shape, approach_change, and/or stall_feedback"
        if 'force_intent' in r: assert r['force_intent'] in names
        _sf = r.get('stall_feedback')
        _STALL_OK = ('raise_redescend', 'lateral_recenter', 'change_approach_yaw', 'confirm_contact', 'clear_reapproach', 'push_aside')
        if isinstance(_sf, str):
            assert _sf in _STALL_OK, f"recovery {r.get('id')} stall_feedback action {_sf!r} unknown"
        _sfa = r.get('stall_feedback_action')
        if _sfa is not None:
            assert _sfa in _STALL_OK, f"recovery {r.get('id')} stall_feedback_action {_sfa!r} unknown"
        for k, ex in r.get('shape', {}).items():
            assert k in ('tilt_deg', 'tilt_dir_deg', 'yaw_deg', 'yaw_delta_deg', 'grasp_dz_objh', 'grasp_dxy_objw'), f"unknown shape key {k}"
            if k == 'grasp_dxy_objw':
                continue  # list of two dimensionless factors; applied in harness
            d = dsl.check_dims(str(ex), preds)
            assert d in ('C', 'N', 'Z'), f"recovery {r.get('id')} shape {k} has wrong dimension {d}"
    for l in lib.get('lessons', []):  # lessons must be qualitative: no hard-coded numbers (evidence field may cite counts)
        for k in ('name', 'signature', 'instead'):
            assert not any(ch.isdigit() for ch in str(l.get(k, ''))), f"lesson {l.get('id')} field {k} contains a number"
    dsl.check_text(lib.get('intent_instructions', ''), 'intent_instructions')
    return True
if __name__ == '__main__': print(validate(json.load(open(sys.argv[1]))))
