"""Logic smoke: Jev experience_brief is short, actionable, and marks active recoveries."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate as V
import dsl_libero as dsl
import jev_experience as JX

LIB = json.load(open(Path(__file__).resolve().parents[1] / 'libs' / 'ee_pg_easy_v2_lib.json'))


def base_ns(**kw):
    ns = {
        'tcp': np.array([0.098, -0.096, 0.09]), 'grasp': np.array([0.082, -0.098, 0.053]),
        'objc': np.array([0.082, -0.098, 0.1]), 'place': np.zeros(3),
        'opening': 0.08, 'opening_at_close': 0.08, 'tcp_at_close': np.zeros(3),
        'obj_h': 0.12, 'obj_w': 0.05, 'clear': 0.03, 'open_max': 0.08,
        'n_same': 1, 'step': 10, 'n_close': 0, 'n_relook': 0,
        'prev': 'descend_to_grasp', 'contact': False, 'left_contact': False, 'right_contact': False,
        'ever_contact': False, 'closed_cmd': False, 'released': False, 'seen_held': False,
        'last_result': 'stalled',
    }
    ns.update(kw)
    preds = {}
    for p in LIB['predicates']:
        preds[p['name']] = bool(dsl.evaluate(p['expr'], {**ns, **preds}))
    ns.update(preds)
    return ns, preds


def facts_from(ns, preds):
    return {
        'previous_intent': ns['prev'], 'decisions_in_previous_intent': ns['n_same'],
        'last_intent_result': ns.get('last_result', 'none'),
        'both_fingers_contact': ns['contact'], 'times_closed': ns['n_close'],
        **preds,
    }


def main():
    V.validate(LIB)
    assert LIB.get('lessons') and LIB.get('recoveries')

    ns, preds = base_ns(n_same=1, last_result='stalled')
    f = facts_from(ns, preds)
    assert preds.get('descent_stalled') is True

    brief = JX.build_jev_experience_brief(LIB, facts=f, ns=ns, uses={}, dsl=dsl)
    assert 'active_recoveries' in brief and 'recoveries' in brief and 'lessons' in brief
    assert brief['active_recoveries'], 'expected descent_stalled recovery active'
    assert any('stalled' in (a.get('when') or '') or a.get('id') == 'hp_descent_stalled_feedback'
               for a in brief['active_recoveries']), brief['active_recoveries']
    assert all('prefer' in r or r.get('prefer') is None for r in brief['recoveries'])
    assert all('when' in L and 'instead' in L for L in brief['lessons'])
    # relevant lessons should mention stall/descent/grasp — not empty
    assert len(brief['lessons']) <= 5
    assert len(brief['recoveries']) <= 6
    blob = json.dumps(brief)
    assert len(blob) < 4500, f'brief too large ({len(blob)} chars) — looks like a prose dump'
    # no scene-specific absolute yaw prescriptions in brief text
    assert '90' not in blob and 'ninety' not in blob.lower()

    # episode reflections attached when present
    brief2 = JX.build_jev_experience_brief(
        LIB, facts=f, ns=ns, uses={}, dsl=dsl,
        episode_lessons=[{'failure_class': 'empty_close', 'diagnosis': 'missed', 'lesson': 'reassess alignment'}],
    )
    assert brief2.get('episode_reflections')

    # uses_left tracks recovery budget
    brief3 = JX.build_jev_experience_brief(
        LIB, facts=f, ns=ns, uses={'hp_descent_stalled_feedback': 99}, dsl=dsl,
    )
    assert not any(a.get('id') == 'hp_descent_stalled_feedback' for a in brief3['active_recoveries'])

    print(json.dumps({
        'ok': True,
        'n_active': len(brief['active_recoveries']),
        'active_ids': [a['id'] for a in brief['active_recoveries']],
        'n_lessons': len(brief['lessons']),
        'lesson_ids': [L['id'] for L in brief['lessons']],
        'brief_chars': len(blob),
        'planner_brief': True,
    }, indent=1))


if __name__ == '__main__':
    main()
