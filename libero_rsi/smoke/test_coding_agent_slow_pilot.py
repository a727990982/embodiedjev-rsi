"""Smoke: coding-agent slow monitor modes, Codex-shaped plan JSON → validator → applicator accepts.

No MuJoCo / Astra / Codex network.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS


def main():
    assert CAS.backend_mode() == 'codex'
    assert CAS.backend_mode('codex') == 'codex'
    for bad in ('mock', 'dual', 'astra', '0', 'off', '1', 'on', 'auto'):
        try:
            CAS.backend_mode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'unsupported backend {bad!r} was accepted')

    # 1) Codex-shaped neighbour push_aside plan
    neighbour = {
        'goal_object_where': 'language goal near neighbourhood clutter',
        'held_or_between_fingers': 'nothing',
        'is_goal_object': None,
        'matched_language_goal': True,
        'issue': 'blocked_by_neighbour',
        'intervene': True,
        'diagnosis': 'neighbour blocking grasp angle',
        'lesson': 'When a neighbour blocks the grasp, push aside then re-approach before closing.',
        'plan': [{'step': 'push_aside', 'direction': 'away_from_target', 'distance_objw': '1.0',
                  'blocker_px': [210, 300]}],
    }
    ok, errs, norm = CAS.validate_intervention(neighbour)
    assert ok, errs
    assert norm['intervene'] is True
    assert norm['issue'] == 'blocked_by_neighbour'
    assert norm['plan'] == [{'step': 'push_aside', 'blocker_px': [210.0, 300.0],
                             'direction': 'away_from_target', 'distance_objw': 1.0}]
    assert CAS.applicator_accepts(neighbour), norm

    # 2) Codex-shaped ok / no-op reply; issue ok forces intervene False
    noop = {
        'goal_object_where': 'ok',
        'held_or_between_fingers': 'nothing',
        'is_goal_object': None,
        'matched_language_goal': True,
        'issue': 'ok',
        'intervene': True,
        'diagnosis': 'no intervention',
        'lesson': '',
        'plan': [],
    }
    ok_n, errs_n, norm_n = CAS.validate_intervention(noop)
    assert ok_n, errs_n
    assert norm_n['intervene'] is False
    assert norm_n['plan'] == []
    assert CAS.applicator_accepts(noop)

    # 3) Intervene without a usable plan is rejected
    ok_e, errs_e, _ = CAS.validate_intervention({'issue': 'empty_grasp', 'intervene': True, 'plan': []})
    assert not ok_e
    assert 'intervene_without_plan' in errs_e

    # 4) Reject bad step
    bad = {'issue': 'other', 'intervene': True, 'plan': [{'step': 'teleport'}]}
    ok_b, errs_b, _ = CAS.validate_intervention(bad)
    assert not ok_b
    assert any('bad_step' in e for e in errs_b)
    assert not CAS.applicator_accepts(bad)

    # 5) Prompt builds and mentions allowed steps
    ctx = CAS.pack_context(
        lang='put the cream cheese in the basket',
        trigger='hard_stall',
        recent_diag=[{'dec': 12, 'intent': 'descend', 'result': 'stalled'}],
        experience_brief={'active': [{'when': 'blocked_by_neighbour', 'prefer': 'push_aside'}]},
        patch={'grasp_dz_objh': 0.0, 'yaw_deg': -20},
        goal_id={'confirmed': True},
        keyframe_paths=[],
        slow_ctx={
            'intent_result': 'stalled',
            'predicates': {'descent_servo_stalled': True},
            'neighbour_observed': True,
        },
    )
    prompt = CAS.build_prompt(ctx)
    assert 'push_aside' in prompt and 'raise_redescend' in prompt
    assert 'Reply ONLY one JSON object' in prompt

    # 6) Harness still documents CODING_AGENT_SLOW hook
    harness = (Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py').read_text()
    assert 'coding_agent_slow' in harness

    print(json.dumps({
        'ok': True,
        'cases': ['modes', 'neighbour_push', 'ok_noop', 'reject_no_plan', 'reject_bad', 'prompt', 'harness_hook'],
        'sample_plan': norm['plan'],
    }))


if __name__ == '__main__':
    main()
