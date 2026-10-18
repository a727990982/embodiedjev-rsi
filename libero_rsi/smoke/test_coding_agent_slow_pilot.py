"""Smoke: coding-agent slow monitor mock → valid plan JSON → applicator accepts.

No MuJoCo / Astra / Codex network.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS


def main():
    assert CAS.backend_mode('mock') == 'mock'
    assert CAS.backend_mode('codex') == 'codex'

    # 1) Neighbour stall → push_aside
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
    parsed, rec = CAS.call_monitor(ctx, mode='mock')
    assert rec['backend'] == 'mock'
    ok, errs, norm = CAS.validate_intervention(parsed)
    assert ok, errs
    assert norm['intervene'] is True
    assert norm['issue'] == 'blocked_by_neighbour'
    assert norm['plan'][0]['step'] == 'push_aside'
    assert CAS.applicator_accepts(norm), norm

    # 2) Empty grasp → raise_redescend
    ctx2 = CAS.pack_context(
        lang='put the ketchup in the basket',
        trigger='spectator_defer',
        recent_diag=[],
        experience_brief={},
        patch={},
        goal_id=True,
        slow_ctx={'intent_result': 'empty_close', 'predicates': {'empty_close': True}},
    )
    p2, _ = CAS.call_monitor(ctx2, mode='mock')
    assert p2['plan'][0]['step'] == 'raise_redescend'
    assert CAS.applicator_accepts(p2)

    # 3) Identity → relook
    ctx3 = CAS.pack_context(
        lang='butter',
        trigger='hard_identity_unstable',
        recent_diag=[],
        experience_brief={},
        patch={},
        goal_id=False,
        slow_ctx={'intent_result': 'close_refused_identity'},
    )
    p3, _ = CAS.call_monitor(ctx3, mode='mock')
    assert p3['plan'][0]['step'] == 'relook'
    assert CAS.applicator_accepts(p3)

    # 4) Progressing → ok / empty plan
    ctx4 = CAS.pack_context(
        lang='cream',
        trigger='async_watch',
        recent_diag=[],
        experience_brief={},
        patch={},
        goal_id=True,
        slow_ctx={'intent_result': 'approached', 'predicates': {}},
    )
    p4, _ = CAS.call_monitor(ctx4, mode='mock')
    assert p4['intervene'] is False
    assert p4['plan'] == []
    assert CAS.applicator_accepts(p4)

    # 5) Reject bad step
    bad = {'issue': 'other', 'intervene': True, 'plan': [{'step': 'teleport'}]}
    ok_b, errs_b, _ = CAS.validate_intervention(bad)
    assert not ok_b
    assert any('bad_step' in e for e in errs_b)

    # 6) Prompt builds and mentions allowed steps
    prompt = CAS.build_prompt(ctx)
    assert 'push_aside' in prompt and 'raise_redescend' in prompt
    assert 'Reply ONLY one JSON object' in prompt

    # 7) Harness retains the Codex monitor hook
    harness = (Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py').read_text()
    assert "CAS.call_monitor(snapshot['coding_context'], mode='codex')" in harness
    assert 'coding_agent_slow' in harness

    print(json.dumps({
        'ok': True,
        'cases': ['neighbour_push', 'empty_raise', 'identity_relook', 'ok_noop', 'reject_bad', 'prompt', 'harness_hook'],
        'sample_plan': norm['plan'],
    }))


if __name__ == '__main__':
    main()
