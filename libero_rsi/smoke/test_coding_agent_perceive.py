"""Smoke: coding-agent perceive mock → Astra-shaped keypoints validate (no sim/network)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS


def main():
    assert CAS.perceive_backend_mode('0') == 'off'
    assert CAS.perceive_backend_mode('mock') == 'mock'
    assert CAS.perceive_backend_mode('codex') == 'codex'
    assert CAS.perceive_backend_mode('astra') == 'off'

    ctx = {
        'language_goal': 'put the cream cheese in the basket',
        'phase': 'perceive_start',
        'extra_context': '',
        'img_size': 512,
    }
    parsed, rec = CAS.call_perceive(ctx, mode='mock')
    assert rec['backend'] == 'mock'
    ok, errs, norm = CAS.validate_keypoints(parsed, img_size=512)
    assert ok, errs
    assert isinstance(norm['grasp_px'], list) and len(norm['grasp_px']) == 2
    assert isinstance(norm['place_px'], list) and len(norm['place_px']) == 2
    assert 0 <= norm['grasp_px'][0] < 512 and 0 <= norm['grasp_px'][1] < 512
    assert norm['matched_language_goal'] is True
    assert 0.0 <= norm['grasp_depth_m'] <= 0.06
    assert 'goal_object_named' in norm

    # ketchup variant
    ctx2 = dict(ctx, language_goal='put the ketchup in the basket')
    p2, _ = CAS.call_perceive(ctx2, mode='mock')
    assert 'ketchup' in p2['goal_object_named'].lower()

    # reject missing grasp
    bad_ok, bad_errs, _ = CAS.validate_keypoints({'place_px': [10, 10]}, img_size=512)
    assert not bad_ok
    assert any('grasp_px' in e for e in bad_errs)

    # reject oob
    oob_ok, oob_errs, _ = CAS.validate_keypoints(
        {'grasp_px': [999, 10], 'place_px': [10, 10], 'matched_language_goal': True},
        img_size=512,
    )
    assert not oob_ok
    assert any('grasp_px_oob' in e for e in oob_errs)

    # prompt mentions schema
    prompt = CAS.build_perceive_prompt(ctx)
    assert 'grasp_px' in prompt and 'matched_language_goal' in prompt
    assert 'cream cheese' in prompt.lower() or 'put the cream' in prompt.lower()

    # harness documents hook
    harness = (Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py').read_text()
    assert 'CODING_AGENT_PERCEIVE' in harness
    assert 'harness_call_perceive' in harness
    assert 'coding_agent_perceive' in harness

    # round-trip harness_call_perceive mock
    h_parsed, h_rec = CAS.harness_call_perceive(
        lang='put the butter in the basket',
        phase='relook',
        image_paths=[],
        context='relook after empty close',
        mode='mock',
    )
    assert h_rec['backend'] == 'mock'
    assert 'butter' in h_parsed['goal_object_named'].lower()

    print(json.dumps({
        'ok': True,
        'cases': ['mock_cream', 'mock_ketchup', 'reject_missing', 'reject_oob', 'prompt', 'harness_hook', 'harness_relook'],
        'sample_grasp_px': norm['grasp_px'],
        'sample_place_px': norm['place_px'],
    }))


if __name__ == '__main__':
    main()
