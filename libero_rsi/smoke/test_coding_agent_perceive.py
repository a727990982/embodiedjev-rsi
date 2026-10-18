"""Smoke: coding-agent perceive modes, Codex-shaped keypoints → Astra-shaped validation (no sim/network)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS


def main():
    assert CAS.perceive_backend_mode('0') == 'off'
    assert CAS.perceive_backend_mode('codex') == 'codex'
    assert CAS.perceive_backend_mode('astra') == 'off'
    for bad in ('mock', '1', 'on'):
        try:
            CAS.perceive_backend_mode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'perceive_backend_mode({bad!r}) should raise')

    # Codex-shaped reply: no object_center_px, out-of-range depth, string match flag
    reply = {
        'goal_object_named': 'cream cheese',
        'goal_object_where': 'cream cheese on the table near image centre',
        'place_region_named': 'basket',
        'place_region_where': 'basket toward image right',
        'matched_language_goal': 'true',
        'grasp_px': [196, 310],
        'grasp_yaw_deg': 5,
        'grasp_depth_m': 0.5,
        'object_height_m': 0.025,
        'place_px': [416.0, 243.0],
        'object_held_by_gripper': 'false',
        'confidence': 0.85,
    }
    ok, errs, norm = CAS.validate_keypoints(reply, img_size=512)
    assert ok, errs
    assert norm['grasp_px'] == [196.0, 310.0]
    assert norm['place_px'] == [416.0, 243.0]
    assert norm['object_center_px'] == norm['grasp_px']
    assert norm['grasp_depth_m'] == 0.06
    assert norm['object_width_m'] == 0.05
    assert norm['matched_language_goal'] is True
    assert norm['object_held_by_gripper'] is False
    assert norm['goal_object_named'] == 'cream cheese'

    # matched_language_goal normalization
    for raw, want in (('false', False), (False, False), ('unsure', None), (None, None)):
        _, _, n = CAS.validate_keypoints(dict(reply, matched_language_goal=raw), img_size=512)
        assert n['matched_language_goal'] is want, (raw, n['matched_language_goal'])

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
    ctx = {
        'language_goal': 'put the cream cheese in the basket',
        'phase': 'perceive_start',
        'extra_context': '',
        'img_size': 512,
    }
    prompt = CAS.build_perceive_prompt(ctx)
    assert 'grasp_px' in prompt and 'matched_language_goal' in prompt
    assert 'cream cheese' in prompt.lower() or 'put the cream' in prompt.lower()

    # harness documents hook
    harness = (Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py').read_text()
    assert 'CODING_AGENT_PERCEIVE' in harness
    assert 'harness_call_perceive' in harness
    assert 'coding_agent_perceive' in harness

    print(json.dumps({
        'ok': True,
        'cases': ['modes', 'normalize_reply', 'match_normalize', 'reject_missing', 'reject_oob', 'prompt', 'harness_hook'],
        'sample_grasp_px': norm['grasp_px'],
        'sample_place_px': norm['place_px'],
    }))


if __name__ == '__main__':
    main()
