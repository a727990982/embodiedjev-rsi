"""Logic smoke: episode-start Great Wall plan (mock) → push_aside on observed neighbour with confirmed identity; harness hooks."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS
import jev_experience as JX


def main():
    assert CAS.start_plan_backend_mode('0') == 'off'
    assert CAS.start_plan_backend_mode('mock') == 'mock'
    assert CAS.start_plan_backend_mode('codex') == 'codex'
    assert CAS.start_plan_backend_mode('astra') == 'astra'

    # auto inherits CODING_AGENT_SLOW
    os.environ['CODING_AGENT_SLOW'] = 'codex'
    os.environ.pop('CODING_AGENT_PERCEIVE', None)
    assert CAS.start_plan_backend_mode('') == 'codex'
    assert CAS.start_plan_backend_mode('auto') == 'codex'
    os.environ['CODING_AGENT_SLOW'] = '0'
    assert CAS.start_plan_backend_mode('') == 'off'
    os.environ['CODING_AGENT_SLOW'] = 'mock'

    # 1) Observed neighbour + confirmed identity → push_aside lead
    brief = {
        'lessons': [{
            'when': 'at episode start a neighbouring object sits close to the language-goal grasp',
            'instead': 'plan a push_aside or clear_reapproach before the first approach',
        }],
        'recoveries': [{'when': 'blocked_by_neighbour', 'prefer': 'push_aside'}],
    }
    ctx = CAS.pack_start_context(
        lang='pick up the cream cheese and place it in the basket',
        experience_brief=brief,
        start_hints={
            'goal_id_confirmed': True,
            'neighbour_observed': True,
            'neighbour_prior': 'If neighbour may collide, plan push_aside before approach.',
        },
        keyframe_paths=[],
    )
    parsed, rec = CAS.call_start_plan(ctx, mode='mock')
    assert rec['backend'] == 'mock_start'
    ok, errs, norm = CAS.validate_start_plan(parsed)
    assert ok, errs
    assert norm['intervene'] is True
    assert norm['issue'] == 'blocked_by_neighbour'
    assert norm['step1_must'] == 'push_aside'
    assert norm['plan'][0]['step'] == 'push_aside'
    assert 'neighbour_block' in norm['predicted_risks']
    assert norm['phase_order'][0] == 'clear_neighbour'
    assert 'retreat' in norm['phase_order']
    # every place followed by retreat
    for i, ph in enumerate(norm['phase_order']):
        if ph == 'place':
            assert i + 1 < len(norm['phase_order']) and norm['phase_order'][i + 1] == 'retreat'
    assert CAS.applicator_accepts(norm)

    # 2) Clear path → approach / no intervene
    ctx2 = CAS.pack_start_context(
        lang='pick up the butter and place it in the basket',
        experience_brief={'lessons': [], 'recoveries': []},
        start_hints={'goal_id_confirmed': True},
    )
    p2, _ = CAS.call_start_plan(ctx2, mode='mock')
    assert p2['intervene'] is False
    assert p2['plan'] == []
    assert p2['step1_must'] in ('approach', 'none')
    assert p2['phase_order'][-1] == 'retreat'
    assert p2['phase_order'].count('place') == 1

    # 3) Identity soft-unsure, no neighbour → relook
    ctx3 = CAS.pack_start_context(
        lang='soup',
        experience_brief={},
        start_hints={'goal_id_confirmed': False},
    )
    p3, _ = CAS.call_start_plan(ctx3, mode='mock')
    assert p3['plan'] and p3['plan'][0]['step'] == 'relook'
    assert p3['step1_must'] == 'relook'

    # 3b) Soft-unsure + neighbour prior → relook, without inventing identity
    ctx3b = CAS.pack_start_context(
        lang='pick up the cream cheese and place it in the basket',
        experience_brief=brief,
        start_hints={
            'goal_id_confirmed': False,
            'identity_matched': None,
            'verify_confirmed': None,
            'neighbour_prior': 'neighbour may collide — plan push_aside before approach',
        },
    )
    p3b, _ = CAS.call_start_plan(ctx3b, mode='mock')
    assert p3b['plan'][0]['step'] == 'relook', p3b
    assert p3b['step1_must'] == 'relook'
    assert p3b['matched_language_goal'] is None

    # 3c) Multi-object language → two place cycles ending in retreat
    ctx_multi = CAS.pack_start_context(
        lang='put both the cream cheese and the butter in the basket',
        experience_brief={'lessons': [], 'recoveries': []},
        start_hints={'goal_id_confirmed': True},
    )
    pm, _ = CAS.call_start_plan(ctx_multi, mode='mock')
    assert pm['phase_order'].count('place') == 2, pm['phase_order']
    assert pm['phase_order'].count('retreat') >= 2
    for i, ph in enumerate(pm['phase_order']):
        if ph == 'place':
            assert pm['phase_order'][i + 1] == 'retreat'

    # 3d) validate_start_plan inserts retreat after place; soft-warns multi incomplete
    raw_sparse = {
        'matched_language_goal': True,
        'issue': 'ok',
        'intervene': False,
        'predicted_risks': [],
        'step1_must': 'approach',
        'phase_order': ['approach', 'grasp', 'lift', 'place'],
        'diagnosis': 'sparse',
        'lesson': '',
        'plan': [],
    }
    ok_s, errs_s, norm_s = CAS.validate_start_plan(
        raw_sparse, lang='pick A and place it in the basket')
    assert ok_s, errs_s
    assert 'place_retreat_normalized' in errs_s
    assert norm_s['phase_order'][-1] == 'retreat'
    ok_m, errs_m, norm_m = CAS.validate_start_plan(
        dict(raw_sparse),
        lang='put both the cream cheese and the butter in the basket',
    )
    assert ok_m, errs_m
    assert 'multi_object_incomplete_phases' in errs_m
    assert norm_m['phase_order'].count('place') >= 2
    assert all(
        norm_m['phase_order'][i + 1] == 'retreat'
        for i, ph in enumerate(norm_m['phase_order']) if ph == 'place'
    )

    # 4) Prompt mentions Great Wall + experience_brief + retreat + multi-object
    prompt = CAS.build_start_prompt(ctx)
    assert 'Great Wall' in prompt or 'episode-START' in prompt
    assert 'push_aside' in prompt
    assert 'predicted_risks' in prompt
    assert 'experience_brief' in prompt
    assert 'retreat' in prompt
    assert 'both' in prompt.lower() or 'multiple objects' in prompt.lower()

    # 5) experience_brief carries episode_start_plan for Jev
    lib = json.load(open(ROOT / 'libs' / 'ee_pg_easy_v2_lib.json'))
    brief2 = JX.build_jev_experience_brief(lib, episode_start_plan=norm)
    assert 'episode_start_plan' in brief2
    assert brief2['episode_start_plan']['step1_must'] == 'push_aside'
    assert 'hp_start_predict_neighbour' in {L.get('id') for L in lib.get('lessons', [])}

    # 6) Harness hooks
    harness = (Path(__file__).resolve().parents[1] / 'run_ep_fbpatch.py').read_text()
    assert 'EPISODE_START_PLAN' in harness
    assert '_run_episode_start_plan' in harness
    assert 'start_plan.json' in harness
    assert "'experience_brief': brief" in harness
    assert 'hp_start_place_then_retreat' in harness
    assert 'hp_start_multi_object_cycles' in harness
    assert 'episode_start_plan=EP.get' in harness

    print(json.dumps({
        'ok': True,
        'cases': ['modes', 'neighbour_push', 'clear_approach', 'identity_relook',
                  'multi_object', 'validate_retreat', 'prompt', 'brief_inject', 'harness_hook'],
        'sample_plan': norm['plan'],
        'step1_must': norm['step1_must'],
    }))


if __name__ == '__main__':
    main()
