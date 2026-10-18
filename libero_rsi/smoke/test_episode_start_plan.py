"""Logic smoke: episode-start Great Wall plan modes, Codex-shaped plan validation, identity gate, brief injection, harness hooks."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coding_agent_slow as CAS
import jev_experience as JX


def main():
    assert CAS.start_plan_backend_mode() == 'codex'
    assert CAS.start_plan_backend_mode('codex') == 'codex'
    for bad in ('mock', 'dual', 'astra', '0', 'off', '1', 'on', 'auto'):
        try:
            CAS.start_plan_backend_mode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'unsupported backend {bad!r} was accepted')

    # 1) Codex-shaped neighbour plan; place without retreat is normalized
    lang = 'pick up the cream cheese and place it in the basket'
    raw_plan = {
        'matched_language_goal': True,
        'issue': 'blocked_by_neighbour',
        'intervene': True,
        'predicted_risks': ['neighbour_block'],
        'step1_must': 'push_aside',
        'phase_order': ['clear_neighbour', 'approach', 'grasp', 'lift', 'place'],
        'diagnosis': 'neighbour near target — clear before first approach',
        'lesson': 'When a neighbour sits next to the target at start, push it aside before approaching.',
        'plan': [{'step': 'push_aside', 'direction': 'away_from_target', 'distance_objw': 1.0}],
    }
    ok, errs, norm = CAS.validate_start_plan(raw_plan, lang=lang)
    assert ok, errs
    assert 'place_retreat_normalized' in errs
    assert norm['intervene'] is True
    assert norm['issue'] == 'blocked_by_neighbour'
    assert norm['step1_must'] == 'push_aside'
    assert norm['plan'][0]['step'] == 'push_aside'
    assert 'neighbour_block' in norm['predicted_risks']
    assert norm['phase_order'][0] == 'clear_neighbour'
    for i, ph in enumerate(norm['phase_order']):
        if ph == 'place':
            assert i + 1 < len(norm['phase_order']) and norm['phase_order'][i + 1] == 'retreat'
    assert CAS.applicator_accepts(norm)

    # 2) Reject unknown risk / step1 / phase
    ok_bad, errs_bad, _ = CAS.validate_start_plan(
        dict(raw_plan, predicted_risks=['meteor'], step1_must='teleport', phase_order=['fly']), lang=lang)
    assert not ok_bad
    assert {'bad_risk:meteor', 'bad_step1:teleport', 'bad_phase:fly'} <= set(errs_bad)

    # 3) step1 push_aside without a leading plan step gets one injected
    ok_inj, errs_inj, norm_inj = CAS.validate_start_plan(
        dict(raw_plan, plan=[{'step': 'relook'}]), lang=lang)
    assert ok_inj, errs_inj
    assert 'step1_plan_injected' in errs_inj
    assert norm_inj['plan'][0]['step'] == 'push_aside'

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

    # 3e) Identity gate: an unconfirmed target cannot be cleared or approached
    gated = CAS.gate_start_plan(norm, confirmed=False)
    assert gated['plan'] == [{'step': 'relook'}]
    assert gated['step1_must'] == 'relook'
    assert gated['phase_order'][0] == 'relook' and 'clear_neighbour' not in gated['phase_order']
    assert 'identity_ambiguous' in gated['predicted_risks']
    assert gated['matched_language_goal'] is None
    refused = CAS.gate_start_plan(dict(norm, matched_language_goal=False), confirmed=True)
    assert refused['plan'] == [{'step': 'relook'}] and refused['issue'] == 'wrong_object'
    assert CAS.gate_start_plan(norm, confirmed=True) == norm

    # 4) Prompt mentions Great Wall + experience_brief + retreat + multi-object
    brief = {
        'lessons': [{
            'when': 'at episode start a neighbouring object sits close to the language-goal grasp',
            'instead': 'plan a push_aside or clear_reapproach before the first approach',
        }],
        'recoveries': [{'when': 'blocked_by_neighbour', 'prefer': 'push_aside'}],
    }
    ctx = CAS.pack_start_context(
        lang=lang,
        experience_brief=brief,
        start_hints={
            'goal_id_confirmed': True,
            'neighbour_observed': True,
            'neighbour_prior': 'If neighbour may collide, plan push_aside before approach.',
        },
        keyframe_paths=[],
    )
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
    assert '_run_episode_start_plan' in harness
    assert 'start_plan.json' in harness
    assert "'experience_brief': brief" in harness
    assert 'hp_start_place_then_retreat' in harness
    assert 'hp_start_multi_object_cycles' in harness
    assert 'episode_start_plan=EP.get' in harness

    print(json.dumps({
        'ok': True,
        'cases': ['modes', 'neighbour_push', 'reject_bad', 'step1_inject',
                  'validate_retreat', 'identity_gate', 'prompt', 'brief_inject', 'harness_hook'],
        'sample_plan': norm['plan'],
        'step1_must': norm['step1_must'],
    }))


if __name__ == '__main__':
    main()
