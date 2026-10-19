"""Smoke: opt-in Jev recovery menu. No MuJoCo, network, or Jev client.

Default rules mode hides the five recoveries and leaves the experience brief unchanged.
jev mode offers them and skips stall_feedback / approach_change rules.
"""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dsl_libero as dsl
import jev_experience as JX
import jev_recovery_menu as JRM
import validate as V

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'run_ep_fbpatch.py'
V2 = ROOT / 'libs' / 'ee_pg_easy_v2_lib.json'
FROZEN = ROOT / 'libs' / 'ee_pg_lib_frozen.json'
MENU = ROOT / 'libs' / 'ee_pg_easy_v2_lib_jev_recovery.json'
ACTIONS = JRM.RECOVERY_ACTIONS


def _ns():
    ns = {
        'tcp': np.array([0.098, -0.096, 0.09]), 'grasp': np.array([0.082, -0.098, 0.053]),
        'objc': np.array([0.082, -0.098, 0.1]), 'place': np.zeros(3),
        'opening': 0.08, 'opening_at_close': 0.08, 'tcp_at_close': np.zeros(3),
        'obj_h': 0.12, 'obj_w': 0.05, 'clear': 0.03, 'open_max': 0.08,
        'n_same': 1, 'step': 10, 'n_close': 0, 'n_relook': 0,
        'prev': 'descend_to_grasp', 'contact': False, 'left_contact': False, 'right_contact': False,
        'ever_contact': False, 'closed_cmd': False, 'released': False, 'seen_held': False,
        'last_result': 'stalled', 'clear_reapproach_n': 0, 'push_aside_n': 0,
    }
    preds = {}
    for p in json.loads(V2.read_text())['predicates']:
        preds[p['name']] = bool(dsl.evaluate(p['expr'], {**ns, **preds}))
    ns.update(preds)
    return ns


def main():
    src = HARNESS.read_text()
    ast.parse(src)
    for needle in (
        'JEV_RECOVERY_MENU', 'recovery_rule_skipped', '_execute_jev_recovery',
        'facts_extra', 'acting_intents', '_note_jev_recovery_outcome', 'choice_menu',
        'recovery_outcomes', "_recovery_mode != 'jev'", 'rules_auto_choose_enabled',
    ):
        assert needle in src, needle
    # Skip auto-rules before burning their use counter.
    skip_at = src.find('recovery_rule_skipped')
    uses_at = src.find("EP['uses'][rc['id']] = uses + 1", skip_at)
    assert 0 < skip_at < uses_at

    # --- mode parse: default and typos stay on rules ---
    assert JRM.parse_mode(None) == ('rules', None)
    assert JRM.parse_mode('') == ('rules', None)
    for raw in ('rules', '0', 'off', 'false', 'auto', 'rule', ' RULES '):
        assert JRM.parse_mode(raw)[0] == 'rules', raw
    for raw in ('jev', '1', 'on', 'true', 'yes', 'menu', 'Jev'):
        assert JRM.parse_mode(raw) == ('jev', None), raw
    for raw in ('both', 'jev+rules', 'rules+jev'):
        assert JRM.parse_mode(raw)[0] == 'both', raw
    mode, warn = JRM.parse_mode('sometimes')
    assert mode == 'rules' and warn and 'sometimes' in warn
    assert JRM.menu_enabled('rules') is False
    assert JRM.menu_enabled('jev') and JRM.menu_enabled('both')
    assert JRM.rules_auto_choose_enabled('rules') and JRM.rules_auto_choose_enabled('both')
    assert JRM.rules_auto_choose_enabled('jev') is False
    assert JRM.instruction_extra('rules') == ''
    assert 'automatic stall-feedback rules are off' in JRM.instruction_extra('jev')
    assert 'still run' in JRM.instruction_extra('both')

    # --- rule skip: stall / approach_change only, and only in jev ---
    stall = {'id': 's', 'stall_feedback': True, 'force_intent': 'move_above_object'}
    named = {'id': 'p', 'stall_feedback': 'push_aside', 'force_intent': 'move_above_object'}
    yaw = {'id': 'y', 'approach_change': True}
    place = {'id': 'rel', 'force_intent': 'open_gripper'}
    for rc in (stall, named, yaw):
        assert JRM.recovery_rule_skipped(rc, 'jev') is True
        assert JRM.recovery_rule_skipped(rc, 'rules') is False
        assert JRM.recovery_rule_skipped(rc, 'both') is False
    assert JRM.recovery_rule_skipped(place, 'jev') is False

    # --- menu filter ---
    specs = JRM.recovery_intent_specs()
    assert [s['name'] for s in specs] == list(ACTIONS)
    base = [
        {'name': 'move_above_object', 'action': 'move'},
        *specs,
        {'name': 'finish', 'action': 'finish'},
    ]
    hidden = [i['name'] for i in JRM.acting_intents(base, 'rules')]
    assert hidden == ['move_above_object', 'finish'], hidden
    assert [i['name'] for i in JRM.acting_intents(base, 'jev')] == [i['name'] for i in base]
    assert [i['name'] for i in JRM.acting_intents(base, 'both')] == [i['name'] for i in base]

    # --- outcomes Jev can read ---
    raised = JRM.format_recovery_outcome(
        'raise_redescend', {'action': 'raise_redescend', 'reason': 'jev_menu:raise_redescend'},
        force_next='move_above_object')
    assert raised.startswith('raise_redescend:ok:raised_and_shallowed:next_move_above_object')
    lateral = JRM.format_recovery_outcome('lateral_recenter', {'action': 'lateral_recenter'})
    assert 'shifted_toward_centre' in lateral
    yaw_ok = JRM.format_recovery_outcome('change_approach_yaw', {'action': 'change_approach_yaw', 'yaw_to': 45.0})
    assert 'yaw_rotated' in yaw_ok and 'yaw_unchanged' not in yaw_ok
    yaw_same = JRM.format_recovery_outcome('change_approach_yaw', {'action': 'change_approach_yaw'})
    assert 'yaw_unchanged' in yaw_same
    cleared = JRM.format_recovery_outcome('clear_reapproach', {'action': 'clear_reapproach', 'clear_reapproach_n': 1})
    assert 'raised_recenter_yaw_kept' in cleared and 'yaw_rotated' not in cleared
    pushed = JRM.format_recovery_outcome('push_aside', {
        'action': 'push_aside', 'pushed': True, 'blocker_mode': 'heuristic_open_axis',
        'servo': 'stalled', 'need_reapproach': True,
    })
    assert pushed.startswith('push_aside:ok:pushed:heuristic_open_axis:servo_stalled:need_reapproach')
    refused = JRM.format_recovery_outcome('push_aside', {
        'action': 'push_aside', 'error': 'push_aside_refused_identity', 'pushed': False,
    })
    assert refused.startswith('push_aside:refused:push_aside_refused_identity')
    capped = JRM.format_recovery_outcome('push_aside', {
        'action': 'push_aside', 'error': 'push_aside_episode_cap:2>=2',
    })
    assert 'refused' in capped and 'push_aside_episode_cap' in capped
    secured = JRM.format_recovery_outcome('push_aside', {'action': 'skip_secured_grasp'}, force_next='lift')
    assert secured.startswith('push_aside:skipped:grasp_secured:next_lift')
    dead = JRM.format_recovery_outcome('clear_reapproach', {
        'action': 'local_deadend_global_relook', 'deadend_reason': 'push_aside_episode_cap',
    })
    assert dead.startswith('clear_reapproach:intercepted:local_deadend:push_aside_episode_cap')

    ep = {'push_aside_n': 2, 'push_aside_approach_n': 0, 'clear_reapproach_n': 1,
          'need_reapproach': True, 'stall_feedbacks': 3, 'last_stall_issue': 'blocked_by_neighbour',
          'recovery_outcomes': [raised, refused]}
    assert JRM.facts_extra('rules', ep) == {}
    assert JRM.facts_extra('off', ep) == {}
    extra = JRM.facts_extra('jev', ep, push_cap=2, push_approach_cap=1)
    assert extra['recovery_menu'] == 'jev'
    assert extra['last_recovery_outcome'] == refused
    assert extra['recent_recovery_outcomes'] == [raised, refused]
    assert extra['push_aside_capped'] is True
    assert extra['clear_reapproach_count'] == 1
    assert extra['need_reapproach'] is True
    assert extra['last_stall_issue'] == 'blocked_by_neighbour'
    assert extra['stall_feedback_count'] == 3
    remembered = {}
    JRM.remember(remembered, 'one')
    JRM.remember(remembered, 'two')
    assert remembered['recovery_outcomes'] == ['one', 'two']

    # --- libraries: frozen and v2 unchanged; menu file is opt-in and valid ---
    v2 = json.loads(V2.read_text())
    frozen = json.loads(FROZEN.read_text())
    menu = json.loads(MENU.read_text())
    assert v2['version'] == 'ee_pg_v14+place_release'
    assert not (set(i['name'] for i in v2['intents']) & set(ACTIONS))
    assert not (set(i['name'] for i in frozen['intents']) & set(ACTIONS))
    V.validate(v2)
    V.validate(frozen)
    V.validate(menu)
    assert 'jev_recovery_menu' in menu['version']
    by_name = {i['name']: i for i in menu['intents']}
    for spec in specs:
        got = by_name[spec['name']]
        assert got['action'] == spec['name'] and got['action'] != 'move'
        assert got['definition'] == spec['definition']
        assert got['fingers'] == 'keep'
    assert menu['intents'][-1]['name'] == 'finish'
    assert any(r.get('stall_feedback') for r in menu['recoveries'])
    assert 'raise_redescend' in menu['intent_instructions']
    assert any(L['id'] == 'hp_jev_recovery_menu' for L in menu['lessons'])

    lib = json.loads(V2.read_text())
    n0 = len(lib['intents'])
    JRM.ensure_recovery_intents(lib)
    JRM.ensure_recovery_intents(lib)
    assert len(lib['intents']) == n0 + len(ACTIONS)
    assert lib['intents'][-1]['name'] == 'finish'
    assert [i['name'] for i in lib['intents']].index('raise_redescend') < [i['name'] for i in lib['intents']].index('finish')
    V.validate(lib)
    # rules choice criteria drop the injected intents; jev keeps them
    rules_menu = [i['name'] for i in JRM.acting_intents(lib['intents'], 'rules')]
    jev_menu = [i['name'] for i in JRM.acting_intents(lib['intents'], 'jev')]
    assert not (set(rules_menu) & set(ACTIONS))
    assert set(ACTIONS) <= set(jev_menu)

    # --- experience brief: rules is identical; jev retargets auto stall rules ---
    ns = _ns()
    assert ns['descent_stalled'] is True
    facts = {'previous_intent': 'descend_to_grasp', 'last_intent_result': 'stalled', 'descent_stalled': True}
    brief_default = JX.build_jev_experience_brief(v2, facts=facts, ns=ns, uses={}, dsl=dsl)
    brief_rules = JX.build_jev_experience_brief(v2, facts=facts, ns=ns, uses={}, dsl=dsl, recovery_menu='rules')
    assert brief_default == brief_rules
    brief_jev = JX.build_jev_experience_brief(v2, facts=facts, ns=ns, uses={}, dsl=dsl, recovery_menu='jev')
    assert 'Automatic stall-feedback rules are off' in brief_jev['note']
    cat = {r['id']: r for r in brief_jev['recoveries']}
    assert cat['hp_neighbour_block_push_aside']['auto'] is False
    assert cat['hp_neighbour_block_push_aside']['prefer'] == 'push_aside'
    assert cat['hp_descent_stalled_feedback']['auto'] is False
    assert cat['hp_descent_stalled_feedback']['prefer'] is None
    assert 'auto' not in cat['hp_place_lower_stall_release']
    brief_both = JX.build_jev_experience_brief(v2, facts=facts, ns=ns, uses={}, dsl=dsl, recovery_menu='both')
    both_cat = {r['id']: r for r in brief_both['recoveries']}
    assert both_cat['hp_descent_stalled_feedback']['auto'] is True
    assert both_cat['hp_descent_stalled_feedback']['prefer'] == 'move_above_object'
    assert 'automatic rule may run before your choice' in both_cat['hp_descent_stalled_feedback']['hint']

    # --- harness loop: jev skips stall rules without spending uses; place release still forces ---
    tree = ast.parse(src)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'rc')
    calls = []

    def run_loop(mode, recoveries, intents):
        calls.clear()
        ep = {'uses': {}, 'force_next_intent': None}
        scope = {
            'LIB': {'recoveries': [dict(rc, trigger='True') for rc in recoveries]},
            'forced': None,
            'dsl': SimpleNamespace(evaluate=lambda expr, ns: True),
            'ns': {},
            'dec': 0,
            'T': [0],
            'EP': ep,
            'JRM': JRM,
            'JEV_RECOVERY_MODE': mode,
            'intents': intents,
            'log': lambda **kw: calls.append(kw),
            '_apply_stall_feedback': lambda *a, **kw: calls.append('stall') or {'action': 'push_aside'},
            'STALL_FB_ACTIONS': ACTIONS,
        }
        exec(compile(ast.Module(body=[loop], type_ignores=[]), '<recovery>', 'exec'), scope)
        return scope['forced'], ep, list(calls)

    forced, ep, fired = run_loop('jev', [stall, place], {'open_gripper': {}, 'move_above_object': {}})
    assert forced == 'open_gripper', forced
    assert 's' not in ep['uses'], ep
    assert ep['uses'].get('rel') == 1
    assert 'stall' not in fired
    forced_b, ep_b, fired_b = run_loop('both', [stall], {'move_above_object': {}})
    assert 'stall' in fired_b and ep_b['uses'].get('s') == 1
    assert forced_b == 'move_above_object'
    forced_r, ep_r, fired_r = run_loop('rules', [named], {'move_above_object': {}})
    assert 'stall' in fired_r and ep_r['uses'].get('p') == 1

    print('OK jev_recovery_menu')


if __name__ == '__main__':
    main()
