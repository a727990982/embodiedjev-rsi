"""Logic smoke: hybrid Astra help (ask_astra intent + hard stall/identity force + shared MAX_SLOW).

No MuJoCo / Astra. Mirrors harness rules in run_ep_fbpatch.py.
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate as V

import harness_source
LIBP = Path(__file__).resolve().parents[1] / 'libs' / 'ee_pg_easy_v2_lib.json'

ASK_ASTRA_NAMES = frozenset({'ask_astra', 'consult_astra'})
FAIL_RESULTS = frozenset({'timeout', 'stalled'})
MAX_SLOW = 4


def ensure_ask_astra_intents(lib):
    names = {i['name'] for i in lib.get('intents', [])}
    ask = {
        'id': 'ask_astra@v0', 'name': 'ask_astra', 'action': 'ask_astra', 'fingers': 'keep',
        'definition': 'Request Astra slow-think help when stuck.',
        'refpoint': 'tcp', 'status': 'hypothesis', 'evidence': 'hybrid',
        'tilt_deg': '0', 'tilt_dir_deg': '0',
    }
    consult = dict(ask, id='consult_astra@v0', name='consult_astra', action='consult_astra',
                   definition='Alias of ask_astra.')
    fin_i = next((i for i, it in enumerate(lib['intents']) if it['name'] == 'finish'), len(lib['intents']))
    if 'ask_astra' not in names:
        lib['intents'].insert(fin_i, ask); fin_i += 1
    if 'consult_astra' not in names:
        lib['intents'].insert(fin_i, consult)
    tip = ' you may select ask_astra (or consult_astra)'
    if 'ask_astra' not in (lib.get('intent_instructions') or ''):
        lib['intent_instructions'] = (lib.get('intent_instructions') or '') + tip
    return lib


def hard_astra_trigger(intent, result, *, goal_id_ok, slow_used, n_same, fallback=True):
    if not fallback or slow_used >= MAX_SLOW:
        return None
    if intent in ASK_ASTRA_NAMES:
        return None
    if result in FAIL_RESULTS:
        return 'hard_stall'
    if not goal_id_ok:
        rs = str(result or '')
        if (rs.startswith('close_refused') or rs.startswith('identity_escape')
                or rs in ('relooked_unmatched', 'close_refused_identity')
                or (intent not in ('relook', 'finish') and n_same >= 2)):
            return 'hard_identity_unstable'
    return None


def shared_budget_after_consult(slow_used, intervened):
    """Intervene already counts; ok-consult still consumes one shared slot."""
    if intervened:
        return min(MAX_SLOW, slow_used)  # already bumped
    if slow_used < MAX_SLOW:
        return slow_used + 1
    return slow_used


def main():
    src = harness_source.read()
    ast.parse(src)
    for needle in (
        'ASK_ASTRA_NAMES', '_ensure_ask_astra_intents', 'def do_ask_astra(',
        'def _hard_astra_trigger(', '_consume_help_budget',
        "act in ASK_ASTRA_NAMES", 'hard_stall', 'hard_identity_unstable',
        "source='ask_astra'", 'jev_ask:',
    ):
        assert needle in src, needle

    lib = json.load(open(LIBP))
    names = [i['name'] for i in lib['intents']]
    assert 'ask_astra' in names and 'consult_astra' in names, names
    assert names.index('ask_astra') < names.index('finish')
    ask = next(i for i in lib['intents'] if i['name'] == 'ask_astra')
    assert ask['action'] == 'ask_astra' and ask.get('fingers') == 'keep'
    assert 'ask_astra' in lib['intent_instructions']
    V.validate(lib)

    # ensure() is idempotent on libs that already have the intents
    n_before = len(lib['intents'])
    ensure_ask_astra_intents(lib)
    assert len(lib['intents']) == n_before
    V.validate(lib)

    # seed lib without intents still gets them
    seed = {'intents': [
        {'id': 'x@v0', 'name': 'move_above_object', 'action': 'move', 'fingers': 'open',
         'definition': 'go above', 'refpoint': 'grasp', 'tilt_deg': '0', 'tilt_dir_deg': '0'},
        {'id': 'f@v0', 'name': 'finish', 'action': 'finish', 'fingers': 'keep',
         'definition': 'stop', 'refpoint': 'tcp', 'tilt_deg': '0', 'tilt_dir_deg': '0'},
    ], 'predicates': [], 'recoveries': [], 'lessons': [], 'intent_instructions': 'pick'}
    ensure_ask_astra_intents(seed)
    sn = [i['name'] for i in seed['intents']]
    assert sn == ['move_above_object', 'ask_astra', 'consult_astra', 'finish'], sn

    # hard: stall forces
    assert hard_astra_trigger('descend_to_grasp', 'stalled', goal_id_ok=True, slow_used=0, n_same=1) == 'hard_stall'
    assert hard_astra_trigger('descend_to_grasp', 'timeout', goal_id_ok=True, slow_used=0, n_same=1) == 'hard_stall'
    # hard: identity unstable
    assert hard_astra_trigger('close_gripper', 'close_refused_identity:x', goal_id_ok=False, slow_used=0, n_same=1) == 'hard_identity_unstable'
    assert hard_astra_trigger('move_above_object', 'reached', goal_id_ok=False, slow_used=0, n_same=2) == 'hard_identity_unstable'
    # no force when identity ok and progress
    assert hard_astra_trigger('descend_to_grasp', 'reached', goal_id_ok=True, slow_used=0, n_same=1) is None
    # ask_astra itself does not hard-retrigger
    assert hard_astra_trigger('ask_astra', 'stalled', goal_id_ok=True, slow_used=0, n_same=1) is None
    # budget exhausted → no hard force
    assert hard_astra_trigger('descend_to_grasp', 'stalled', goal_id_ok=True, slow_used=MAX_SLOW, n_same=1) is None
    # relook while identity bad does not hard-force on first try
    assert hard_astra_trigger('relook', 'relooked_matched', goal_id_ok=False, slow_used=0, n_same=1) is None

    # shared budget: ok-consult consumes; intervene already counted
    assert shared_budget_after_consult(0, intervened=False) == 1
    assert shared_budget_after_consult(3, intervened=False) == 4
    assert shared_budget_after_consult(4, intervened=False) == 4
    assert shared_budget_after_consult(2, intervened=True) == 2

    print('OK ask_astra_help_logic')


if __name__ == '__main__':
    main()
