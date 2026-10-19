"""Logic smoke: place-phase force release on lower thrash (no MuJoCo)."""
from pathlib import Path

import harness_source
LIB = Path(__file__).resolve().parents[1] / 'libs' / 'ee_pg_easy_v2_lib.json'

def main():
    src = harness_source.read()
    assert 'MAX_PLACE_LOWER_STALL' in src
    assert 'MAX_PLACE_LIFT_LOWER_CYCLES' in src
    assert 'def _tcp_near_place_region' in src
    assert 'def _note_place_phase' in src
    assert 'def _maybe_queue_place_force_release' in src
    assert 'place_force_release=True' in src
    assert "EP['force_next_intent'] = 'open_gripper'" in src
    assert '_note_place_phase(it, r)' in src
    assert 'PLACE_CLEARANCE_DEFAULT' in src
    lib = LIB.read_text()
    assert 'hp_place_lower_stall_release' in lib
    assert 'open_gripper' in lib
    print('ok place_force_release_logic')

if __name__ == '__main__':
    main()
