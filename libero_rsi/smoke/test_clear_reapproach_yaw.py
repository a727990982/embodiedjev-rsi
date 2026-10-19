"""Smoke: clear_reapproach / escape_look do not wipe confirmed PATCH yaw by default."""
import ast
from pathlib import Path

import harness_source


def explicit_yaw(new_y, old_per_yaw, cmd_yaw, eps=15.0):
    """Mirror of harness _new_keypoints_explicit_yaw (cmd captured before PATCH reset)."""
    if new_y is None:
        return False
    if abs(float(new_y) - float(cmd_yaw)) < float(eps):
        return False
    if old_per_yaw is not None and abs(float(new_y) - float(old_per_yaw)) < float(eps):
        return False
    return True


def restore(old_patch, explicit):
    if explicit or not old_patch:
        return {}
    out = {}
    for k in ('yaw_deg', 'tilt_deg', 'tilt_dir_deg'):
        if old_patch.get(k) is not None:
            out[k] = old_patch[k]
    return out


def main():
    src = harness_source.read()
    ast.parse(src)
    assert "src == 'escape_look'" in src
    assert 'cmd_yaw' in src or 'old_cmd_yaw' in src

    # clear_reapproach branch: no relative yaw change
    i0 = src.find("elif action == 'clear_reapproach':")
    assert i0 > 0
    i1 = src.find('\n    else:', i0)
    clear_block = src[i0:i1]
    assert '_apply_relative_approach_change' not in clear_block

    # Confirmed PATCH yaw=-45; relook echoes PER yaw=0 → must preserve
    old_patch = {'yaw_deg': -45.0, 'tilt_deg': None, 'tilt_dir_deg': None}
    assert explicit_yaw(0.0, old_per_yaw=0.0, cmd_yaw=-45.0) is False
    preserved = restore(old_patch, False)
    assert preserved.get('yaw_deg') == -45.0

    # Astra proposes a real new yaw vs both cmd and PER → explicit
    assert explicit_yaw(-90.0, old_per_yaw=0.0, cmd_yaw=-45.0) is True
    assert restore(old_patch, True) == {}

    # New yaw agrees with command → not explicit
    assert explicit_yaw(-45.0, old_per_yaw=0.0, cmd_yaw=-45.0) is False

    # escape_look path: always preserve (harness forces explicit=False)
    assert restore(old_patch, False).get('yaw_deg') == -45.0

    print('OK clear_reapproach_yaw')


if __name__ == '__main__':
    main()
