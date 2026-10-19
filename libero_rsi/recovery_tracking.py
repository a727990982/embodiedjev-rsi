"""Shared automatic-recovery checkpoint; no hashes, models or simulator access."""


def handled(ep, step):
    return ep.get('recovery_handled_state') == [step, ep.get('todo_target_id'), bool(ep.get('closed'))]


def mark(ep, step):
    # Record the END of an intervention too: its internal motion/relook is not
    # represented by the old outer action history that triggered the intervention.
    ep['recovery_handled_state'] = [step, ep.get('todo_target_id'), bool(ep.get('closed'))]
    ep['recovery_history_mark'] = len(ep.get('hist', []))
