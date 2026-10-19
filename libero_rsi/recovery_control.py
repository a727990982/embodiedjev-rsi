"""Optional ownership of automatic stall rules; reuse the current recovery menu."""


def parse_mode(value):
    mode = (value or 'both').strip().lower()
    if mode not in ('both', 'jev'):
        raise ValueError('JEV_RECOVERY_MENU must be both (default) or jev')
    return mode


def skip_rule(rule, mode):
    """Never consume a skipped rule's use count, motion or forced next action."""
    return mode == 'jev' and bool(rule.get('stall_feedback') or rule.get('approach_change'))
