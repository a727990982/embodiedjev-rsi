"""Exercise actual perception routing with a fake Astra transport."""
import time
from types import SimpleNamespace
from test_review_regressions import functions


def main():
    calls = []
    def call(image, prompt, meta, effort):
        calls.append((image, prompt, meta, effort))
        return {'grasp_px': [10, 20], 'matched_language_goal': True}, {'cost_usd': .02}
    scope = functions('perceive', rgbd=lambda: ('rgb', 'depth'), PERC_PROMPT='visual prompt',
        META={'bench': 'libero'}, ASTRA_EFFORT='xhigh', time=time,
        P={'astra_cost': 0, 'astra_calls': 0, 'last_astra_wall': 0},
        AC=SimpleNamespace(call=call, to_jpeg=lambda rgb: b'image'))
    parsed, rgb, depth = scope['perceive']('verify current target', phase='relook')
    assert parsed['matched_language_goal'] is True
    assert (rgb, depth) == ('rgb', 'depth')
    assert calls[0][2]['phase'] == 'relook'
    assert 'verify current target' in calls[0][1]
    assert scope['P']['astra_calls'] == 1
    assert scope['P']['astra_cost'] == .02
    print('PASS Astra perception routing')


if __name__ == '__main__':
    main()
