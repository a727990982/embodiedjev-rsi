"""Logic smoke: async slow-think job ownership (B01). No MuJoCo / Astra.

Abandoned worker A must not overwrite newer job B's parsed/rec.
"""
import ast
import threading
import time
from pathlib import Path

import harness_source


def main():
    src = harness_source.read()
    ast.parse(src)
    for needle in (
        '_next_slow_job_id', '_new_slow_result_box', '_abandon_async_slow',
        'result_box', "box.get('job_id') != job_id",
        'async_ownership_mismatch', 'replace_launch',
        # worker writes only to private box
        "result_box['parsed'], result_box['rec'] = parsed, rec",
        # sync abandons / invalidates
        "_abandon_async_slow(reason=f'sync:{crit}')",
    ):
        assert needle in src, f'missing: {needle}'

    # --- Pure ownership simulation (mirrors harness contract) ---
    seq = [0]
    current = {'job_id': None, 'result_box': None, 'applied': None}

    def next_id():
        seq[0] += 1
        return seq[0]

    def new_box(jid):
        return {'job_id': jid, 'parsed': None, 'rec': None, 'done': False}

    def abandon():
        current['job_id'] = None
        current['result_box'] = None

    def launch(label, delay, store):
        jid = next_id()
        box = new_box(jid)
        current['job_id'] = jid
        current['result_box'] = box
        store['jid'] = jid
        store['box'] = box

        def worker():
            time.sleep(delay)
            box['parsed'] = {'issue': label, 'job': jid}
            box['rec'] = {'cost_usd': 0.01, 'label': label}
            box['done'] = True
        threading.Thread(target=worker, daemon=True).start()
        return jid, box

    def poll_apply():
        jid = current['job_id']
        box = current['result_box']
        if jid is None or box is None:
            return None
        if box.get('job_id') != jid:
            return 'ownership_mismatch'
        if not box.get('done'):
            return None
        # ownership still current?
        if current['job_id'] != jid or current['result_box'] is not box:
            return 'abandoned_race'
        current['applied'] = box['parsed']
        current['job_id'] = None
        current['result_box'] = None
        return 'applied'

    # 1) A delayed, replace with B, A finishes then B — only B applies
    a = {}; b = {}
    launch('A', 0.15, a)
    abandon()  # sync/replace invalidates A
    launch('B', 0.05, b)
    time.sleep(0.08)  # B done, A maybe not
    r1 = poll_apply()
    assert r1 == 'applied', r1
    assert current['applied']['issue'] == 'B', current['applied']
    time.sleep(0.12)  # A finishes into orphaned box
    assert a['box']['done'] is True and a['box']['parsed']['issue'] == 'A'
    # A cannot be applied: no current ownership pointing at A's box
    r2 = poll_apply()
    assert r2 is None, r2
    assert current['applied']['issue'] == 'B'

    # 2) Direct overwrite attempt on shared slot pattern must be impossible with private boxes
    # Old bug: worker wrote into shared SLOW_JOB dict. Private boxes stay isolated.
    assert a['box'] is not b['box']
    assert a['box']['parsed']['issue'] == 'A'
    assert b['box']['parsed']['issue'] == 'B'

    # 3) Mismatch detection
    current['job_id'] = 99
    current['result_box'] = b['box']  # box still has B's id
    assert poll_apply() == 'ownership_mismatch'

    print('test_async_slow_ownership: OK')


if __name__ == '__main__':
    main()
