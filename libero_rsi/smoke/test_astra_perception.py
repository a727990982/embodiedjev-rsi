"""Exercise Astra perception/identity through the real Codex transport with a fake CLI process."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from test_review_regressions import functions
import coding_agent_slow as CAS
import robot_geometry as RG


class CodexVisionTests(unittest.TestCase):
    def test_harness_perception_and_identity_use_astra_via_codex(self):
        for phase in ('perceive_start', 'relook', 'escape_look'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp:
                raw = {'grasp_px': [200, 300], 'object_center_px': [200, 300],
                       'place_px': [400, 240], 'matched_language_goal': True}
                replies = [raw, {'is_language_goal_object': True, 'reason': 'marked goal'}]
                calls, logs = [], []
                def run(cmd, **kwargs):
                    calls.append((cmd, kwargs))
                    for i, token in enumerate(cmd):
                        if token == '-i':
                            self.assertTrue(Path(cmd[i + 1]).is_file())
                    return SimpleNamespace(returncode=0, stdout=json.dumps(replies.pop(0)), stderr='')
                rgb = np.zeros((512, 512, 3), dtype=np.uint8)
                dep = np.ones((512, 512))
                scope = functions('perceive', 'verify_grasp_identity',
                    RG=RG, astra_geometry=lambda *a, **kw: ({'status': 'unavailable'}, []),
                    remember_geometry_advice=lambda *a: None,
                    rgbd=lambda: (rgb, dep), np=np, Image=Image, time=time,
                    obs={'robot0_eye_in_hand_image': rgb.copy()}, ep_dir=temp,
                    LANG='put butter in basket', _active_language_goal=lambda: 'put butter in basket', S=512, T=[3], ASTRA_EFFORT='xhigh',
                    P={'astra_cost': 0, 'astra_calls': 0, 'last_astra_wall': 0},
                    log=lambda **kw: logs.append(kw))
                with patch.object(CAS, '_resolve_codex_bin', return_value='/fake/codex'), \
                     patch.object(CAS, '_proxy_env', return_value={}), \
                     patch.object(CAS, 'CODEX_MODEL', 'planning-model-override'), \
                     patch.object(CAS.subprocess, 'run', side_effect=run):
                    parsed, actual_rgb, actual_dep = scope['perceive']('current target', phase=phase)
                    verified, marked = scope['verify_grasp_identity'](parsed, actual_rgb)
                self.assertTrue(parsed['matched_language_goal'])
                self.assertTrue(verified['is_language_goal_object'])
                self.assertIs(actual_dep, dep)
                self.assertTrue(np.any(marked != rgb))
                self.assertEqual(scope['P']['astra_calls'], 2)
                self.assertEqual([x['backend'] for x in logs], ['codex_perceive', 'codex_identity'])
                for cmd, kwargs in calls:
                    self.assertEqual(cmd[:2], ['/fake/codex', 'exec'])
                    self.assertEqual(cmd[cmd.index('-m') + 1], 'gpt-6-astra')
                    self.assertIn('model_reasoning_effort="xhigh"', cmd)
                    self.assertEqual(cmd[-1], '-')
                    self.assertIn('put butter in basket', kwargs['input'])
                self.assertEqual(calls[0][0].count('-i'), 2)
                self.assertIn('RED circle+cross', calls[1][1]['input'])

    def test_keypoints_validated_and_uncertain_identity_preserved(self):
        raw = {'grasp_px': [200, 300], 'object_center_px': [200, 300],
               'place_px': [400, 240], 'matched_language_goal': None}
        with patch.object(CAS, 'call_codex', return_value=(raw, {})):
            parsed, rec = CAS.call_perceive({'img_size': 512}, mode='codex')
        self.assertIsNone(parsed['matched_language_goal'])
        self.assertEqual(rec['backend'], 'codex_perceive')
        raw['grasp_px'] = [-1, 5000]
        with patch.object(CAS, 'call_codex', return_value=(raw, {})):
            with self.assertRaisesRegex(ValueError, 'codex_perceive_invalid'):
                CAS.call_perceive({'img_size': 512}, mode='codex')

    def test_cli_failure_is_not_replaced_by_mock_or_direct_client(self):
        with patch.object(CAS, 'call_codex', side_effect=RuntimeError('CLI unavailable')) as call:
            with self.assertRaisesRegex(RuntimeError, 'CLI unavailable'):
                CAS.harness_call_perceive(lang='goal', phase='relook', image_paths=['frame.jpg'])
            with self.assertRaisesRegex(RuntimeError, 'CLI unavailable'):
                CAS.call_identity_check('check mark', ['marked.jpg'])
        self.assertEqual(call.call_count, 2)

    def test_dual_is_still_rejected_for_perception(self):
        with patch.object(CAS, 'call_codex') as call:
            with self.assertRaisesRegex(ValueError, 'unsupported_planning_backend'):
                CAS.call_perceive({}, mode='dual')
        call.assert_not_called()

    def test_planning_keeps_its_configured_model(self):
        with patch.object(CAS, '_resolve_codex_bin', return_value='/fake/codex'), \
             patch.object(CAS, '_proxy_env', return_value={}), \
             patch.object(CAS, 'CODEX_MODEL', 'planning-model-override'), \
             patch.object(CAS.subprocess, 'run', return_value=SimpleNamespace(
                 returncode=0, stdout='{"issue":"ok","intervene":false,"plan":[]}', stderr='')) as run:
            _, rec = CAS.call_monitor({}, mode='codex')
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index('-m') + 1], 'planning-model-override')
        self.assertEqual(rec['model'], 'planning-model-override')


if __name__ == '__main__':
    unittest.main()
