"""Depth unit/alignment, regional evidence and routing checks without model calls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import depth_observation as DO
import robot_geometry as RG
from test_review_regressions import functions


class DepthTests(unittest.TestCase):
    def test_depth_conversion_and_flip_match_rgb_without_stepping(self):
        raw = np.array([[.2, 1.], [.4, np.nan]])[..., None]
        cu = SimpleNamespace(get_real_depth_map=lambda sim, value: value*2)
        with patch.dict(sys.modules, {'robosuite.utils': SimpleNamespace(camera_utils=cu)}):
            scope = functions('depth_views', np=np, sim=object(), obs={
                'agentview_depth': raw, 'robot0_eye_in_hand_depth': np.full((2,2),.3)})
            result = scope['depth_views']()
        np.testing.assert_allclose(result['front'], [[.8,np.nan],[.4,np.nan]], equal_nan=True)
        np.testing.assert_allclose(result['wrist'], .6)
        raw[:] = 0
        self.assertAlmostEqual(result['front'][0,0], .8)  # Detached snapshot.

    def test_full_region_obstacle_and_invalid_pixels_are_preserved(self):
        rgb = np.full((128,256,3), 60, dtype=np.uint8)
        depth = np.ones((128,256))
        depth[32:48,64:96] = .25  # Obstacle region away from the centre/keypoint.
        depth[:16,:32] = np.nan
        im, info = DO.panel(rgb, depth, view='front', step=9)
        self.assertEqual(im.size, (512,176))
        np.testing.assert_array_equal(np.asarray(im)[:128,:256], rgb)
        cells = {t['cell']:t for t in info['tiles']}
        self.assertEqual(cells['r2c2']['depth_m_min_p10_median'], [.25,.25,.25])
        self.assertEqual(cells['r0c0']['valid_fraction'], 0)
        self.assertIsNone(cells['r0c0']['depth_m_min_p10_median'])
        self.assertEqual(len(info['tiles']),64)
        json.dumps(info, allow_nan=False)

    def test_missing_saturated_and_bad_alignment_are_explicit(self):
        rgb = np.zeros((80,80,3), dtype=np.uint8)
        _, missing = DO.panel(rgb, None, view='wrist', step=0)
        self.assertEqual(missing['status'], 'unavailable')
        _, far = DO.panel(rgb, np.full((80,80),3.), view='front', step=0)
        self.assertEqual(far['saturated_far_fraction'], 1)
        self.assertEqual(far['tiles'][0]['depth_m_min_p10_median'], [3.,3.,3.])
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            DO.panel(rgb, np.ones((40,80)), view='front', step=0)

    def test_depth_panel_exists_even_when_robot_geometry_is_unavailable(self):
        rgb = np.zeros((80,80,3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            scope = functions('astra_geometry', RG=RG, DO=DO, np=np, Image=Image,
                m=SimpleNamespace(), sim=SimpleNamespace(data=None),
                obs={'agentview_image':rgb,'robot0_eef_quat':[0,0,0,1]},
                EP={}, ep_dir=temp, T=[3], eef=lambda:np.zeros(3),
                depth_views=lambda:{'front':np.full((80,80),.7)}, log=lambda **kw:None)
            geom, images = scope['astra_geometry']('test')
            self.assertEqual(geom['status'],'unavailable')
            self.assertEqual(geom['depth_views']['front']['status'],'available')
            self.assertEqual(len(images),1)
            self.assertTrue(Path(images[0]).is_file())
            path=geom['depth_views']['front']['metric_depth_path']
            np.testing.assert_allclose(np.load(path,allow_pickle=False),.7)

    def test_selected_route_survives_advice_without_executable_commands(self):
        raw={'evidence':'front r2c2 obstacle at .25 m',
             'path_candidates':[{'route':'direct','assessment':'blocked','evidence':'palm corridor blocked'},
                                {'route':'raised','assessment':'preferred','evidence':'front upper tiles visible'}],
             'selected_path':'raised','plan':[{'step':'teleport'}]}
        normalized=RG.advice(raw)
        self.assertEqual(normalized['selected_path'],'raised')
        self.assertEqual(len(normalized['path_candidates']),2)
        self.assertNotIn('plan',normalized)
        raw['selected_path']='direct'
        self.assertEqual(RG.advice(raw)['selected_path'],'reobserve')


if __name__ == '__main__': unittest.main()
