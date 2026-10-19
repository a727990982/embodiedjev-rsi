"""Encode and decode actual production video frames; no simulator or model calls."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
from test_review_regressions import functions


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for recording smoke')
class VideoTests(unittest.TestCase):
    def test_front_wrist_frames_encode_to_decodable_mp4(self):
        with tempfile.TemporaryDirectory() as temp:
            front=np.full((512,512,3),[180,20,20],dtype=np.uint8)
            wrist=np.full((512,512,3),[20,180,20],dtype=np.uint8)
            step=[12]
            scope=functions('_start_video','vid_frame',_I=Image,_D=ImageDraw,
                obs={'agentview_image':front,'robot0_eye_in_hand_image':wrist,
                     'robot0_gripper_qpos':[0.02,-0.02]},
                T=step,HORIZON=1200,DONE=[False],
                CUR={'intent':'carry_to_place','fb_t':-999,'slow_t':-999})
            path=Path(temp)/'episode.mp4'
            writer=scope['_start_video'](str(path));scope['VID']=writer
            try:
                for t in (12,13):
                    step[0]=t;scope['vid_frame']([0,0,0,0,0,0,1])
            finally:
                writer.stdin.close()
                self.assertEqual(writer.wait(timeout=20),0)
            decoded=subprocess.run(['ffmpeg','-v','error','-i',str(path),
                '-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True,timeout=20).stdout
            frames=np.frombuffer(decoded,dtype=np.uint8).reshape(2,572,1024,3)
            np.testing.assert_allclose(frames[0,200,200],[180,20,20],atol=5)
            np.testing.assert_allclose(frames[0,200,700],[20,180,20],atol=5)
            self.assertTrue(np.any(frames[0,516:] != 0), 'Step/action caption must be visible')


if __name__=='__main__':unittest.main()
