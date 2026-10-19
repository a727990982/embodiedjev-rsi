"""Real quaternion math and deterministic sensor doubles; no model or simulator calls."""
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import motion_feedback as MF
from test_review_regressions import functions
from test_candidate_execution import fixture as recovery_fixture
from test_transport_feedback import fixture as executor_fixture
from test_pose_menu import POSE_FUNCS


class OrientationTests(unittest.TestCase):
    def scope(self, *, yaw=45., actual_yaw=0., spin=0., horizon=1000):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        pos=np.array([0.,0.,.2]);t=[0];events=[]
        obs={'robot0_eef_quat':Rot.from_euler('z',actual_yaw,degrees=True).as_quat()}
        world={'yaw':actual_yaw}
        def step(action):
            t[0]+=1
            world['yaw']+=spin
            obs['robot0_eef_quat']=Rot.from_euler('z',world['yaw'],degrees=True).as_quat()
        scope=functions('servo','commanded_rotation','orientation_feedback',*POSE_FUNCS,
            np=np,Rot=Rot,MF=MF,Image=Image,Q0=Rot.identity(),T=t,DONE=[False],HORIZON=horizon,
            eef=lambda:pos.copy(),step=step,obs=obs,EP={'closed':False},ep_dir=temp.name,
            PATCH={'yaw_deg':yaw,'tilt_deg':0.,'tilt_dir_deg':0.},
            RSHAPE={'yaw':None,'tilt':None,'tdir':None},SHAPE={'yaw':None,'tilt':0.,'tdir':0.},
            PER={'yaw':yaw},_cmd_yaw_deg=lambda:yaw,
            robot_contacts=lambda:{'environment_contact_parts':[],'self_contact_parts':[]},
            _combo_rgb=lambda:(np.zeros((3,6,3),dtype='uint8'),None,None),
            log=lambda **kw:events.append(kw))
        return scope,pos,events

    def test_locked_rotation_stalls_even_when_position_is_reached(self):
        s,pos,events=self.scope()
        self.assertEqual(s['servo'](pos,-1,ori_tol=.08),'stalled')
        report=s['EP']['last_motion'];o=report['orientation']
        self.assertTrue(report['position_reached']);self.assertFalse(o['within_tolerance'])
        self.assertTrue(o['required_for_reached']);self.assertAlmostEqual(o['remaining_angle_deg'],45)
        self.assertAlmostEqual(o['actual_rotation_angle_deg'],0)
        self.assertAlmostEqual(o['requested_rotation_angle_deg'],45)
        self.assertEqual(report,s['EP']['last_orientation_motion'])
        self.assertEqual(report,s['EP']['last_failed_motion'])
        self.assertTrue(Path(s['EP']['motion_before_path']).exists())
        self.assertEqual(events[-1]['motion_feedback'],report)

    def test_rotation_only_progress_does_not_false_stall_and_both_conditions_finish(self):
        s,pos,_=self.scope(spin=1.)
        self.assertEqual(s['servo'](pos,-1,ori_tol=.08,max_steps=80),'reached')
        report=s['EP']['last_motion'];o=report['orientation']
        self.assertGreater(s['T'][0],8)
        self.assertTrue(report['position_reached']);self.assertTrue(o['within_tolerance'])
        self.assertLess(o['remaining_angle_rad'],.08)
        self.assertAlmostEqual(o['actual_rotation_angle_deg'],s['T'][0])

    def test_position_only_completion_does_not_claim_orientation_completion(self):
        s,pos,_=self.scope()
        self.assertEqual(s['servo'](pos,-1),'reached')
        o=s['EP']['last_motion']['orientation']
        self.assertFalse(o['required_for_reached']);self.assertFalse(o['within_tolerance'])
        self.assertNotIn('last_orientation_motion',s['EP'])

    def test_orientation_alone_does_not_complete_a_pose_motion(self):
        s,pos,_=self.scope(yaw=0.)
        self.assertEqual(s['servo'](pos+np.array([0,0,.1]),-1,ori_tol=.08),'stalled')
        report=s['EP']['last_motion']
        self.assertFalse(report['position_reached'])
        self.assertTrue(report['orientation']['within_tolerance'])

    def test_timeout_and_horizon_report_residual_without_claiming_reached(self):
        s,pos,_=self.scope(spin=1.)
        self.assertEqual(s['servo'](pos,-1,ori_tol=.08,max_steps=2),'timeout')
        self.assertAlmostEqual(s['EP']['last_motion']['orientation']['remaining_angle_deg'],43)
        s,pos,_=self.scope(horizon=0)
        self.assertEqual(s['servo'](pos,-1,ori_tol=.08),'horizon_or_done')
        self.assertFalse(s['EP']['last_motion']['orientation']['within_tolerance'])
        self.assertEqual(s['T'][0],0)

    def test_shortest_rotation_handles_wrap_quaternion_sign_and_pitch(self):
        s,_,_=self.scope(yaw=-179.,actual_yaw=179.)
        self.assertAlmostEqual(s['orientation_feedback']()['remaining_angle_deg'],2)
        s['obs']['robot0_eef_quat']=-s['obs']['robot0_eef_quat']
        self.assertAlmostEqual(s['orientation_feedback']()['remaining_angle_deg'],2)
        goal=Rot.from_euler('xyz',[25,35,60],degrees=True)
        s['obs']['robot0_eef_quat']=goal.as_quat()
        self.assertAlmostEqual(s['orientation_feedback'](goal)['remaining_angle_deg'],0)
        s['obs']['robot0_eef_quat']=Rot.from_euler('xyz',[0,35,60],degrees=True).as_quat()
        self.assertAlmostEqual(s['orientation_feedback'](goal)['remaining_angle_deg'],25)

    def test_current_measurement_updates_without_rewriting_last_rotation_attempt(self):
        s,pos,_=self.scope()
        s['servo'](pos,-1,ori_tol=.08)
        old=deepcopy(s['EP']['last_orientation_motion'])
        s['obs']['robot0_eef_quat']=Rot.from_euler('z',45,degrees=True).as_quat()
        self.assertTrue(s['orientation_feedback']()['within_tolerance'])
        s['servo'](pos,-1)
        self.assertEqual(s['EP']['last_orientation_motion'],old)
        self.assertFalse(old['orientation']['within_tolerance'])

    def test_equal_commanded_astra_yaw_still_verifies_actual_rotation(self):
        s,pos,_=self.scope()
        it=next(i for i in s['_pose_menu_intent_specs']() if i['name']=='rotate_yaw_toward_astra')
        self.assertEqual(s['_execute_pose_menu'](it,{'tcp':pos}),'stalled')
        self.assertAlmostEqual(s['EP']['last_motion']['orientation']['remaining_angle_deg'],45)

    def test_failed_yaw_recovery_does_not_queue_successful_approach(self):
        scope,ns,_,_,calls=recovery_fixture()
        def locked(target,*a,**kw):
            calls.append(kw)
            scope['EP']['last_motion']={'result':'stalled','orientation':{'within_tolerance':False}}
            return 'stalled'
        scope['servo']=locked
        report=scope['_apply_stall_feedback'](ns,'test',action='change_approach_yaw')
        self.assertEqual(calls[0]['ori_tol'],.08)
        self.assertEqual(report['motion_result'],'stalled')
        self.assertFalse(report['orientation']['within_tolerance'])
        self.assertIsNone(scope['EP'].get('force_next_intent'))

    def test_executor_retains_failed_rotation_outcome_if_later_feedback_moves(self):
        s,ep,_,_=executor_fixture()
        s['LIB']=dict(s['LIB'],intents=list(s['LIB']['intents'])+[
            {'name':'change_approach_yaw','action':'move','fingers':'keep'}])
        ep['last_motion']={'result':'reached','start_step':100}
        s['_apply_stall_feedback']=lambda *a,**kw:{'action':'change_approach_yaw','motion_result':'stalled'}
        self.assertEqual(s['execute']('change_approach_yaw',s['namespace']()[0]),'stalled')


if __name__=='__main__':unittest.main()
