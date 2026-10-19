"""Regressions for recorded empty-close, premature release and blind-stall loops."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import motion_feedback as MF
import coding_agent_slow as CAS
from test_review_regressions import functions
from test_transport_feedback import fixture, LIB


class ActionTests(unittest.TestCase):
    def test_close_above_or_beside_target_is_excluded_and_forced_close_does_not_move(self):
        for point in ([0., 0., .6], [.05, 0., .5]):
            scope, ep, _, commands = fixture()
            scope['eef'] = lambda: np.array(point)
            ns = scope['namespace']()[0]
            self.assertNotIn('close_gripper', scope['_acting_menu']({x['name']:x for x in LIB['intents']}))
            self.assertEqual(scope['execute']('close_gripper', ns), 'close_not_executed:not_at_grasp')
            self.assertEqual(commands, [])
            self.assertEqual(ep['n_close'], 0)

    def test_recorded_failed_and_successful_close_positions(self):
        # PR10 fixed-code eval, init3–8: six first empty closes, four confirmed closes.
        # Rounded proprio/visual waypoint facts from the saved decisions; no sim truth.
        empty = [([.066,-.101,.066],[.063,-.101,.004],.03),
                 ([.054,-.097,.067],[.049,-.097,.004],.03),
                 ([.064,-.103,.055],[.060,-.103,.004],.025),
                 ([.059,-.099,.064],[.054,-.098,.001],.03),
                 ([.068,-.098,.067],[.063,-.098,.004],.03),
                 ([.053,-.097,.056],[.049,-.097,.004],.025)]
        secured = [([.048,-.097,.013],[.049,-.097,.009],.03),
                   ([.064,-.091,.010],[.065,-.098,.001],.03),
                   ([.070,-.101,.020],[.070,-.099,.016],.03),
                   ([.052,-.082,.019],[.053,-.082,.015],.025)]
        for cases, expected in [(empty,False),(secured,True)]:
            for tcp, grasp, height in cases:
                self.assertEqual(MF.grasp_position(tcp,grasp,.02,height)['ready'],expected)

    def test_close_at_position_needs_no_specific_preceding_intent(self):
        scope, ep, _, commands = fixture()
        scope['_confirm_grasp_after_close'] = lambda:(True,{})
        ep['prev'] = 'clear_reapproach'
        self.assertEqual(scope['execute']('close_gripper', scope['namespace']()[0]), 'closed')
        self.assertEqual(ep['n_close'], 1)
        self.assertEqual(commands[0][0], 'grip')

    def test_new_localization_before_close_is_rechecked(self):
        scope, ep, _, commands = fixture()
        old = dict(scope['namespace']()[0])
        current = dict(old, grasp=np.array([0.,0.,.2]))
        scope['namespace'] = lambda:(current,{})
        self.assertEqual(scope['execute']('close_gripper',old), 'close_not_executed:not_at_grasp')
        self.assertEqual(commands, [])

    def test_confirmed_target_cannot_be_released_by_opening_approach_or_retry(self):
        for action in ('reopen_and_retry','move_above_object','descend_to_grasp','retreat','open_gripper'):
            scope, ep, sensors, commands = fixture()
            ep.update(closed=True,grasp_confirm=True)
            sensors.update(contacts=(True,True),opening=.0555)
            ns = scope['namespace']()[0]
            self.assertNotIn(action,scope['_acting_menu']({x['name']:x for x in LIB['intents']}))
            self.assertIn('not_executed',scope['execute'](action,ns))
            self.assertEqual(commands,[])
            self.assertTrue(ep['closed'])

    def test_release_at_visual_destination_is_allowed_and_wrong_grasp_can_reopen(self):
        scope, ep, sensors, commands = fixture()
        ep.update(closed=True,grasp_confirm=True)
        sensors.update(contacts=(True,True),opening=.0555)
        ns=scope['namespace']()[0];ns.update(place=np.array([0.,0.,.45]),clear=.05)
        self.assertIn(scope['execute']('open_gripper',ns),('reached','at_target'))
        self.assertFalse(ep['closed'])
        ep.update(closed=True,grasp_confirm=True);sensors['identity']=False
        self.assertEqual(scope['execute']('reopen_and_retry',ns),'reopened')

    def test_unchanged_failed_descent_requires_geometry_or_approach_change(self):
        scope, ep, _, commands = fixture()
        ns=scope['namespace']()[0]
        ep['blocked_motions']={'descend_to_grasp':{'target_xyz_m':ns['grasp'].tolist(),'orientation_deg':[0,0,0]}}
        scope['_motion_orientation']=lambda shape=None:[0,0,0]
        self.assertEqual(scope['execute']('descend_to_grasp',ns),'motion_not_executed:unchanged_failed_waypoint')
        self.assertEqual(commands,[])
        scope['_motion_orientation']=lambda shape=None:[25,0,0]
        self.assertIn('descend_to_grasp',scope['_acting_menu']({x['name']:x for x in LIB['intents']}))

    def test_exhausted_help_and_relook_are_not_repeated_as_actions(self):
        scope, _, _, _=fixture();scope['P'].update(slow=4,relook=4)
        intents={name:{'name':name} for name in ('ask_astra','consult_astra','relook','finish')}
        self.assertEqual(set(scope['_acting_menu'](intents)),{'finish'})


class MotionTests(unittest.TestCase):
    def test_contacts_include_palm_arm_fixture_and_self_without_object_identity(self):
        parts={1:'left_finger',2:'palm_or_wrist',3:'arm',4:'right_finger'}
        result=MF.contact_summary([(1,0),(2,99),(3,2)],parts)
        self.assertEqual(result['environment_contact_parts'],['left_finger','palm_or_wrist'])
        self.assertEqual(result['self_contact_parts'],['arm','palm_or_wrist'])
        self.assertNotIn('99',json.dumps(result))

    def test_stalled_servo_records_requested_and_actual_motion_and_before_image(self):
        class Rotation:
            @classmethod
            def from_rotvec(cls,*a):return cls()
            @classmethod
            def from_euler(cls,*a):return cls()
            @classmethod
            def from_quat(cls,*a):return cls()
            def inv(self):return self
            def as_rotvec(self):return np.zeros(3)
            def __mul__(self,other):return self
        ep={};t=[0];events=[]
        contacts={'environment_contact_parts':['arm'],'self_contact_parts':[]}
        with tempfile.TemporaryDirectory() as temp:
            scope=functions('servo',MF=MF,np=np,Rot=Rotation,Q0=Rotation(),EP=ep,T=t,DONE=[False],
                HORIZON=600,PER={'yaw':0},PATCH={},RSHAPE={},SHAPE={},Image=Image,ep_dir=temp,
                obs={'robot0_eef_quat':[0,0,0,1]},eef=lambda:np.array([0.,0.,.1]),
                robot_contacts=lambda:contacts,_combo_rgb=lambda:(np.zeros((3,6,3),dtype='uint8'),None,None),
                log=lambda **kw:events.append(kw),step=lambda a:t.__setitem__(0,t[0]+1))
            # The real servo indexes the known runtime shape keys.
            scope['PATCH'].update(yaw_deg=None,tilt_deg=None,tilt_dir_deg=None)
            scope['RSHAPE'].update(yaw=None,tilt=None,tdir=None)
            scope['SHAPE'].update(yaw=None,tilt=None,tdir=None)
            self.assertEqual(scope['servo'](np.array([0.,0.,0.]),-1),'stalled')
            report=ep['last_motion']
            self.assertEqual(report['actual_delta_m'],[0,0,0])
            self.assertEqual(report['requested_delta_m'],[0,0,-.1])
            self.assertEqual(report['environment_contact_parts_during_motion'],['arm'])
            self.assertEqual(t[0],8)
            self.assertTrue(Path(ep['motion_before_path']).is_file())
            self.assertTrue(events)

    def test_projection_matches_existing_backprojection_convention(self):
        cp=[0,0,1];rot=np.eye(3).tolist()
        self.assertEqual(MF.project([0,0,0],cp,rot,256,512),[256,256])
        self.assertEqual(MF.project([.1,.2,0],cp,rot,256,512),[281.6,204.8])
        self.assertIsNone(MF.project([0,0,2],cp,rot,256,512))

    def test_failed_motion_before_image_reaches_jev_as_explicit_history(self):
        scope,ep,_,_=fixture()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'before.png';Image.new('RGB',(2,2)).save(p)
            ep.update(failed_motion_before_path=str(p),
                      last_failed_motion={'result':'stalled','start_step':12},
                      last_motion={'result':'reached','start_step':25})
            import jev_observation as JO
            scope['_current_observation']=lambda:({key:0 for key in JO.FIELDS},{'front':np.zeros((2,2,3),dtype='uint8')})
            scope['ep_dir']=temp
            state,assets=scope['_jev_input'](include_execution_history=True)
            self.assertEqual(state['motion_before_image']['step'],12)
            self.assertIn(state['motion_before_image']['asset'],assets)
            self.assertEqual(set(state['cameras']),{'front'})

    def test_reserved_motion_diagnosis_is_spent_once_even_on_network_error(self):
        ep={};p={'slow':4}
        scope=functions('_sync_slow','_hard_astra_trigger',EP=ep,P=p,MAX_SLOW=4,FALLBACK=True,
            _recovery_context=lambda:{'no_progress':False},
            ASK_ASTRA_NAMES=set(),FAIL_RESULTS={'stalled'},SLOW_JOB={'thread':None,'job_id':None},
            _drain_orphan_slow_costs=lambda:None,_combo_rgb=lambda:(None,None,None),
            _slow_ctx=lambda *a:'{}',_prepare_slow_snapshot=lambda *a:{'coding_context':{}},
            _next_slow_job_id=lambda:1,log=lambda **kw:None,T=[10])
        self.assertEqual(scope['_hard_astra_trigger']('descend_to_grasp','stalled'),'motion_obstruction')
        with patch.object(CAS,'call_monitor',side_effect=RuntimeError('offline')) as call:
            self.assertIn('sync_error',scope['_sync_slow']('descend_to_grasp','stalled','motion_obstruction')[0])
            self.assertEqual(scope['_sync_slow']('descend_to_grasp','stalled','motion_obstruction')[0],'slow_cap')
        self.assertEqual(call.call_count,1)
        self.assertIsNone(scope['_hard_astra_trigger']('descend_to_grasp','stalled'))

    def test_reserved_diagnosis_can_apply_after_general_cap_without_extending_it(self):
        calls=[];p={'slow':4,'planning_calls':4}
        with tempfile.TemporaryDirectory() as temp:
            scope=functions('_finish_slow',AUTO_RECOVERY_RULES=True,P=p,MAX_SLOW=4,EP={'closed':False},
                _account_slow_rec=lambda rec:None,identity_fields=lambda parsed:{'matched_language_goal':True},
                FB_LESSONS=[],ID_EVENTS=[],Image=Image,ep_dir=temp,dec=8,T=[100],PATCH={},CUR={},
                _apply_episode_plan=lambda *a,**kw:calls.append(a) or ['clear_reapproach'])
            reply={'issue':'blocked_by_neighbour','intervene':True,'is_goal_object':True,
                   'matched_language_goal':True,'plan':[{'step':'clear_reapproach'}]}
            args=(reply,{},None,np.zeros((2,2,3),dtype='uint8'),'descend_to_grasp','stalled','sync_codex')
            self.assertEqual(scope['_finish_slow'](*args,'motion_obstruction',8)[0],'slow_cap_drop')
            self.assertEqual(calls,[])
            self.assertEqual(scope['_finish_slow'](*args,'reserved_motion_obstruction',8)[0],'intervened')
            self.assertEqual(len(calls),1)
            self.assertEqual(p['slow'],4)

    def test_start_clearance_is_structured_and_unknown_is_not_clear(self):
        raw={'issue':'ok','intervene':False,'plan':[]}
        ok,_,out=CAS.validate_start_plan(raw)
        self.assertTrue(ok)
        self.assertEqual(out['obstacle_assessment']['status'],'uncertain')
        raw['obstacle_assessment']=dict(out['obstacle_assessment'],status='guaranteed_safe')
        self.assertFalse(CAS.validate_start_plan(raw)[0])


if __name__=='__main__':unittest.main()
