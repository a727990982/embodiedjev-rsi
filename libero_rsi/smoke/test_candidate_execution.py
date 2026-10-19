"""Candidate descriptions match program commands, not predicted physical success."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jev_observation as JO
import dsl_libero as dsl
from test_review_regressions import functions


def fixture():
    tcp=np.array([0.,0.,.08]);commands=[]
    ep={'closed':False,'approach_changes':0,'hist':[]}
    patch={'grasp_dz_objh':.1,'grasp_dxy_objw':[0.,0.],
           'yaw_deg':None,'tilt_deg':None,'tilt_dir_deg':None}
    per={'obj_h':.03,'obj_w':.02,'objc':np.array([.02,0,.02]),'yaw':0}
    ns={'tcp':tcp.copy(),'grasp':np.array([.0,0,.01]),'objc':per['objc'],
        'obj_h':.03,'obj_w':.02,'left_contact':False,'right_contact':False}
    def servo(target,*args,**kw):
        commands.append((np.array(target).tolist(),args,kw));tcp[:]=target;return 'reached'
    scope=functions('_candidate_execution','_clear_reapproach_delta','_approach_yaw_plan',
        '_relative_yaw_delta_deg','_cmd_yaw_deg','_motion_orientation',
        '_apply_stall_feedback','_apply_relative_approach_change','_nudge_grasp_xy','_raise_tcp',
        EP=ep,PER=per,PATCH=patch,np=np,dsl=dsl,LIB={'intents':[{'name':'lift'}]},SHAPE={'yaw':None,'tilt':0,'tdir':0},
        RSHAPE={'yaw':None,'tilt':None,'tdir':None},T=[50],CUR={},GOAL_TARGET={},
        eef=lambda:tcp.copy(),servo=servo,log=lambda **kw:None,
        _local_deadend_should_trigger=lambda **kw:(False,None),
        _note_grasp_progress=lambda ns:None,_note_patch_mutation=lambda *a:None,
        _record_local_stall_family=lambda *a:None,
        LOCAL_LOOP_FAMILY={'clear_reapproach'}, STALL_FB_ACTIONS={'clear_reapproach','change_approach_yaw'})
    menu={n:{'name':n,'definition':n,'fingers':'open','action':'move',
             'refpoint':'grasp + vec(0,0,0.05)' if n=='move_above_object' else 'grasp'}
          for n in ['move_above_object','descend_to_grasp','clear_reapproach','change_approach_yaw']}
    state={'observation_step':50,'motion_context':{'recovery_validation':
           {'preparations_without_descent':3,'no_progress':True,'descent_tested_since_preparation':False}}}
    return scope,ns,menu,state,commands


class CandidateTests(unittest.TestCase):
    def test_read_only_and_candidate_membership_order_unchanged(self):
        scope,ns,menu,state,calls=fixture()
        before=deepcopy((scope['EP'],scope['PATCH'],menu,state))
        expanded=scope['_candidate_execution'](menu,ns,state)
        criteria=JO.candidate_definitions(menu,expanded)
        self.assertEqual(list(criteria),list(menu))
        self.assertEqual(before,(scope['EP'],scope['PATCH'],menu,state))
        self.assertEqual(calls,[])
        for name in menu:
            self.assertIn('program_steps',expanded[name])
            self.assertTrue(expanded[name]['unverified_physics'])
            self.assertIn('no physics rollout',expanded[name]['source'])
            self.assertEqual(expanded[name]['observation_step'],50)
            self.assertEqual(json.loads(criteria[name].split('Current execution description: ')[1]),expanded[name])

    def test_moves_use_current_library_waypoint_and_commanded_orientation(self):
        scope,ns,menu,state,_=fixture()
        scope['PATCH']['yaw_deg']=35
        menu['move_above_object']['refpoint']='grasp + vec(0,0,0.1)'
        d=scope['_candidate_execution'](menu,ns,state)
        step=d['move_above_object']['program_steps'][1]
        np.testing.assert_allclose(step['commanded_tcp_target_m'],[0,0,.11])
        self.assertEqual(step['commanded_orientation_deg'][0],35)
        ns['grasp']=np.array([.02,0,.025])
        updated=scope['_candidate_execution'](menu,ns,state)
        np.testing.assert_allclose(updated['descend_to_grasp']['program_steps'][1]['commanded_tcp_target_m'],ns['grasp'])
        self.assertIn('separate choice',updated['descend_to_grasp']['conditional_effects'][0])

    def test_clear_description_matches_executor_offset_and_raise(self):
        for left,right in [(False,False),(True,False),(False,True)]:
            scope,ns,menu,state,calls=fixture()
            ns.update(left_contact=left,right_contact=right)
            d=scope['_candidate_execution'](menu,ns,state)['clear_reapproach']
            scope['_apply_stall_feedback'](ns,'test',action='clear_reapproach')
            np.testing.assert_allclose(calls[0][0],[0,0,.08+d['program_steps'][0]['requested_raise_m']])
            self.assertEqual(scope['PATCH']['grasp_dxy_objw'],d['program_steps'][1]['xy_object_width_units'])
            self.assertEqual(scope['PATCH']['grasp_dz_objh'],d['program_steps'][1]['z_object_height_units'])
            self.assertEqual(scope['EP']['force_next_intent'],'move_above_object')
            self.assertTrue(d['current_recovery_evidence']['no_progress'])

    def test_yaw_description_accounts_for_preincrement_clipping_and_contact(self):
        for current,count,left,right,expected in [(0,0,False,False,-45),(0,1,False,False,45),(90,0,True,False,45),(-90,0,False,True,-45)]:
            scope,ns,menu,state,calls=fixture()
            scope['PATCH']['yaw_deg']=current;scope['EP']['approach_changes']=count
            ns.update(left_contact=left,right_contact=right)
            d=scope['_candidate_execution'](menu,ns,state)['change_approach_yaw']
            self.assertEqual(d['program_steps'][0]['yaw_to_deg'],expected)
            scope['_apply_stall_feedback'](ns,'test',action='change_approach_yaw')
            self.assertEqual(scope['PATCH']['yaw_deg'],d['program_steps'][0]['yaw_to_deg'])
            self.assertEqual(scope['EP']['approach_changes'],count+1)
            self.assertEqual(len(calls),1)
            self.assertIn('no orientation-convergence',d['unverified_physics'][-1])

    def test_existing_deadend_guard_is_described_without_running_it(self):
        scope,ns,menu,state,calls=fixture()
        scope['_local_deadend_should_trigger']=lambda **kw:(True,'consec_local_loop')
        d=scope['_candidate_execution'](menu,ns,state)['clear_reapproach']
        self.assertEqual(d['program_steps'][0]['operation'],'existing_global_deadend_guard')
        self.assertEqual(calls,[])

    def test_closed_bilateral_contact_describes_existing_hold_guard(self):
        scope,ns,menu,state,calls=fixture()
        scope['EP']['closed']=True;ns['contact']=True
        d=scope['_candidate_execution'](menu,ns,state)['clear_reapproach']
        self.assertEqual(d['program_steps'][0]['queued_followup'],'lift')
        self.assertTrue(d['program_steps'][0]['local_preparation_skipped'])
        self.assertEqual(calls,[])

    def test_other_candidates_and_custom_actions_keep_their_description(self):
        scope,ns,menu,state,_=fixture()
        menu['custom']={'definition':'custom action'}
        menu['move_above_object']['action']='relook'
        d=scope['_candidate_execution'](menu,ns,state)
        base=JO.candidate_definitions(menu)
        result=JO.candidate_definitions(menu,d)
        self.assertEqual(result['custom'],base['custom'])
        self.assertEqual(result['move_above_object'],base['move_above_object'])


if __name__=='__main__':unittest.main()
