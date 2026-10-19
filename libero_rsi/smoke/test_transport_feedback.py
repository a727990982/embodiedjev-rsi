"""Exercise real candidate/execute/history functions without simulator truth."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dsl_libero as dsl
import jev_observation as JO
import motion_feedback as MF
from test_review_regressions import functions

TRANSPORT = {'lift', 'carry_to_place', 'lower_to_place'}
FAILED = 'grasp_failed:transport_not_executed'
LIB = json.loads((ROOT/'libs/ee_pg_easy_v2_lib.json').read_text())


def fixture():
    ep = {'closed':False, 'grasp_confirm':False, 'grasp_secured_once':False,
          'n_close':0, 'closes':[], 'hist':[], 'result_hist':[], 'vision_serial':0,
          'task_todo':{'items':[], 'active_id':None}}
    sensor = {'contacts':(False,False), 'opening':0.002, 'identity':True}
    commands = []
    per = {'seen_held':False, 'obj_h':0.05, 'yaw':0}
    ns = {'grasp':np.array([0.,0.,0.5]), 'tcp':np.array([0.,0.,0.5]), 'obj_w':.03, 'obj_h':.05, 'place':np.array([1.,1.,.3]), 'clear':.05}
    scope = functions('_transport_grasp_ready', '_action_block_reason', '_acting_menu', 'execute',
        '_confirm_grasp_after_close', '_jev_input', EP=ep, PER=per, LIB=LIB,
        TRANSPORT_INTENTS=TRANSPORT, GRASP_FAILED=FAILED, OPEN_MAX=0.08, MF=MF,
        CLOSE_NOT_READY='close_not_executed:not_at_grasp', MOTION_RECOVERIES={'back_off','clear_reapproach','change_approach_yaw'},
        namespace=lambda:(ns,{}), FALLBACK=True, P={'slow':0,'relook':0}, MAX_SLOW=4, MAX_RELOOK=4,
        _motion_context=lambda:{}, _startup_obstacle_advice=lambda:{},
        ASK_ASTRA_NAMES={'ask_astra','consult_astra'}, T=[100], DONE=[False],
        contacts=lambda:sensor['contacts'], opening=lambda:sensor['opening'],
        goal_id_confirmed=lambda:sensor['identity'], np=np, dsl=dsl, JO=JO,
        Image=Image, LANG='put butter in basket', log=lambda **kw:None,
        eef=lambda:np.array([0.,0.,0.5]),
        grip=lambda *args:commands.append(('grip',args)),
        servo=lambda *args,**kwargs:commands.append(('servo',args)) or 'reached',
        _clear_reapproach_if_aligned=lambda reason:None, SHAPE={})
    return scope,ep,sensor,commands


class TransportTests(unittest.TestCase):
    def test_empty_or_stale_grasp_never_offers_transport(self):
        scope,ep,sensor,_=fixture()
        intents={x['name']:x for x in LIB['intents']}
        cases=[{}, {'closed':True}, {'closed':True,'grasp_secured_once':True},
               {'closed':True,'grasp_confirm':True,'grasp_secured_once':True}]
        for values in cases:
            ep.update(closed=False,grasp_confirm=False,grasp_secured_once=False)
            ep.update(values)
            menu=scope['_acting_menu'](intents)
            self.assertFalse(TRANSPORT & set(menu))
            self.assertIn('close_gripper',menu)
            self.assertIn('reopen_and_retry',menu)

    def test_current_verified_hold_enables_transport_and_loss_removes_it(self):
        scope,ep,sensor,_=fixture()
        ep.update(closed=True,grasp_confirm=True)
        sensor.update(contacts=(True,True),opening=0.04)
        intents={x['name']:x for x in LIB['intents']}
        self.assertTrue(TRANSPORT <= set(scope['_acting_menu'](intents)))
        sensor.update(contacts=(False,False))
        self.assertFalse(TRANSPORT & set(scope['_acting_menu'](intents)))
        sensor.update(contacts=(True,True),identity=False)
        self.assertFalse(TRANSPORT & set(scope['_acting_menu'](intents)))

    def test_forced_transport_cannot_implicitly_close_or_move_empty_gripper(self):
        for intent in TRANSPORT:
            scope,ep,_,commands=fixture()
            ep.update(grasp_secured_once=True)
            result=scope['execute'](intent,{})
            self.assertEqual(result,f'{intent}_not_executed:grasp_unconfirmed')
            self.assertEqual(commands,[])
            self.assertFalse(ep['grasp_secured_once'])

    def test_failed_grasp_returns_explicit_result_to_next_jev_observation(self):
        scope,ep,sensor,commands=fixture()
        result=scope['execute']('close_gripper',scope['namespace']()[0])
        self.assertEqual(result,FAILED)
        self.assertFalse(ep['grasp_confirm'])
        self.assertFalse(ep['closed'])
        self.assertFalse(any(kind=='servo' for kind,_ in commands))
        ep['hist'].append('close_gripper');ep['result_hist'].append(result)
        proprio={key:0 for key in JO.FIELDS}
        scope['_current_observation']=lambda:(proprio,{'front':np.zeros((2,2,3),dtype='uint8')})
        with tempfile.TemporaryDirectory() as temp:
            scope['ep_dir']=temp
            state,_=scope['_jev_input'](include_execution_history=True)
        self.assertEqual(state['execution_history']['recent_actions'],
                         [{'intent':'close_gripper','result':FAILED}])
        self.assertNotIn('carry_to_place',scope['_acting_menu']({x['name']:x for x in LIB['intents']}))

    def test_confirmation_exception_cannot_report_closed_success(self):
        scope,ep,_,_=fixture()
        def fail():raise RuntimeError('injected sensor failure')
        scope['_confirm_grasp_after_close']=fail
        self.assertEqual(scope['execute']('close_gripper',scope['namespace']()[0]),FAILED)
        self.assertFalse(ep['grasp_confirm'])

    def test_confirmed_carry_moves_instead_of_returning_closed(self):
        scope,ep,sensor,commands=fixture()
        ep.update(closed=True,grasp_confirm=True)
        sensor.update(contacts=(True,True),opening=0.04)
        ns={'place':np.array([1.,1.,0.3]),'grasp':np.array([0.,0.,0.3]),
            'objc':np.array([0.,0.,0.3]),'tcp':np.array([0.,0.,0.5])}
        self.assertEqual(scope['execute']('carry_to_place',ns),'reached')
        self.assertEqual([kind for kind,_ in commands],['servo'])
        np.testing.assert_allclose(commands[0][1][0],[1.,1.,0.5])


if __name__=='__main__':unittest.main()
