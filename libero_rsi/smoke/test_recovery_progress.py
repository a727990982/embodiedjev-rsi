"""Preparation loops must escalate without forcing a descent or claiming clearance."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import motion_feedback as MF
import coding_agent_slow as CAS
from test_review_regressions import functions
from test_transport_feedback import fixture


class RecoveryTests(unittest.TestCase):
    def cycle(self, heights=(.05,.05,.05), contact_parts=None):
        intents=['move_above_object','clear_reapproach','move_above_object',
                 'clear_reapproach','move_above_object']
        results=['reached','recovery_prepared:clear_reapproach:descent_unverified','reached',
                 'recovery_prepared:clear_reapproach:descent_unverified','reached']
        positions=[[0,0,z] for z in [heights[0],.12,heights[1],.12,heights[2]]]
        return intents,results,positions,[0,0,0],contact_parts or [[] for _ in intents]

    def test_alternating_motor_successes_are_not_task_progress(self):
        state=MF.recovery_progress(*self.cycle())
        self.assertTrue(state['no_progress'])
        self.assertEqual(state['preparations_without_descent'],2)
        self.assertAlmostEqual(state['error_reduction_m'],0)

    def test_meaningful_error_reduction_or_contact_clearance_does_not_escalate(self):
        self.assertFalse(MF.recovery_progress(*self.cycle((.06,.05,.04)))['no_progress'])
        self.assertFalse(MF.recovery_progress(*self.cycle(contact_parts=[['palm_or_wrist'],[],[],[],[]]))['no_progress'])

    def test_small_drift_cannot_disguise_loop_but_later_plateau_is_detected(self):
        args=list(self.cycle((.05,.049,.048)))
        self.assertTrue(MF.recovery_progress(*args)['no_progress'])
        args=list(self.cycle((.06,.04,.04)))
        args[0]+=['clear_reapproach','move_above_object']
        args[1]+=['recovery_prepared:clear_reapproach:descent_unverified','reached']
        args[2]+=[[0,0,.12],[0,0,.04]];args[4]+=[[],[]]
        self.assertTrue(MF.recovery_progress(*args)['no_progress'])

    def test_real_descent_resets_preparation_loop_even_when_it_stalls(self):
        args=list(self.cycle())
        args[0].append('descend_to_grasp');args[1].append('stalled')
        args[2].append([0,0,.025]);args[4].append(['palm_or_wrist'])
        state=MF.recovery_progress(*args)
        self.assertFalse(state['no_progress'])
        self.assertEqual(state['preparations_without_descent'],0)

    def test_repeated_approach_without_recovery_is_not_this_trigger(self):
        args=list(self.cycle());args[0]=['move_above_object']*5;args[1]=['reached']*5
        self.assertFalse(MF.recovery_progress(*args)['no_progress'])

    def test_recorded_init8_loop_triggers_before_ordinary_budget_was_exhausted(self):
        data=json.loads((ROOT/'smoke/fixtures/init8_recovery_loop.json').read_text())
        trigger_steps=[]
        for window in data['windows']:
            h=[];r=[];p=[];c=[]
            for action in window:
                h.append(action['intent'])
                # Translate the old executor label; measured inputs stay fixed.
                result=action['result']
                if result.startswith('recovery:'):
                    result=result.replace('recovery:','recovery_prepared:',1)+':descent_unverified'
                r.append(result);p.append(action['tcp']);c.append(action['contacts'])
                if MF.recovery_progress(h,r,p,data['visual_grasp_estimate'],c)['no_progress']:
                    trigger_steps.append(action['step'])
        self.assertEqual(trigger_steps,[103,593])

    def test_context_contains_actual_preparation_and_resets_on_target_switch(self):
        h,r,p,g,c=self.cycle()
        ep={'hist':h,'result_hist':r,'tcp_hist':p,'recovery_contact_hist':c,
            'last_recovery_preparation':{'step':40,'action':'clear_reapproach'},'last_descent_step':20}
        scope=functions('_recovery_context',EP=ep,MF=MF,namespace=lambda:({'grasp':np.array(g)},{}))
        state=scope['_recovery_context']()
        self.assertTrue(state['no_progress'])
        self.assertFalse(state['descent_tested_since_preparation'])
        ep['recovery_history_start']=len(h);ep['last_recovery_preparation']=None
        self.assertFalse(scope['_recovery_context']()['no_progress'])

    def test_exhausted_budget_uses_shared_reserve_once_even_on_error(self):
        ep={};p={'slow':4};seen=[]
        scope=functions('_hard_astra_trigger','_sync_slow',EP=ep,P=p,MAX_SLOW=4,FALLBACK=True,
            _recovery_context=lambda:{'no_progress':True},ASK_ASTRA_NAMES=set(),FAIL_RESULTS={'stalled'},
            SLOW_JOB={'thread':None,'job_id':None},_drain_orphan_slow_costs=lambda:None,
            _combo_rgb=lambda:(None,None,None),_slow_ctx=lambda *args:'{}',
            _prepare_slow_snapshot=lambda ctx,crit,*args:seen.append(crit) or {'coding_context':{}},
            _next_slow_job_id=lambda:1,log=lambda **kw:None,T=[100])
        trigger=scope['_hard_astra_trigger']('move_above_object','reached')
        self.assertEqual(trigger,'recovery_no_progress')
        with patch.object(CAS,'call_monitor',side_effect=RuntimeError('offline')) as call:
            self.assertIn('sync_error',scope['_sync_slow']('move_above_object','reached',trigger)[0])
            self.assertEqual(scope['_sync_slow']('move_above_object','reached',trigger)[0],'slow_cap')
        self.assertEqual(call.call_count,1)
        self.assertEqual(seen,['reserved_motion_obstruction'])
        self.assertIsNone(scope['_hard_astra_trigger']('move_above_object','reached'))
        self.assertIsNone(scope['_hard_astra_trigger']('descend_to_grasp','stalled'))
        self.assertEqual(p['slow'],4)

    def test_executor_reports_preparation_without_claiming_descent(self):
        scope,ep,_,commands=fixture()
        scope['LIB']={**scope['LIB'],'intents':scope['LIB']['intents']+[{'name':'clear_reapproach','fingers':'open'}]}
        scope['_apply_stall_feedback']=lambda ns,reason,action:{'action':action}
        self.assertEqual(scope['execute']('clear_reapproach',scope['namespace']()[0]),
                         'recovery_prepared:clear_reapproach:descent_unverified')
        self.assertFalse(commands)
        self.assertEqual(ep['n_close'],0)


if __name__=='__main__':unittest.main()
