"""Release settling and shared recovery regressions from the init8 endgame.

Sensor doubles replay the observed step boundaries, not actual object dynamics.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import coding_agent_slow as CAS
import recovery_tracking as TRACK
from test_review_regressions import functions
from test_transport_feedback import fixture


class FinalizationTests(unittest.TestCase):
    def test_repeated_open_command_does_not_restart_settling(self):
        scope,ep,_,commands=fixture()
        ep['closed']=True
        scope['execute']('open_gripper',scope['namespace']()[0])
        self.assertEqual([a[1][1] for a in commands if a[0]=='grip'],[15,20])
        before=list(commands)
        scope['execute']('open_gripper',scope['namespace']()[0])
        self.assertEqual(commands,before)

    def settle_scope(self, success_at=153, horizon=600):
        t=[142];done=[False];events=[];commands=[]
        obs={'robot0_eef_pos':np.array([-.044,.294,.175])}
        count=[142]
        def env_step(action):
            commands.append(action.copy());count[0]+=1
            return obs,0,False,{}
        env=SimpleNamespace(step=env_step,env=SimpleNamespace(_check_success=lambda:count[0]>=success_at))
        ep={'closed':False,'todo_target_id':'cheese','hist':['open_gripper']}
        s=functions('settle_after_release','post_release_retreat_target','grip','step','eef',
            np=np,T=t,DONE=done,EP=ep,env=env,obs=obs,HORIZON=horizon,VID=None,DIAGF=None,
            log=lambda **kw:events.append(kw))
        return s,commands,events

    def test_settling_advances_physics_and_stops_at_official_success(self):
        s,commands,events=self.settle_scope()
        with patch.object(CAS,'call_monitor',side_effect=AssertionError('no model during settling')):
            s['settle_after_release']()
        self.assertEqual(s['T'][0],153);self.assertTrue(s['DONE'][0])
        self.assertEqual(len(commands),11)
        self.assertTrue(all(np.array_equal(a,[0,0,0,0,0,0,-1]) for a in commands))
        self.assertEqual(events[-1]['post_release_settle']['settle_steps'],11)
        self.assertTrue(events[-1]['post_release_settle']['official_success'])

    def test_wait_is_bounded_and_respects_horizon_and_existing_success(self):
        for horizon,expected in ((600,20),(148,6)):
            s,commands,_=self.settle_scope(success_at=999,horizon=horizon)
            s['settle_after_release']()
            self.assertEqual(len(commands),expected);self.assertFalse(s['DONE'][0])
        s,commands,_=self.settle_scope();s['DONE'][0]=True;s['settle_after_release']()
        self.assertEqual(commands,[])

    def test_retreat_is_release_relative_but_retries_do_not_raise_the_goal_again(self):
        s,_,_=self.settle_scope(success_at=999);s['settle_after_release']()
        fn=s['post_release_retreat_target']
        # Recorded bug: the old max(current_z, place_z+3*h) target equalled current TCP.
        target=np.array([-.044,.294,.175]);original=target.copy()
        first=fn(target)
        np.testing.assert_allclose(first,[target[0],target[1],.235])
        np.testing.assert_array_equal(target,original)
        np.testing.assert_allclose(fn(first),first)
        s['EP']['todo_target_id']=None  # Final visual completion still needs physical withdrawal.
        np.testing.assert_allclose(fn(target),first)
        s['EP']['todo_target_id']='next_object'
        np.testing.assert_allclose(fn(target),target)
        s['EP']['todo_target_id']='cheese';s['EP']['closed']=True
        np.testing.assert_allclose(fn(target),target)

    def test_normal_release_defers_model_help_to_jev_current_image_review(self):
        calls=[]
        s=functions('_request_help',EP={'just_released':True},T=[162],TRACK=TRACK,
            FAIL_RESULTS={'stalled','timeout'},ASK_ASTRA_NAMES={'ask_astra'},
            _abandon_async_slow=lambda why:calls.append(why),log=lambda **kw:None)
        with patch.object(CAS,'call_monitor',side_effect=AssertionError('no immediate after-release call')):
            self.assertIsNone(s['_request_help']('open_gripper','at_target'))
        self.assertEqual(calls,['post_release_settle'])
        self.assertNotIn('just_released',s['EP'])

    def recovery_scope(self):
        ep={'hist':['retreat']*4,'tcp_hist':[np.zeros(3)]*4,'closed':False,'todo_target_id':'cheese',
            'n_close':1,'result_hist':['at_target']*4,'grip_hist':[False]*4}
        t=[142];seen=[]
        def finish(*args):
            t[0]+=9  # Recovery internally moves from 142 to 151, then relooks.
            seen.append('applied')
            return 'intervened',{}
        s=functions('_sync_slow','do_fallback','_fallback_once','fb_trigger','_slow_critical',
            TRACK=TRACK,np=np,EP=ep,T=t,P={'slow':0,'fb':0,'relook':0},PATCH={},
            FALLBACK=True,MAX_FB=3,MAX_RELOOK=4,MAX_SLOW=4,SLOW_JOB={},dec=11,
            _stage_fail_grip_unchanged=lambda **kw:None,
            _abandon_async_slow=lambda **kw:None,_drain_orphan_slow_costs=lambda:None,
            _combo_rgb=lambda:(None,None,None),_slow_ctx=lambda *a:'{}',_next_slow_job_id=lambda:1,
            _prepare_slow_snapshot=lambda *a:{'coding_context':{}},_finish_slow=finish,
            do_relook=lambda reason:seen.append('relook') or 'relooked_matched',log=lambda **kw:None)
        return s,seen

    def test_slow_recovery_and_next_loop_fallback_share_the_handled_state(self):
        s,seen=self.recovery_scope()
        with patch.object(CAS,'call_monitor',return_value=({'issue':'other'},{})) as call:
            self.assertEqual(s['_sync_slow']('retreat','at_target','no_progress_same_x3')[0],'intervened')
            self.assertEqual(s['T'][0],151)
            self.assertIsNone(s['fb_trigger']())
            self.assertEqual(s['do_fallback']('no_progress_same_intent'),('recovery_already_handled',None))
            self.assertEqual(s['_sync_slow']('retreat','at_target','no_progress_2_3'),('recovery_already_handled',None))
            s['EP']['hist'].append('retreat');s['EP']['tcp_hist'].append(np.zeros(3))
            self.assertIsNone(s['fb_trigger']())  # Another no-op selection is not a new physical state.
            self.assertEqual(call.call_count,1);self.assertEqual(seen,['applied'])

    def test_new_motion_does_not_reuse_old_history_but_new_failures_can_escalate(self):
        s,seen=self.recovery_scope()
        with patch.object(CAS,'call_monitor',return_value=({'issue':'other'},{})) as call:
            s['_sync_slow']('retreat','at_target','no_progress_same_x3')
            s['T'][0]+=1
            self.assertFalse(TRACK.handled(s['EP'],s['T'][0]))
            self.assertIsNone(s['fb_trigger']());self.assertIsNone(s['_slow_critical']('retreat','at_target'))
            s['EP']['hist'].extend(['retreat']*4);s['EP']['tcp_hist'].extend([np.zeros(3)]*4)
            self.assertEqual(s['fb_trigger'](),'no_progress_same_intent')
            self.assertEqual(s['do_fallback']('no_progress_same_intent')[0],'intervened')
            self.assertEqual(call.call_count,2);self.assertEqual(seen.count('relook'),1)

    def test_failed_model_call_is_not_retried_automatically_without_new_state(self):
        s,seen=self.recovery_scope()
        with patch.object(CAS,'call_monitor',side_effect=RuntimeError('network failure')) as call:
            self.assertTrue(s['_sync_slow']('retreat','at_target','no_progress_same_x3')[0].startswith('sync_error'))
            self.assertEqual(s['do_fallback']('no_progress_same_intent')[0],'recovery_already_handled')
            self.assertEqual(call.call_count,1);self.assertEqual(seen,[])

    def test_target_change_and_explicit_jev_request_remain_available(self):
        s,_=self.recovery_scope();TRACK.mark(s['EP'],s['T'][0])
        s['EP']['todo_target_id']='butter'
        self.assertFalse(TRACK.handled(s['EP'],s['T'][0]))
        TRACK.mark(s['EP'],s['T'][0])
        with patch.object(CAS,'call_monitor',return_value=({'issue':'ok'},{})) as call:
            s['_sync_slow']('ask_astra','at_target','jev_ask:review')
            self.assertEqual(call.call_count,1)

    def test_success_inside_recovery_stops_before_another_intent(self):
        for phase in ('fallback','rule'):
            with self.subTest(phase=phase):
                t=[151];done=[False]
                ep={'closed':False,'prev':'retreat'}
                def fallback(*a):
                    t[0]=153;done[0]=True
                    return 'intervened',None
                def rules(*a):
                    t[0]=153;done[0]=True
                    return None,True
                s=functions('run_decision_loop',EP=ep,T=t,DONE=done,LIB={'intents':[]},
                    HORIZON=600,MAX_DEC=60,dec=11,P={'fb':0},CUR={},
                    _refresh_task_progress=lambda *a:None,
                    fb_trigger=lambda:'stuck' if phase=='fallback' else None,
                    do_fallback=fallback,_apply_recovery_rules=rules,
                    namespace=lambda:({},{}),facts=lambda *a:{},
                    log=lambda **kw:None,
                    JC=SimpleNamespace(choose=lambda *a,**kw:self.fail('must not request another decision')))
                run={'res':{},'fb_events':[],'slow_events':[],'jev_calls':0}
                self.assertEqual(s['run_decision_loop'](run),'success')
                self.assertEqual(s['rt'].dec,11)
                self.assertEqual(run['jev_calls'],0)


if __name__=='__main__':unittest.main()
