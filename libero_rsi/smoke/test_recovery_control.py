"""Exercise rule ownership against real rule-loop and plan-application code."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import recovery_control as RC
from test_review_regressions import functions


class ControlTests(unittest.TestCase):
    def test_default_retains_main_behavior_and_typo_fails(self):
        self.assertEqual(RC.parse_mode(None),'both')
        self.assertEqual(RC.parse_mode('jev'),'jev')
        with self.assertRaises(ValueError):RC.parse_mode('jve')

    def run_rules(self,mode,rules):
        tree=ast.parse((ROOT/'run_ep_fbpatch.py').read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='rc')
        calls=[];ep={'uses':{},'force_next_intent':None}
        scope=dict(LIB={'recoveries':[dict(r,trigger='True') for r in rules]},forced=None,
            dsl=SimpleNamespace(evaluate=lambda *a:True),ns={},dec=0,T=[0],EP=ep,RC=RC,
            RECOVERY_MODE=mode,intents={'open_gripper':{},'move_above_object':{}},
            log=lambda **kw:None,_apply_stall_feedback=lambda *a,**kw:calls.append('stall') or {},
            _apply_relative_approach_change=lambda *a:calls.append('approach'),
            STALL_FB_ACTIONS={'clear_reapproach'})
        exec(compile(ast.Module(body=[loop],type_ignores=[]),'rule_loop','exec'),scope)
        return scope['forced'],ep,calls

    def test_skipped_rules_do_not_move_force_or_spend_uses(self):
        for key in ['stall_feedback','approach_change']:
            forced,ep,calls=self.run_rules('jev',[{'id':'stall',key:True,'force_intent':'move_above_object'},
                {'id':'release','force_intent':'open_gripper'}])
            self.assertEqual(calls,[])
            self.assertNotIn('stall',ep['uses'])
            self.assertEqual(ep['uses']['release'],1)
            self.assertEqual(forced,'open_gripper')
        forced,ep,calls=self.run_rules('both',[{'id':'stall','stall_feedback':True,'force_intent':'move_above_object'}])
        self.assertEqual(calls,['stall']);self.assertEqual(forced,'move_above_object')
        self.assertEqual(ep['uses']['stall'],1)

    def test_empty_plan_does_not_invent_recovery_but_explicit_plan_still_runs(self):
        for mode,steps,expected in [('jev',[],[]),('jev',['clear_reapproach'],['clear_reapproach']),('both',[],['push_aside'])]:
            calls=[]
            scope=functions('_apply_episode_plan','identity_fields',np=np,AUTO_RECOVERY_RULES=mode=='both',
                EP={'closed':False,'last_result':'stalled'},LIB={'intents':[]},PATCH={},
                NEIGHBOUR_STALL_ISSUES=set(),goal_id_confirmed=lambda:True,
                _should_open_on_intervention=lambda **kw:False,namespace=lambda:({},{}),
                _apply_stall_feedback=lambda *a,**kw:calls.append(kw.get('action')) or {'action':kw.get('action')})
            reply={'issue':'blocked_by_neighbour','is_goal_object':True,'matched_language_goal':True,
                   'plan':[{'step':s} for s in steps]}
            scope['_apply_episode_plan'](reply,None)
            self.assertEqual(calls,expected)


if __name__=='__main__':unittest.main()
