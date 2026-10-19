"""Robot-only geometry, projection, advisory ownership; no rollout or model service.

Optional MuJoCo checks compile a tiny synthetic model and run forward kinematics,
never mj_step. Other tests need only offline requirements.txt dependencies.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot_geometry as RG
import coding_agent_slow as CAS
from test_review_regressions import functions
from test_transport_feedback import fixture
import jev_observation as JO


def model_fixture():
    names = ['world','robot0_arm','gripper0_hand','secret_object']
    m = SimpleNamespace(nbody=4, njnt=2, body_id2name=lambda b:names[b],
        body_parentid=np.array([0,0,1,0]), body_pos=np.array([[0,0,0],[0,0,0],[.2,0,0],[99,99,99]]),
        body_geomadr=np.array([0,0,1,2]), body_geomnum=np.array([0,1,1,1]),
        geom_type=np.array([3,6,2]), geom_size=np.array([[.04,.2,0],[.06,.04,.02],[99,99,99]]),
        geom_contype=np.ones(3),geom_conaffinity=np.ones(3),
        jnt_bodyid=np.array([1,3]),jnt_type=np.array([3,3]),jnt_qposadr=np.array([0,1]),
        joint_id2name=lambda j:['robot0_joint','secret_joint'][j],
        jnt_axis=np.array([[0,0,1],[0,0,1]]),jnt_pos=np.zeros((2,3)),jnt_limited=np.ones(2),
        jnt_range=np.array([[-1,1],[-99,99]]),camera_name2id=lambda n:{'agentview':0,'robot0_eye_in_hand':1}[n],
        cam_fovy=np.array([90.,90.]))
    d = SimpleNamespace(body_xpos=np.array([[0,0,0],[0,0,0],[.2,0,0],[99,99,99]]),
        body_xmat=np.tile(np.eye(3).reshape(1,9),(4,1)),
        geom_xpos=np.array([[0,0,0],[.2,0,0],[99,99,99]]),geom_xmat=np.tile(np.eye(3).reshape(1,9),(3,1)),
        qpos=np.array([.3,999.]),cam_xpos=np.array([[0,0,1],[.1,0,1]]),
        cam_xmat=np.tile(np.eye(3).reshape(1,9),(2,1)))
    return m,d


def snap(m,d):
    return RG.snapshot(m,d,[.2,0,-.05],[0,0,0,1],
                       {'front':('agentview',400,200),'wrist':('robot0_eye_in_hand',400,200)},7)


class GeometryTests(unittest.TestCase):
    def test_malformed_advice_is_uncertain_and_never_imports_plan_fields(self):
        self.assertIsNone(RG.advice([]))
        result=RG.advice({'collision':[], 'visibility':{}, 'evidence':None, 'suggestion':[],
                          'plan':[{'step':'push_aside'}], 'force_next_intent':'close_gripper'})
        self.assertEqual(result['collision'],'uncertain')
        self.assertEqual(result['visibility'],'uncertain')
        self.assertNotIn('plan',result);self.assertNotIn('force_next_intent',result)

    def test_dimensions_joint_state_and_tcp_offsets_use_robot_model(self):
        m,d=model_fixture(); s=snap(m,d)
        self.assertEqual(s['status'],'available')
        self.assertEqual([b['name'] for b in s['bodies']],['robot0_arm','gripper0_hand'])
        np.testing.assert_allclose(s['bodies'][0]['bounds'][0]['dimensions_m'],[.08,.08,.48])
        np.testing.assert_allclose(s['bodies'][1]['bounds'][0]['dimensions_m'],[.12,.08,.04])
        np.testing.assert_allclose(s['bodies'][1]['tcp_offset_body_m'],[0,0,-.05])
        self.assertEqual(s['joints'][0]['position'],.3)
        self.assertEqual(s['joints'][0]['limits'],[-1,1])
        self.assertNotIn('secret',json.dumps(s))
        before=deepcopy(s)
        d.body_xpos[3]=[-88,77,66];d.geom_xpos[2]=[88,77,66];d.qpos[1]=-888
        self.assertEqual(snap(m,d),before)

    def test_projection_matches_rectangular_image_and_tracks_wrist_camera(self):
        m,d=model_fixture(); s=snap(m,d)
        c=s['cameras']['front']
        self.assertAlmostEqual(c['focal_px'],100)
        body={'position_world_m':[0,0,0],'rotation_world':np.eye(3).tolist()}
        box={'center_body_m':[.2,.1,0],'dimensions_m':[0,0,0]}
        np.testing.assert_allclose(RG.projected_edges(body,box,c)[0][0],[220,90])
        old=s['cameras']['wrist']['projected_bounds']
        d.cam_xpos[1,0]+=.3
        new=snap(m,d)
        self.assertNotEqual(new['cameras']['wrist']['projected_bounds'],old)
        self.assertEqual(new['cameras']['front'],s['cameras']['front'])

    def test_near_plane_and_behind_camera_are_not_mirrored(self):
        m,d=model_fixture(); c=snap(m,d)['cameras']['front']
        body={'position_world_m':[0,0,2],'rotation_world':np.eye(3).tolist()}
        box={'center_body_m':[0,0,0],'dimensions_m':[.1,.1,.1]}
        self.assertEqual(RG.projected_edges(body,box,c),[])
        body['position_world_m']=[0,0,1]
        self.assertTrue(np.isfinite(np.array(RG.projected_edges(body,box,c))).all())

    def test_unsupported_geometry_is_reported_not_treated_as_clear(self):
        m,d=model_fixture();m.geom_type[0]=0
        s=snap(m,d)
        self.assertEqual(s['status'],'partial')
        self.assertIn('unsupported robot geom',s['omissions'][0])
        self.assertEqual(RG.snapshot(SimpleNamespace(),None,[0,0,0],[0,0,0,1],{},0)['status'],'unavailable')

    def test_overlay_keeps_original_and_labels_projection(self):
        m,d=model_fixture();s=snap(m,d);rgb=np.full((200,400,3),60,dtype='uint8')
        before=rgb.copy();im=RG.overlay(rgb,s,'front')
        self.assertEqual(im.size,(400,200))
        self.assertFalse(np.array_equal(rgb,np.asarray(im)))
        np.testing.assert_array_equal(rgb,before)

    def test_geometry_snapshot_and_images_reach_existing_astra_call(self):
        m,d=model_fixture();logs=[];ep={'todo_target_id':'target'}
        rgb=np.full((200,400,3),60,dtype='uint8')
        obs={'agentview_image':rgb[::-1],'robot0_eye_in_hand_image':rgb[::-1], 'robot0_eef_quat':[0,0,0,1]}
        with tempfile.TemporaryDirectory() as tmp:
            scope=functions('astra_geometry','remember_geometry_advice','perceive',RG=RG,np=np,Image=Image,
                m=m,sim=SimpleNamespace(data=d),obs=obs,EP=ep,ep_dir=tmp,T=[7],
                eef=lambda:np.array([.2,0,-.05]),rgbd=lambda:(rgb,np.ones((200,400))),
                _active_language_goal=lambda:'put target in destination',S=400,ASTRA_EFFORT='xhigh',
                P={'astra_cost':0,'astra_calls':0,'last_astra_wall':0},time=__import__('time'),
                log=lambda **kw:logs.append(kw))
            raw={'grasp_px':[200,100],'place_px':[300,100], 'robot_geometry_advice':{
                'collision':'possible_obstruction','visibility':'robot_occlusion','evidence':'front view hand covers target',
                'suggestion':'consider retreat for a clearer view','verify':'inspect current target alignment',
                'plan':[{'step':'push_aside'}]}}
            with patch.object(CAS,'call_codex',return_value=(raw,{'cost_usd':0})) as call:
                scope['perceive'](phase='relook')
            prompt,images=call.call_args.args
            self.assertEqual(len(images),4)
            self.assertTrue(all(Path(p).is_file() for p in images))
            self.assertIn('robot0_arm',prompt);self.assertNotIn('secret_object',prompt)
            self.assertIn('Jev chooses the next action',prompt)
            self.assertEqual(ep['robot_geometry_advice']['target_id'],'target')
            self.assertNotIn('plan',ep['robot_geometry_advice'])
            self.assertNotIn('force_next_intent',ep)
            self.assertEqual(scope['T'],[7])
            self.assertEqual(scope['P']['astra_calls'],1)

    def test_advice_rejects_other_target_and_older_snapshot(self):
        ep={'todo_target_id':'new'}
        scope=functions('remember_geometry_advice',RG=RG,EP=ep,T=[30],log=lambda **kw:None)
        fn=scope['remember_geometry_advice'];raw={'evidence':'visible hand','suggestion':'consider back off'}
        fn(raw,{'target_id':'old','observation_step':25},'relook')
        self.assertNotIn('robot_geometry_advice',ep)
        fn(raw,{'target_id':'new','observation_step':25},'relook')
        fn({'evidence':'older'}, {'target_id':'new','observation_step':20},'async')
        self.assertEqual(ep['robot_geometry_advice']['evidence'],'visible hand')
        self.assertNotIn('force_next_intent',ep)

    def test_jev_receives_advice_without_changing_menu_or_stepping(self):
        scope,ep,_,commands=fixture()
        ep.update(todo_target_id='target',robot_geometry_advice={'target_id':'target','observation_step':70,
            'suggestion':'consider back off','source':'relook'})
        scope['_current_observation']=lambda:({k:0 for k in JO.FIELDS},{'front':np.zeros((2,2,3),dtype='uint8')})
        intents={i['name']:i for i in scope['LIB']['intents']}
        before=deepcopy(scope['_acting_menu'](intents))
        with tempfile.TemporaryDirectory() as temp:
            scope['ep_dir']=temp
            state,_=scope['_jev_input'](include_execution_history=True)
        self.assertEqual(state['robot_geometry_advice']['age_steps'],30)
        self.assertEqual(scope['_acting_menu'](intents),before)
        self.assertEqual(list(scope['_acting_menu'](intents)),list(before))
        self.assertEqual(commands,[])


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'optional MuJoCo model compilation')
class CompiledGeometryTests(unittest.TestCase):
    def test_compiled_mesh_and_current_joint_pose_without_physics_steps(self):
        import mujoco
        xml='''<mujoco><asset><mesh name="shape" scale="0.1 0.2 0.3"
          vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/></asset>
          <worldbody><camera name="agentview" pos="0 0 2"/>
          <body name="robot0_link" pos="0.2 0 0"><joint name="robot0_joint" axis="0 0 1" range="-60 60"/>
          <geom type="mesh" mesh="shape"/><body name="gripper0_hand" pos="0.3 0 0">
          <geom type="box" size="0.04 0.03 0.02"/>
          <camera name="robot0_eye_in_hand" pos="0 0 1"/></body></body>
          <body name="secret_object" pos="9 9 9"><freejoint/><geom size="0.1"/></body>
          </worldbody></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        before=d.qpos.copy();s=snap(m,d)
        hand=next(b for b in s['bodies'] if b['name']=='gripper0_hand')
        np.testing.assert_allclose(hand['bounds'][0]['dimensions_m'],[.08,.06,.04],atol=1e-7)
        arm=next(b for b in s['bodies'] if b['name']=='robot0_link')
        # Authored tetrahedron vertices must fit inside the transformed conservative bound.
        bound=arm['bounds'][0];lo=np.array(bound['center_body_m'])-np.array(bound['dimensions_m'])/2
        hi=np.array(bound['center_body_m'])+np.array(bound['dimensions_m'])/2
        vertices=np.array([[0,0,0],[.1,0,0],[0,.2,0],[0,0,.3]])
        self.assertTrue(np.all(vertices>=lo-1e-6));self.assertTrue(np.all(vertices<=hi+1e-6))
        np.testing.assert_array_equal(d.qpos,before);self.assertEqual(d.time,0)
        old=s['cameras']['wrist']['position_world_m']
        d.qpos[0]=.5;mujoco.mj_forward(m,d);new=snap(m,d)
        self.assertNotEqual(new['cameras']['wrist']['position_world_m'],old)
        self.assertAlmostEqual(new['joints'][0]['position'],.5)
        self.assertNotIn('secret',json.dumps(new));self.assertEqual(d.time,0)


if __name__=='__main__':unittest.main()
