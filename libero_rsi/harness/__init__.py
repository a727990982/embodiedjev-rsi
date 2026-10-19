"""LIBERO honest adapter + RSI library policy (Astra perception, Jev intent choice, typed rule library).

Policy-visible signals (documented in ADAPTER.md):
  proprioception: robot0_eef_pos (tcp), gripper finger qpos -> opening, last gripper command (closed_cmd)
  tactile proxy: per-finger contact flags from MuJoCo contact list (finger pad/collision geoms of gripper0 touching any
                 geom that is not robot/gripper/mount and not a static fixture body [world/table/floor]). Only booleans; no
                 identity, pose or force of the touched object.
  Astra: one RGB-D (agentview) keypoint perception at t=0 (grasp/object-centre/place pixels -> depth backprojection with
         known camera), plus re-looks (library intent `relook` or early fallback) from the current image.
  Grasp-target ownership (single shared goal): PER holds keypoint base; PATCH holds episode offsets on that same base.
         Priority: latest relook keypoints > slow-think patch on current target > episode-start perception.
         Relook / keypoint refresh overwrites PER and clears position offsets in PATCH; confirmed approach yaw/tilt in PATCH
         survive unless new keypoints explicitly change yaw (or identity-bad full reset). Slow-think / stall mutates PATCH only.
         On stall with identity confirmed: must update the shared target (raise/shift/relook-refresh); relook executes fresh perception.
  success: official env (BDDL check_success) -> done flag. Binary only, ends the episode.
No simulator object/target ground-truth poses reach the policy.
Jev receives task todo, same-frame images/proprioception, measured motion/contact feedback and historical startup obstacle advice.
Close/release applicability is checked by the executor; unchanged failed waypoints require recovery.
Codex planning help: the existing ask_astra/consult_astra intent names request Codex; hard rules and a spectator Jev may request synchronous help on distress signals; general help shares MAX_SLOW; one additional motion-obstruction diagnosis is available after exhaustion (if MAX_SLOW > 0).
Codex handles episode-start global planning and sync/async slow-think. Astra, invoked through Codex CLI, handles visual perception and identity verification; Jev selects each intent. Model roles are fixed.
Start planning includes task_todo; Jev reviews outcome progress on current images after actions. Jev selects applicable actions for the active objective.
After close_gripper: proprio+lift grasp confirm (log grasp_confirm); secured → lift, suppress push_aside/local_deadend undo.
Hard local-deadend rule: repeated local stall family (push_aside/clear_reapproach/raise_redescend/lateral) without progress, or push_aside episode/approach cap → retreat + global front relook + SYNCHRONOUS slow-think replan (Codex slow-think); Jev motion held until plan applied.
Place-phase: after carry_to_place reached, N consecutive lower_to_place stall/timeout (MAX_PLACE_LOWER_STALL) or M lift↔lower cycles (MAX_PLACE_LIFT_LOWER_CYCLES) while TCP/held XY is over the place region → force open_gripper (log place_force_release); avoid endless lift↔lower thrash over basket.
Freshness/isolation: each async slow-think job has a unique job_id + private result_box (abandoned workers cannot overwrite a newer job); after keypoint refresh, close is blocked until TCP re-approaches the new grasp XY; recovery mutations recompute namespace/facts before execute and spectator.

Modules, lowest layer first:
  config       budgets, tolerances and executor result labels (environment variables)
  state        shared episode containers (EP, PER, PATCH, ...), runtime handles `rt`, decision log
  robot        simulator, stepping, video, contacts, camera, servo/grip
  vision       Astra perception, identity gate, keypoint/PATCH ownership, relooks
  observation  policy namespace, Jev inputs, motion context, candidate gating and descriptions
  grasping     grasp confirmation, place-phase forced release
  progress     task todo checkpoints and Jev todo review
  recovery     stall feedback, push-aside, local dead-end, identity-deadlock escape
  pose         optional pose menu (POSE_MENU): library install and geometric execution
  executor     execute one intent, including fine steps
  planning     Codex slow-think jobs and the episode-start plan
  assist       ask_astra, hard triggers, spectator Jev, early fallback
  episode      start-up, decision loop, result.json
Modules refer to each other as `module.function` so that import cycles between
layers that call back into each other (for example recovery <-> planning) resolve
at call time.
"""
