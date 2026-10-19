# embodiedjev-rsi

Public snapshot of the LIBERO RSI / EmbodiedJev harness. This repo had no prior tree, so the agent module, episode runner, experience brief, libraries, logic smokes, and review notes all live under `libero_rsi/`.

Standard movement intents now use **Jev intent → Jev XYZ/finger Choices → a
bounded LIBERO servo block → fresh observation**. This adapts the two-stage
controller from Jev-as-Policy. Directions are chosen together in one motor
request; distance follows the visual-estimate waypoint and controller limits.
See [Cartesian policy migration](libero_rsi/reviews/cartesian_policy.md) for the
execution contract, differences from upstream, call accounting and validation.

| Path | Role |
|---|---|
| `libero_rsi/coding_agent_slow.py` | Codex CLI transport for start planning, slow-think, Astra perception and identity checks. |
| `libero_rsi/run_ep_fbpatch.py` | Command-line entry point for one episode. Needs LIBERO, MuJoCo, Codex CLI, and the Jev client to actually roll out. |
| `libero_rsi/harness/` | Episode harness package: Jev intents, stall feedback, grasp confirm, local-deadend relook, place force-release. The module map is in `harness/__init__.py`; it can be imported without the simulator. |
| `libero_rsi/jev_experience.py` | Builds library advice for Astra/Codex planning; cached predicates are not Jev input. |
| `libero_rsi/grasp_experience.py` | Curated conditional grasp advice for acting Jev, matched from its existing observations. |
| `libero_rsi/robot_geometry.py` | Own-robot dimensions, joint/body poses, camera projections and visual advice normalization. |
| `libero_rsi/jev_todo.py` | Jev completion questions, classification records, and atomic progress updates. |
| `libero_rsi/task_todo.py` | Task objectives, evidence-backed progress, and active object selection. |
| `libero_rsi/jev_observation.py` | Same-frame camera/proprioception input and applicable intent definitions. |
| `libero_rsi/jev_choice.py` | Bundled TypeSafe Jev 1.14 image-capable HTTP client. |
| `libero_rsi/cartesian_policy.py` | XYZ/finger motor questions and signed, endpoint-clamped displacement. |
| `libero_rsi/validate.py` | Library validator. Imports sibling `dsl_libero.py`. |
| `libero_rsi/dsl_libero.py` | Expression DSL and dimensional checks used by the validator and logic smokes. |
| `libero_rsi/libs/*.json` | Experience libraries, including `ee_pg_easy_v2_lib.json`. |
| `libero_rsi/smoke/test_*.py` | Offline tests of validators, routing and harness logic; no simulator or network. |
| `libero_rsi/reviews/*.md` | Design notes for the behaviors below. |

## Todo and current-image decisions

Every Astra/Codex perception, identity, start-plan and slow-think request now
includes aligned metric depth for the available front/wrist cameras. Each
geometry image is a two-panel RGB-geometry/depth view, leaving the original RGB
images and their keypoint coordinates unchanged within the four-image limit.
`robot_geometry.depth_views` includes camera-axis units, a fixed 0.10–2.00 m
colour scale, valid/saturated fractions and an 8×8 grid of regional minimum,
10th-percentile and median distances. Full-resolution metric arrays are saved
as `depth_*.npy`; invalid/far-plane pixels remain unknown. Missing camera depth
is labelled unavailable, including when robot geometry itself is unavailable.

The visual instructions require depth inspection on every call and comparison
of direct, raised and lateral corridors for planning. `robot_geometry_advice`
can carry three evidence-backed `path_candidates` and a `selected_path` to Jev,
with the existing target/age checks. These are suggestions for available actions;
the code does not invent a new waypoint program or certify whole-arm swept-volume
clearance. Hidden surfaces, invalid pixels and future arm configurations remain
unknown. No simulator object poses, names or collision oracle are supplied.

Offline checks cover buffer conversion/alignment, regional obstacle evidence,
missing and saturated depth, advice delivery and all existing Codex request
routes. Real-model depth interpretation and collision avoidance remain untested.

Astra perception/identity, startup planning and slow-think now receive a
`robot_geometry` snapshot and annotated current camera views alongside original
images (within the existing four-image request limit). The snapshot reads only
the loaded robot0/gripper0/mount0 model bodies, own joint positions/limits and the
front/wrist camera calibration. It reports link frame offsets, current poses,
TCP offsets and body-local dimensions from collision/visual geometry. Mesh
vertices use MuJoCo's compiled transforms; no nominal Franka dimensions are
hard-coded. Surrounding object poses, meshes and joints are excluded.

Cyan collision-model and magenta visual-model boxes conservatively bound each
robot body. They are not exact silhouettes: boxes can overlap or show through
scene objects. Current projections do not certify visibility, clearance, an IK
solution or a future arm configuration. Unsupported geometry/cameras are listed
as omissions. Full snapshot JSON and annotated PNGs are saved under `geometry_*`
and referenced in the decision log; raw images are retained.

Astra can return `robot_geometry_advice` with separate collision and visibility
assessments, image evidence, a conditional suggestion and an outcome check. The
host passes descriptive fields to acting Jev with the source step, target and
age. It never executes that field or changes the action menu. Target switches
clear it, older replies cannot replace newer advice, and asynchronous results
still pass the existing freshness checks before acceptance. Existing recovery
rules remain unchanged. Jev checks current images and selects the action.

Offline tests cover robot-only data selection, primitive/mesh bounds, rectangular
camera projections, moving wrist-camera poses, near-plane clipping and advice
delivery without extra calls or movement. An optional MuJoCo test compiles a tiny
synthetic model and runs forward kinematics without physics steps. This is not
validation on the deployed LIBERO robot or evidence of better collision avoidance.

Astra (`gpt-6-astra`), invoked through Codex CLI, writes `task_todo` as part of
`start_plan.json`. Each item names an object, its desired final outcome and a
visual completion condition. A multi-object task has one complete outcome per
requested object. Planning never marks an item done. The current todo is saved
in `task_todo.json`, with updates also recorded in `decisions.jsonl`.

Each object has six nested progress checkpoints: verify identity, secure grasp,
lift clear, carry above the destination, lower for release, and release/retreat
with verified final placement. The executor refreshes these before decisions and
after actions/recovery using grasp confirmation, current tactile support and
completed motion results. After a confirmed grasp and completed lift, Jev sees
those checkpoints checked and `next_stage=carry_to_destination`. Losing the
grasp or reopening prematurely resets the dependent checkpoints. A refreshed
localization invalidates old motion waypoints. These updates make no model calls
and do not change the active object or force an action sequence.

Jev sees older scene descriptions under `last_visual_review`, separate from the
fresh `progress` and `current_grasp_confirmed` fields. A pending final outcome
does not mean grasping failed. An expected release still awaits visual placement
review; sending an open command never checks off the final outcome. Successful
placement can make an unobserved intermediate checkpoint `not_required`, which
does not claim that the corresponding motion was executed.

Acting Jev receives the todo, current front/wrist images (wrist when available),
end-effector pose, finger joint positions/opening, commanded open/closed state,
per-finger contacts and official success. Current images and sensors come from the
same simulation step, after any recovery motion. `motion_context` adds robot-part
contact locations (including palm/arm and fixtures), the last requested/actual
movement and remaining error, plus a visual-waypoint proximity check. Failed
motion before-images and startup obstacle advice are explicitly historical.
Commanded waypoints are estimates, never simulator object truth or a certificate
of a collision-free path. Cached `seen_held` and library predicates are not sent.

`motion_context.current_orientation` compares the full commanded world rotation
with the current measured quaternion, reporting the shortest 3-D residual in
radians/degrees and its tolerance. Each servo report also records initial,
target and actual quaternions, net requested/actual rotation, position arrival,
and whether orientation was required for `reached`. Position-only completion
must not be interpreted as rotation completion. The most recent explicit rotation
attempt remains in `last_orientation_motion` with its original step, even after
a later translation; target switches clear this history.

Yaw-changing recovery and pose-menu rotation/home require position AND orientation
convergence, using the existing pose-menu tolerance of 0.08 rad (about 4.6 degrees).
Failed yaw recovery returns its motion failure instead of a prepared result and
does not queue the normal approach follow-up. Rotation toward Astra verifies the
real pose even when the two commanded yaws already agree. Rotational progress
counts against servo stall detection for orientation-constrained motions. This
verifies pose arrival, not environmental clearance or grasp success.

Acting Jev also receives `grasp_experience`: up to three short entries with an
applicable condition, interpretation, options to consider, outcome checks and
source references. The ten curated entries cover identity uncertainty, failed
capture, refused execution, untested recovery preparation, stalled motion,
alignment, capture verification, supported transport, release and placement
review. All are labeled `curated_hypothesis`; references to past failures do not
establish that the proposed recovery works. The selector uses current task
checkpoints, proprioception, motion context and completed actions since the
current target was selected. It does not infer obstruction from an old failure
image or startup prediction, and emits no matches for stale checkpoints or a
completed task. Supported holds and pending placement reviews take priority over
old grasp failures.

This advice is attached to the existing acting request and recorded verbatim in
`decisions.jsonl`. It does not change candidates, ordering, guards or forced
recovery rules, and makes no extra model call or simulator step. Jev evaluates
the advice against current images and selects the action. Spectator and todo
review requests do not receive this action advice. Astra/Codex retains its
existing library brief; no episode reflection is automatically promoted into
this curated catalog. Offline fixtures and fake execution chains verify these
boundaries, not improved model decisions or physical task success.

Acting Jev additionally receives the last eight completed intent/result pairs,
plus the consecutive count of the same intent and the same intent/result pair.
Counts cover the full trailing streak even when it exceeds eight actions.
These are past executor outcomes, not assertions about the current scene or
task completion. The instructions explain stage progress and preserving a
confirmed grasp. Candidate and executor checks reject closes above/off the grasp
position, opening/reapproach motions that would discard a supported target grasp,
and release before the destination waypoint. An unchanged failed motion requires
a changed target/orientation or an executed back-off before retry.

After an actual `open_gripper` release, the executor holds the fingers open and
advances up to 20 additional physics steps so the object can settle. Existing
terminal/horizon guards stop this wait immediately; a repeated open command does
not restart it. The next normal check is Jev's current-image todo review, rather
than an unconditional synchronous Astra call. If Jev selects `retreat`, its goal
is at least 6 cm above the recorded release pose. That clearance height is fixed
for this release, so retries do not keep raising it. `release_finalization` records
the actual wait steps and clearance goal. Neither release nor waiting claims
success: only the official environment check does.

Automatic slow-think and early fallback share a handled-state checkpoint (physics
step, active target and gripper command). They record both the request state and
the state reached by recovery, including errors, and exclude pre-recovery action
history from subsequent repetition checks. An unchanged physical state cannot
trigger another automatic Astra transaction through the other path. New physical
steps or targets allow fresh assessment, and explicit Jev help remains available
within its existing budget. If a recovery reaches official success or the horizon,
the loop stops before another model decision or forced intent.
`lift`, `carry_to_place` and `lower_to_place` are offered only with a confirmed
target grasp supported by current gripper contacts/opening. The executor repeats
this check, including for forced actions, and never implicitly closes an empty
gripper for a transport request. A failed close returns
`grasp_failed:transport_not_executed`; a blocked transport returns
`<intent>_not_executed:grasp_unconfirmed`. These failure results enter the next
Jev history and trigger the existing failure/todo-review paths.

Astra's startup assessment receives an annotated TCP approach/descent route,
visual waypoint estimates, commanded orientation and finger opening. It reports
clear/risk/uncertain, the affected phase/robot part and a suggested recovery.
Jev can choose `back_off` (preserves grip), `clear_reapproach`, or
`change_approach_yaw` using the existing recovery executor. General slow-think
uses `MAX_SLOW`; when that budget is exhausted, one additional **motion-stall**
diagnosis is available if `MAX_SLOW > 0` and fallback is enabled. It is spent even
on a failed API request. No unbounded retry or runtime code change is introduced.

Candidates describe applicability and actual motion. An empty closed gripper
should reopen; `reopen_and_retry` opens **at the current pose**, leaving alignment
or relook to the next decision. Repeating `close_gripper` while already closed
does not move. Spectator Jev uses the same current-image input boundary.

After each executed intent, Jev checks every todo item from fresh, same-frame
images and proprioception before the next decision. Astra writes concrete
completion conditions at startup, for example: **Is the butter inside the
basket and no longer held by the gripper?** Jev chooses `done`, `pending`,
`uncertain` (visible but ambiguous), or `unobserved` (cannot inspect the required
object/region). A release command, an object hovering above the basket, or a
wrong object in the basket is not completion. Prior verdicts are omitted from
the review request to avoid copying an earlier tick.

One multi-question Jev request checks all items; the main thread validates the
complete reply, updates the todo, and switches/relocalizes the target before
asking Jev for its next action. These are **separate synchronous calls**, so
next-action selection sees the updated target. Unobserved items preserve earlier
status and its original evidence timestamp; visible contradictory evidence can
reopen a completed item. At a budget limit, the final non-successful scene is
reviewed without initiating another target-directed motion. Official success
completes remaining items directly and is still required to finish successfully.

Status changes invalidate outstanding slow advice; unchanged reviews do not.
Target changes also clear old grasp/approach state and trigger Astra localization
and identity verification. Astra/Codex retains startup planning, perception and
recovery help; routine todo checks use Jev. Logs retain review questions,
classifications, probabilities and references to the exact current frames.

Controller coordinates and DSL predicates remain internal to control/recovery;
removing them would make motion execution impossible. Library recovery rules
and safety gates can still override Jev. The start planner's phase/clearance
proposal is retained for planning records and slow-think, but is not executed
against an old target after todo localization. See [runtime contract](libero_rsi/ADAPTER.md).

## Local dead-end → synchronous global relook

Recovery preparation is not a successful retry. `clear_reapproach` raises the
open hand, recenters and adjusts the visual grasp estimate, then queues
`move_above_object`; it never descends. Its executor result is now
`recovery_prepared:clear_reapproach:descent_unverified`. Jev still chooses whether
the current view supports a descent test, another approach, or help.

`motion_context.recovery_validation` reports the latest preparation position/commanded orientation
and whether a descent has been tested since. Two preparation cycles between three
approach endpoints trigger `recovery_no_progress` when they reduce distance to the
same visual grasp estimate by less than 6 mm and clear no robot-part contact.
Only approach endpoints are compared: the upward retreat itself is not progress.
Real descent attempts reset the preparation cycle; task-target changes discard
the previous target's history. This trigger can spend the same single reserved
diagnosis used for motion stalls, even after ordinary help is exhausted, including
when the diagnostic request fails. The latest failed-motion report and historical
before image remain available to that diagnosis. This detects and escalates a
preparation loop; it does not establish that the revised recovery succeeds in physics.

Repeated local stall-family actions (`push_aside`, `clear_reapproach`, `raise_redescend`, lateral recenter) without a secured grasp, or hitting the push-aside caps, abort the local loop:

1. Retreat for a clear front (agentview) look.
2. Global relook of keypoints.
3. **Synchronous Codex** slow-think before Jev moves again.

Caps: `LOCAL_DEADEND_N` (default 3), `MAX_LOCAL_DEADEND_GLOBAL` (default 2), `MAX_PUSH_ASIDE` (default 2), `MAX_PUSH_ASIDE_PER_APPROACH` (default 1).

## Grasp confirm and place force-release

After `close_gripper`, `_confirm_grasp_after_close` checks proprio (opening and pads) plus a short closed lift probe. `grasp_confirm=true` forces lift and suppresses `push_aside`, local-deadend retreat, and stall undo while the object is held. A failed confirm reopens and reapproaches.

During place, consecutive `lower_to_place` stalls (`MAX_PLACE_LOWER_STALL`, default 3) or lift↔lower cycles (`MAX_PLACE_LIFT_LOWER_CYCLES`, default 2), while the held object is near the place XY, set `place_force_release` and force `open_gripper`.

## Fixed model roles

| Role | Model | Invocation |
|---|---|---|
| Episode-start plan/todo | Astra (`gpt-6-astra`) | Codex CLI |
| Sync/async slow-think | `CODING_AGENT_CODEX_MODEL` (default `gpt-6-astra`) | Codex CLI |
| Visual perception, keypoints and identity verification | Astra (`gpt-6-astra`) | Codex CLI |
| Todo completion review | TypeSafe `jev-1.14.0` | `jev_todo.review` → `jev_choice.choose_questions` |
| Per-step intent selection and spectator decision | TypeSafe `jev-1.14.0` | Bundled `jev_choice.choose`, with image assets |

Codex is the invocation mechanism; Astra is the visual model. Perception and marked-grasp identity verification both use `codex exec -m gpt-6-astra`. There is no dual/comparison mode or direct Astra-client transport. The former `CODING_AGENT_SLOW`, `EPISODE_START_PLAN`, and `CODING_AGENT_PERCEIVE` role switches are no longer read. The mock backend is removed from all helpers; unsupported backend values are rejected.

Early recovery refreshes perception with Astra through Codex CLI, then requests Codex planning. CLI failures do not switch to a direct Astra client or to mock results. The existing `ask_astra`, `consult_astra`, and `defer_to_astra` intent labels are retained for the shipped libraries and logs, but their help path invokes Codex planning.

Codex looks up `codex` on `PATH`, or `CODING_AGENT_CODEX_BIN`. Slow-think uses `CODING_AGENT_CODEX_MODEL` and `CODING_AGENT_CODEX_EFFORT`; start-plan/todo-generation calls pin Astra and use `CODING_AGENT_CODEX_EFFORT`. Perception/identity uses `gpt-6-astra` with `ASTRA_EFFORT` (default `xhigh`). A proxy is used only when `CODING_AGENT_HTTP_PROXY` or `CODING_AGENT_SOCKS` is set. Do not commit proxy URLs, keys, or host details.

Async slow-think captures a detached copy of policy state and a unique image before starting its worker. Workers return results and log events in private job boxes; the main thread writes the decision log. Simulator diagnostic files are never passed to the coding monitor, including when `DIAG` is enabled. Async advice is discarded after target revision, gripper state, close-attempt count, or identity confirmation changes. At shutdown pending calls are joined and logged without applying their plans.

## Running an episode

The bundled Jev client requires `TYPESAFE_API_KEY` in the process environment.
It pins `jev-1.14.0` and sends images using TypeSafe assets and the image opt-in
header. It does not load a similarly named external client or silently use a
text-only model. The [TypeSafe image protocol](https://docs.typesafe.ai/images)
is used for acting, spectator decisions and todo reviews.

Real rollouts also require an authenticated Codex CLI, LIBERO with its assets,
MuJoCo/robosuite, the packages in `requirements.txt`, `ffmpeg` with libx264, and rendering dependencies. `astra_client`
and an external `jev_choice` are not needed. After configuring credentials and
dependencies:

```bash
python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIB.json OUTDIR
```

Real episodes record `OUTDIR/SUITE_tTASK_iINIT/episode.mp4` by default, with
front/wrist views and step/action captions. Encoding uses one CPU thread. Set
`VIDEO=/path/to/episode.mp4` to choose another output path; offline tests use
`VIDEO=''` to disable recording. The result includes the video path and encoder
exit code. A missing ffmpeg fails before model calls instead of silently omitting
the recording.

A failed start plan stops the episode before Jev decisions and is recorded as `harness_error`. Libraries are validated before environment creation. Recovery-rule errors are logged and end the episode with `harness_error`, rather than silently continuing with potentially partial state changes. The same applies to any exception from robot motion, grasp confirmation, stall feedback, or place-phase bookkeeping: the traceback goes to `result.json` under `error`.

Only two kinds of failure are survivable. External model calls (Astra perception, identity checks, Codex planning) have their documented fallback results such as `relook_error` or `sync_error:...`. Diagnostics (log records, image dumps, environment teardown) and library DSL fallbacks (a predicate that fails evaluates to false; a bad `gain` uses 1.0) are recorded rather than swallowed: each one appears as a `harness_warning` line in `decisions.jsonl` and in the `nonfatal_errors` list in `result.json`.

## Fine-step menu

Default off. `FINE_STEPS=1` merges `libero_rsi/libs/ee_pg_fine_steps.json` into whichever library the episode was given, before validation. Unset or `FINE_STEPS=0` leaves that library byte-for-byte as loaded. `ee_pg_easy_v2_lib.json` and `ee_pg_lib_frozen.json` are not edited.

The merged intents are single-axis world translations of five millimetres, one centimetre, and three centimetres on ±x, ±y, and ±z. The harness clamps each one to three centimetres per axis, servos it on the existing move path, and keeps the current gripper command. The next Jev state reports the servo result and the new end-effector position in `motion_context.last_fine_step` (`last_fine_step_text` is the same sentence).

Do not pass the overlay file alone: it is not a pick-and-place menu. Suggested ablation, same suite, tasks, and init seeds, library `libero_rsi/libs/ee_pg_easy_v2_lib.json`:

- Baseline: `FINE_STEPS` unset.
- Fine steps: `FINE_STEPS=1`.

The episode score is `success` in `result.json` (official LIBERO check). Success rate is the mean of `success` over seeds. Also record `decisions`, `jev_calls`, `gate_jev_calls`, `astra_calls` (perception and identity), and `planning_calls` (start plan and slow-think, including `ask_astra`). When the flag is on, `result.json` also has `fine_steps_enabled` and `fine_step_events`; both keys are omitted on a baseline run.

## Smoke tests

From the repo root, with Python 3:

```bash
pip install -r requirements.txt
python libero_rsi/smoke/run_all.py
```

`run_all.py` checks for NumPy, Pillow and SciPy first and names any that are
missing, then runs each `smoke/test_*.py` in its own process and prints the
output of failing tests (`-v` prints all output). It exits 1 if a test fails
and 2 if a package is missing. The video test is skipped without `ffmpeg`. A
single test can also be run directly, e.g. `python libero_rsi/smoke/test_stage_todo.py`.
No simulator and no network.

The regression suite executes selected production functions with environment/model doubles, covering identity authorization, fixed backend routing and Codex failure handling, event timestamps, async snapshots, diagnostic isolation, recovery failures, and validation of every shipped library. These tests do not establish simulation success.

`libs/smoke_copies/` contains duplicate library fixtures retained from the original snapshot; these are not independent policy variants or experimental results.

## Call-chain audit

See [the earlier call-chain audit](libero_rsi/reviews/call_chain_audit.md) for verified wiring, fixed regressions, and remaining environment requirements. `test_runtime_chain.py` executes the top-level runner with fake services; it does not test LIBERO physics or robot success.

`astra_calls` counts perception/identity replies; `planning_calls` counts start/slow-planning replies. `jev_calls` counts acting decisions, `gate_jev_calls` counts spectator decisions, and `todo_review_calls` counts separate Jev progress-review requests; `todo_review_model` records their model. Sum these three Jev counters for successful Jev requests. A review requests one Choice answer per todo item, not one HTTP request per item. Codex CLI does not provide measured usage in this adapter, so its zero-valued cost fields are placeholders, not evidence of free calls.

## Optional pose menu

`POSE_MENU=1` adds in-place yaw, side/top align and return-home actions. Use
`POSE_MENU_KINDS=rotate`, `align` or `home` (comma-separated) to select families,
or load `libs/ee_pg_pose_menu_lib.json`. These controls default off and preserve
the gripper command. Align uses current visual estimates, not simulator object
poses. Rotation/home require orientation convergence in the measured-motion
servo; ordinary translation keeps its positional stopping rule. Historical
outcomes appear in `motion_context.pose_action_outcome`. A failed home action
retains its attempted command overrides instead of claiming the home pose was
reached. Empty-hand fine/pose motion after release preserves placement review;
Jev still verifies final placement and whether another grasp is necessary.

## Recovery decision ownership

The existing recovery menu and motion/identity/grasp safeguards remain unchanged.
`JEV_RECOVERY_MENU=both` (default) retains current automatic recovery rules.
`JEV_RECOVERY_MENU=jev` skips library `stall_feedback` / `approach_change` rules
before they spend a use counter, move the arm or force an intent. It also disables
host-inserted stall steps/chooser calls when an Astra plan lacks a recovery step.
Explicit Astra plan steps still execute. Identity-deadlock escape, local-deadend
limits, no-progress diagnosis and grasp/release guards remain active, so this is
control of routine automatic rules, not an unguarded or exclusively Jev policy.
`motion_context.recovery_control` and `result.json.recovery_control` record the mode.
This PR adds no duplicate action library, outcome format or alternate observation path.

## Current-state execution descriptions

The acting criteria for `clear_reapproach`, `move_above_object`,
`descend_to_grasp` and `change_approach_yaw` include a description built from the
same decision-step namespace. It separates `program_steps`, `conditional_effects`
and `unverified_physics`. Move targets come from the loaded library expressions;
recovery offsets/yaw use calculations shared with execution. Commanded targets
are derived from visual estimates, never object ground truth or physics lookahead.
Descriptions include current repetition evidence, the existing forced approach
follow-up, and any pre-existing local-deadend interception. Requested displacement,
rotation and clearance are not promises of achieved motion or a successful grasp.
The candidate set/order, action guards and Jev's selection remain unchanged.
The exact descriptions and criteria are logged before the acting request. Other
candidates retain their existing definitions; there are no extra model calls or
simulation steps to produce these descriptions.
