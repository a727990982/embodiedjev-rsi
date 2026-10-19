# embodiedjev-rsi

Public snapshot of the LIBERO RSI / EmbodiedJev harness. This repo had no prior tree, so the agent module, episode runner, experience brief, libraries, logic smokes, and review notes all live under `libero_rsi/`.

| Path | Role |
|---|---|
| `libero_rsi/coding_agent_slow.py` | Codex CLI transport for start planning, slow-think, Astra perception and identity checks. |
| `libero_rsi/run_ep_fbpatch.py` | Episode harness: Jev intents, stall feedback, grasp confirm, local-deadend relook, place force-release. Needs LIBERO, MuJoCo, Codex CLI, and the Jev client to actually roll out. |
| `libero_rsi/jev_experience.py` | Builds library advice for Astra/Codex planning; cached predicates are not Jev input. |
| `libero_rsi/task_todo.py` | Task objectives, evidence-backed progress, and active object selection. |
| `libero_rsi/jev_observation.py` | Same-frame camera/proprioception input and applicable intent definitions. |
| `libero_rsi/jev_choice.py` | Bundled TypeSafe Jev 1.14 image-capable HTTP client. |
| `libero_rsi/validate.py` | Library validator. Imports sibling `dsl_libero.py`. |
| `libero_rsi/dsl_libero.py` | Expression DSL and dimensional checks used by the validator and logic smokes. |
| `libero_rsi/libs/*.json` | Experience libraries, including `ee_pg_easy_v2_lib.json`. |
| `libero_rsi/smoke/test_*.py` | Offline tests of validators, routing and harness logic; no simulator or network. |
| `libero_rsi/reviews/*.md` | Design notes for the behaviors below. |

## Todo and current-image decisions

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

Acting Jev additionally receives the last eight completed intent/result pairs,
plus the consecutive count of the same intent and the same intent/result pair.
Counts cover the full trailing streak even when it exceeds eight actions.
These are past executor outcomes, not assertions about the current scene or
task completion. The instructions explain stage progress and preserving a
confirmed grasp. Candidate and executor checks reject closes above/off the grasp
position, opening/reapproach motions that would discard a supported target grasp,
and release before the destination waypoint. An unchanged failed motion requires
a changed target/orientation or an executed back-off before retry.
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

Astra reviews todo progress after release, relook, retreat, help requests,
stall/timeout or empty-grasp reports, before the next decision. During execution, only current-image
evidence can mark an outcome done; official full-task success completes the
remaining todo items with an explicit official-success record. Occlusion preserves an older verified outcome
with its original timestamp; visible contradictory evidence can reopen it.
Changing the active item cancels pending slow advice, clears old grasp/approach
state and re-localizes and verifies the new target. Uncertain identity still
blocks closing. `finish` cannot end the task before official success.

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
| Episode-start plan/todo and visual todo review | Astra (`gpt-6-astra`) | Codex CLI |
| Sync/async slow-think | `CODING_AGENT_CODEX_MODEL` (default `gpt-6-astra`) | Codex CLI |
| Visual perception, keypoints and identity verification | Astra (`gpt-6-astra`) | Codex CLI |
| Per-step intent selection and spectator decision | TypeSafe `jev-1.14.0` | Bundled `jev_choice.choose`, with image assets |

Codex is the invocation mechanism; Astra is the visual model. Perception and marked-grasp identity verification both use `codex exec -m gpt-6-astra`. There is no dual/comparison mode or direct Astra-client transport. The former `CODING_AGENT_SLOW`, `EPISODE_START_PLAN`, and `CODING_AGENT_PERCEIVE` role switches are no longer read. The mock backend is removed from all helpers; unsupported backend values are rejected.

Early recovery refreshes perception with Astra through Codex CLI, then requests Codex planning. CLI failures do not switch to a direct Astra client or to mock results. The existing `ask_astra`, `consult_astra`, and `defer_to_astra` intent labels are retained for the shipped libraries and logs, but their help path invokes Codex planning.

Codex looks up `codex` on `PATH`, or `CODING_AGENT_CODEX_BIN`. Slow-think uses `CODING_AGENT_CODEX_MODEL` and `CODING_AGENT_CODEX_EFFORT`; start/todo calls pin Astra and use `CODING_AGENT_CODEX_EFFORT`. Perception/identity uses `gpt-6-astra` with `ASTRA_EFFORT` (default `xhigh`). A proxy is used only when `CODING_AGENT_HTTP_PROXY` or `CODING_AGENT_SOCKS` is set. Do not commit proxy URLs, keys, or host details.

Async slow-think captures a detached copy of policy state and a unique image before starting its worker. Workers return results and log events in private job boxes; the main thread writes the decision log. Simulator diagnostic files are never passed to the coding monitor, including when `DIAG` is enabled. Async advice is discarded after target revision, gripper state, close-attempt count, or identity confirmation changes. At shutdown pending calls are joined and logged without applying their plans.

## Running an episode

The bundled Jev client requires `TYPESAFE_API_KEY` in the process environment.
It pins `jev-1.14.0` and sends images using TypeSafe assets and the image opt-in
header. It does not load a similarly named external client or silently use a
text-only model. The [TypeSafe image protocol](https://docs.typesafe.ai/images)
is used for both acting and spectator decisions.

Real rollouts also require an authenticated Codex CLI, LIBERO with its assets,
MuJoCo/robosuite, SciPy, NumPy, Pillow, `ffmpeg` with libx264, and rendering dependencies. `astra_client`
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

A failed start plan stops the episode before Jev decisions and is recorded as `harness_error`. Libraries are validated before environment creation. Recovery-rule errors are logged and end the episode with `harness_error`, rather than silently continuing with potentially partial state changes.

## Smoke tests

From the repo root (Python 3; `numpy` and Pillow for the tests that import them):

```bash
failed=0
for t in libero_rsi/smoke/test_*.py; do
  python "$t" || failed=1
done
test "$failed" -eq 0
```

Each test adds `libero_rsi/` to `sys.path`, so `validate` and `dsl_libero` resolve as siblings. NumPy and Pillow are required for the tests that import them. No simulator and no network.

The regression suite executes selected production functions with environment/model doubles, covering identity authorization, fixed backend routing and Codex failure handling, event timestamps, async snapshots, diagnostic isolation, recovery failures, and validation of every shipped library. These tests do not establish simulation success.

`libs/smoke_copies/` contains duplicate library fixtures retained from the original snapshot; these are not independent policy variants or experimental results.

## Call-chain audit

See [the earlier call-chain audit](libero_rsi/reviews/call_chain_audit.md) for verified wiring, fixed regressions, and remaining environment requirements. `test_runtime_chain.py` executes the top-level runner with fake services; it does not test LIBERO physics or robot success.

`astra_calls` counts perception/identity and todo-review replies; `todo_review_calls` is also reported separately; `planning_calls` counts start/slow-planning replies. Codex CLI does not provide measured usage in this adapter, so its zero-valued cost fields are placeholders, not evidence of free calls.
