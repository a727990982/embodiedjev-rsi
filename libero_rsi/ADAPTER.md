# LIBERO harness runtime contract

Run from the repository root:

```bash
python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIBRARY.json OUTDIR [nofallback]
```

`SUITE` is a LIBERO benchmark registry key; `TID` and `INIT` select the task and initial state. The script only calls `harness.episode.main`; the harness modules can be imported (with `libero_rsi/` on `sys.path`) without LIBERO, which is loaded when an episode starts. One episode runs per process: shared state lives in `harness.state`.

## External dependencies

The rollout needs a working LIBERO installation (BDDL files, initial states and simulation assets), its compatible MuJoCo/robosuite stack, NumPy, SciPy, Pillow, and a working offscreen rendering backend. The script defaults to EGL. This repository does not pin or provide a validated simulator installation.

The repository includes the TypeSafe Jev client in `jev_choice.py`. Set
`TYPESAFE_API_KEY` in the runtime environment. The model is pinned to
`jev-1.14.0`; requests use `https://api.typesafe.ai/v1/systemone` with base64 image
assets and `TypeSafe-Danger-Zone-Images: true`. Keys are not stored in source or
request logs. This API returns choice probabilities, not raw model logits.

| Adapter API | Behavior |
|---|---|
| `jev_choice.choose(state, instructions, criteria, assets=..., meta=...)` | Returns `(choice, probabilities, latency)`; requires current image assets and rejects a wrong model or invalid candidate/probability reply. |

No external `jev_choice` or `astra_client` is loaded. Astra via Codex CLI writes
the episode-start todo and provides perception, identity verification and todo
progress review. Codex CLI also runs sync/async slow-think; Jev chooses per-step
intents. There is no dual mode, mock backend or text-only image fallback.

Codex requires its CLI and configured authentication. Configure the slow-think model and planning effort with `CODING_AGENT_CODEX_MODEL` and `CODING_AGENT_CODEX_EFFORT`, and the executable with `CODING_AGENT_CODEX_BIN`. Perception and marked-grasp identity verification explicitly use `gpt-6-astra` with `ASTRA_EFFORT` (default `xhigh`), independently of the slow-think model override. Start planning/todo and todo reviews also pin Astra; they use the planning effort. There is no mock backend. Offline tests inject fixed replies at the CLI boundary; they do not demonstrate visual capability.

For early recovery the harness abandons pending advice, refreshes keypoints/identity using Astra through Codex CLI, and calls synchronous Codex planning. A perception failure stops that recovery attempt; a Codex failure is recorded without invoking a separate client or substituting a mock result. Existing help-intent labels (`ask_astra`, `consult_astra`, `defer_to_astra`) keep their names in libraries/logs but route to Codex planning.

## Jev inputs and todo lifecycle

`_current_observation` captures front RGB, optional wrist RGB, end-effector
position/orientation, finger joint positions/opening, commanded open/closed state
and individual contact booleans from one simulation step. `_jev_input` adds the
language goal, todo and official success, encodes image assets and logs matching
frames. Contact flags never identify the touched object.

Acting and spectator Jev receive `motion_context`: robot-part contacts, last
commanded waypoint and actual displacement, remaining error, blocked intents,
help budgets, and a close-position check against the visual grasp estimate.
Robot contacts include fingers, palm/wrist and arm against the environment
(including fixtures), plus self-contact. They expose no other-body names, IDs,
poses or forces. Contact is not automatically classified as blockage.

The last failed motion includes a historical front/wrist before-image, separately
labelled from current camera assets; Astra's monitor receives the same temporal
pair. Startup obstacle advice is also historical. The startup overlay shows only
the proposed TCP route. Finger opening is not full hand width, and full arm
swept-volume collision checking is absent. Astra must report uncertainty for
occluded or otherwise unverified clearance.

Acting additionally receives the last eight completed intent/result pairs and
full trailing repetition counts, including forced high-level actions. These are
past outcomes, not current scene truth. Todo progress retains the sensor-refreshed
grasp flag. Cached visual held estimates, library predicates and simulator object
poses are not included in policy input.

Close requires XY and height proximity to the effective visual grasp estimate
(including episode offsets). Tolerances are 10–15mm scaled by estimated dimensions,
allowing controller/vision error while rejecting an above-target close. Position
is rechecked after any identity relook, including forced execution. Refusal returns
`close_not_executed:not_at_grasp` without closing.

A supported target grasp disables opening approach/retry motions. `open_gripper`
requires TCP XY within 4cm of the visual place point and height within 1.5cm of
its estimated release waypoint. Identity/contact loss permits recovery again.
This is estimated geometry, not verified object placement.

An unchanged failed waypoint is withheld (6mm target / 3 degree orientation
comparison), including forced calls. Changed geometry/orientation or successful
back-off permits retry. `back_off` preserves the grip; `clear_reapproach` and
`change_approach_yaw` reuse the existing executor when no current target hold is
supported. Exhausted relook/help choices are removed. General help uses MAX_SLOW;
one separately recorded motion diagnosis can run after exhaustion when fallback
is enabled and MAX_SLOW > 0. Failed requests spend the slot too;
`result.json.motion_diagnosis_used` records it.

Transport candidates (`lift`, `carry_to_place`, `lower_to_place`) require a
closed gripper, confirmed target identity, a successful grasp confirmation, and
current tactile support. An old `grasp_secured_once` or visual held estimate
alone cannot enable them. The executor enforces the same precondition for direct
and forced calls; it returns `<intent>_not_executed:grasp_unconfirmed` without
moving or closing when it fails. A failed grasp attempt returns
`grasp_failed:transport_not_executed`, not `closed`, even if the fingers moved.
`closed` is returned after a successful grasp confirmation. The explicit failures
are passed to acting Jev history and existing failure/todo-review handling.

Episode recording defaults to `episode.mp4` inside the episode output directory.
It requires ffmpeg/libx264, uses one encoder thread, and includes front/wrist
images and the actual horizon in the caption. `VIDEO` overrides the path; an
explicit empty `VIDEO` disables recording for offline tests. Encoder completion
is awaited before writing `result.json`, which includes `video_path` and
`video_exit_code` when recording is enabled.

The start planner must return `task_todo.items`, each with unique `id`, `target`,
`goal`, and `completion_condition`. All start as pending, regardless of a planner
claim of completion. Each item describes an object's desired final outcome,
not a fixed sequence of motor commands. The first non-done item is active.

`task_todo.initialize` adds six nested checkpoints to each object: identity,
secure grasp, lift clearance, transport, release height, and verified placement
with clearance. Planner-supplied checkpoint completions are ignored. The main
loop refreshes progress before selection and after execution/recovery. Identity
uses the existing visual gate; current grasp uses the same live precondition as
transport. Motion checkpoints require a completed `reached`/`at_target` result
with a supported grasp. They record achieved waypoints in this grasp, not current
object pose or collision-free-path certification. Lowering requires a previously
reached destination checkpoint. Loss of grasp, premature reopening or identity
loss revokes dependent checkpoints; localization/patch revision changes invalidate
motion waypoints. These operations do not invoke a model, retarget an object,
alter candidates, or force Jev's next intent. They are distinct from Jev's
final-outcome review and do not turn `pending` into a grasp-failure signal.

After a supported grasp and transport, actual `open_gripper` completion enters
`awaiting_placement_review`; it does not complete placement. Retreat/review may
follow without reacquiring the object. Visible placement failure resets progress;
occlusion retains the pending review. Verified final placement completes the last
checkpoint and marks unobserved intermediate checkpoints `not_required`, not
fabricated `done`. Step transitions never change top-level `active_id`.

For Jev, `decision_view` nests the old outcome evidence/timestamp under
`last_visual_review`; current stage progress has its own timestamp. Saved todo
artifacts retain the original review fields as well as stage progress.

For a pending/uncertain post-release outcome, Jev answers an additional Choice question mapped to `needs_regrasp`:
true for visible failed placement requiring another grasp, false when only
clearance/verification remains, or null if uncertain. Only true revokes the
placement attempt's progress. This prevents a successful release with the gripper
still near the rim from incorrectly restarting the grasp stages.

After each executed intent, including forced intents,
Jev reviews every item using a new same-frame image/sensor snapshot. A complete
review returns each id with `pending`, `done`, `uncertain` or `unobserved` and an
classification record linked to frames and probabilities. `unobserved` retains its previous status/evidence/timestamp;
it cannot create a new completion. Explicit contradictory evidence can reset a
done item. Malformed reviews stop the episode rather than fabricate progress.

The main thread owns todo updates. Only changed item statuses abandon outstanding slow advice.
A target switch additionally clears grasp/approach state, refreshes depth-based
controller keypoints for that specific target, and verifies the marked grasp
identity. A planner's identity assertion never substitutes for verification.
`task_todo.json`, decision events and the final result retain todo progress.
On official success, any remaining todo items are completed with explicit official-success evidence. Only official success ends the task successfully; visually done todos are not
an official success signal.

`reopen_and_retry` only opens in place. Subsequent alignment or relook is a
separate choice. Standard intent definitions in `jev_observation.py` describe
applicability using current images/sensors, not removed cached predicates.
Library DSL recoveries can still force actions; this change is not a removal of
the deterministic recovery controller.

## Diagnostics and asynchronous execution

`DIAG` writes privileged simulator state, including object poses, for offline analysis. These files and the final diagnostic result are not monitor inputs. The monitor receives only policy decision history, the captured policy state, experience, and captured images.

Each async job owns detached nested state, an image file, and a private result box. The worker never reads live `EP`, `PATCH`, `GOAL_TARGET` or decision files. Only the main thread applies results and writes decision log events. Job IDs reject abandoned jobs. Decision age, target revision, gripper command, close-attempt count, and identity state are checked before applying async advice. Shutdown joins active/orphan CLI calls and harvests their records without applying further actions. Each CLI subprocess has a timeout.

Libraries must pass `validate.py` before a simulator is created. A recovery exception is logged with its rule ID and terminates the episode as `harness_error`; there is no rollback for partially executed robot actions.

The monitor receives the initial plan, current todo/active goal, controller facts and episode lessons. Todo and active goal are detached with the asynchronous snapshot. A plan's `relook` step invokes visual perception; keypoints included in a planning reply trigger a visual refresh rather than replacing the perception base directly. Start-planning failures prevent entry into the Jev loop.

The adapter rejects nonzero CLI exit codes even if stdout contains JSON. Missing requested images, malformed/nonfinite geometry, and invalid plan adjustments are errors. Planning calls and visual calls are counted separately; zero cost from the CLI adapter is an unmeasured placeholder.

Todo review receives the canonical host todo separately from the decision view. Historical evidence survives unobserved outcomes, while previous verdicts and stage checkpoints are hidden from the classifier. Post-release regrasp questions share the same image request. Reviews run after each intent; only changed item statuses invalidate asynchronous planning.
