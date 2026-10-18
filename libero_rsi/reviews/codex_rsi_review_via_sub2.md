# LIBERO RSI stack: read-only code review

Review date: 2026-10-18. Current library: `ee_pg_v13`.

All five requested files were read in full. References below use approximate line numbers from this review. Supporting reads covered `jev_experience.py`, the relevant orchestration/schema/update code in `rsi_libero_v2.py`, `astra_client.py`, and `ADAPTER.md`. Backup comparisons covered goal ownership, pre-stall library/validator/DSL/playground behavior, and the pre-ask-Astra runner. No module was imported or executed, no model request was made, and no playground, validation, smoke test, or training job was started. The only file written is this review. Examples below are static control-flow traces, not observed test results.

Short names:

- **Runner**: `libero_rsi/run_ep_fbpatch.py`
- **Playground**: `rsi_playground_ee.py` (private bench; not in this snapshot)
- **Library**: `libero_rsi/libs/ee_pg_easy_v2_lib.json`
- **Validator**: `libero_rsi/validate.py`
- **DSL**: `libero_rsi/dsl_libero.py`
- **Experience**: `libero_rsi/jev_experience.py`
- **Orchestrator**: `rsi_libero_v2.py` (private bench; not in this snapshot)

## 1. Executive summary

- **No P0 finding is established by this static review. There are several P1 correctness defects.** Highest priorities are async result isolation, closure after identity/target refresh, and durable episode-to-library commits.
- **Async cancellation does not isolate the old worker.** Workers retain the same mutable `SLOW_JOB` dictionary after it is cleared and reused. Old advice can overwrite a new job's reply; even correctly associated replies are checked only against decision age, not target revision, grasp state, or physical movement.
- **A confirmed object identity is being treated as permission to close at the current TCP.** `execute()` can relook, move the target estimate to a different object, and immediately close without approaching the new target. Ordinary closes also lack executable geometric preconditions.
- **Goal ownership is only partly implemented.** `GOAL_TARGET['rev']` is recorded but not used to reject stale work. Recovery motion/patches are followed by execution against an old namespace. `RSHAPE` remains another persistent offset/pose layer outside the advertised reset contract.
- **Yaw preservation is a real improvement, but still has a deterministic held-object failure.** Fresh yaw is compared with old perception yaw rather than effective commanded yaw; held-object refresh can retain old `PER['yaw']` while clearing the effective patch. Orientation completion is also absent from `servo()` termination.
- **The playground can lose accepted learning on interruption and skip its reconstruction on resume.** It saves the working library before each episode, not immediately after reflection; the final update is written only to the frozen artifact. Resume relies on result existence rather than a committed reflection transaction or library hash.
- **Several requested improvements already exist.** `descent_stalled` uses `last_result`; lessons/recoveries reach both acting and spectator Jev through `experience_brief`; stall feedback offers multiple actions; `ASTRA_EFFORT` defaults to `xhigh` in the runner and playground. These are not missing features. Remaining problems concern semantics, freshness, enforcement, and evidence quality.
- **The current policy is still a single-object pick-and-place controller.** A generic language prompt does not supply multi-stage progress, articulated-joint motion, verified held-object transforms, richer placement goals, or a stage-aware budget.

## 2. Concrete bugs

Severity convention: **P1** = likely to corrupt control, learning, or an overnight run on a reachable path; **P2** = narrower correctness defect, missing enforcement, or materially misleading evidence. Latent defects in accepted future library entries are identified explicitly.

### B01 — P1: abandoned async workers can overwrite later jobs

**Location:** Runner `_slow_worker`, `_clear_slow_job`, `_launch_async_slow`, `_sync_slow`, approximately 1071–1154.

Evidence:

```python
slot = SLOW_JOB
th = threading.Thread(target=_slow_worker, args=(jpeg, prompt, meta, slot), daemon=True)
# worker:
slot['parsed'], slot['rec'] = parsed, rec
# cancellation:
_clear_slow_job()  # updates the same dict in place
```

`_sync_slow()` joins for only 0.05 seconds, clears the dictionary, and leaves the original worker alive. The orphan helper merely waits; it does not revoke the old worker's reference. A later async launch reuses that dictionary. If old worker A finishes after new worker B has populated its result but before B is polled, A can replace B's `parsed`/`rec`; the poll then pairs A's advice with B's `dep`, `snap_dec`, intent, and image. An old exception can similarly poison a later slot.

**Impact:** wrong-image keypoints or plans can be applied to the current episode; costs can be lost or attributed to the wrong job. This is a shared-state race, not merely a missing cost counter.

**Suggested correction:** allocate a distinct record for every request, attach an immutable job ID and cancellation state, and let workers write only their own record. Collect completion/accounting independently of whether a result remains eligible for control.

### B02 — P1: async freshness ignores state and target ownership

**Location:** Runner `_poll_async_slow` 1082–1107; main loop 1468–1570; `GOAL_TARGET` 218, 288, 301.

The sole freshness test is `stale = dec - snap_dec`, accepting `stale <= 1`. `GOAL_TARGET['rev']` is never captured or compared by async jobs. A single decision can contain many physical steps, a relook, closure, release, or an Astra intervention. At the top of the loop, `do_fallback()` can refresh keypoints and execute a plan without cancelling a pending async job. Later polling can overwrite that newer base with older keypoints within the one-decision window.

`_finish_slow()` also interprets snapshot `is_goal_object` against **current** `EP['closed']`. Decision age alone does not prove these refer to the same grasp episode.

**Suggested correction:** snapshot target revision, identity revision, gripper/grasp epoch, physical step, and relevant pose; invalidate at refresh/close/release/push/stage transitions. Require compatible state before applying, and avoid applying control replies during finalization after success or a terminal stop.

### B03 — P1: relook can unlock a close at the old target position

**Location:** Runner `execute` 756–783; `do_relook` 741–745.

When identity is unconfirmed, the close branch calls `do_relook()`. If the reply confirms a newly located target, execution falls through directly to:

```python
EP['closed'] = True; grip(1, 20)
```

There is no move to refreshed `grasp`, no fresh proximity/alignment check, and no invalidation of the pending close. A TCP over candidate A can therefore close over A after relook identifies candidate B. The logged close then reports `goal_id_confirmed: True` despite the physical mismatch.

More generally, `execute()` does not enforce the `close_gripper` definition's `at_grasp` condition. Any selected move with `fingers='close'` closes first and returns, including `lift` or `carry_to_place` selected while open. Text definitions are advisory, not preconditions.

**Suggested correction:** a changed target invalidates the pending manipulation. Recompute facts and return to approach. Require fresh target identity plus suitable geometry/orientation before a close; use capture evidence for subsequent lifting. Do not require bilateral contact for every initial close, since a correctly positioned open gripper may not touch until it closes.

### B04 — P1: slow plans and fallback disagree about identity and release

**Location:** Runner `_apply_episode_plan` 881–913; `_finish_slow` 1017–1043; `do_fallback` 1387–1403, 1443–1446.

The slow path distinguishes an open gripper with `is_goal_object=null` from an incorrect target when `matched_language_goal=true`. Fallback instead executes:

```python
if failure_class == 'wrong_object' or igo_n is not True:
    ... set_goal_id(False, ...)
else:
    set_goal_id(True, ...)
if EP['closed']:
    EP['closed'] = False; grip(-1, 15)
```

Consequences:

- An open, correctly identified target with nothing held can be relabeled `wrong_object`, clearing a useful pose and revoking identity.
- Conversely, `is_goal_object=true` can set the close gate true even when `matched_language_goal=false`; fallback does not reconcile held identity with target/place identity.
- Every fallback opens a closed gripper, including a correctly held goal object during carry or a placement-path stall. It may drop the object before a recovery is selected.
- Keypoints are committed **before** this release, with `held_now=True` potentially preserving the old grasp base. The subsequent physical state no longer matches that held-object interpretation.
- Fallback logs `matched_language_goal` from `is_goal_object`, erasing the distinction in reflection evidence.

**Suggested correction:** share a single plan application path with explicit target, held-object, and place identity fields. Preserve a verified grasp during path/placement repair. Release only as an explicit, stage-appropriate step.

### B05 — P1: recovery mutates the robot/target but execution uses the old namespace

**Location:** Runner main loop 1474–1525; `_apply_stall_feedback` 538–624; `execute` 786–795.

The main loop computes `ns, preds = namespace(); f = facts(ns, preds)` **before** running recoveries. A recovery can raise the TCP, change yaw, and change `PATCH['grasp_dz_objh']` or XY offsets. It then forces an intent, but `execute(it, ns)` receives the pre-recovery namespace.

For a stall recovery forcing `move_above_object`, `refpoint` is evaluated using the old effective grasp, not the just-patched one. Learned shape changes have the same issue. The logged `facts=f` also describe the state before recovery motion, so the executed action and evidence diverge.

The spectator is called after execution with the same `ns, f` (1558), mixing new result/history with old gripper/contact/target facts.

**Suggested correction:** rederive namespace, predicates, and facts after every intervention and before selection/execution. Preserve separate labeled before/after snapshots in logs rather than reusing one snapshot across mutations.

### B06 — P1: accepted learning is not durably committed, and resume skips it

**Location:** Playground `main` 107–155; Orchestrator `run_one` 33–44.

The working library is saved at 112 and 117, before an episode. After `lib, ok, bad = apply(...)` at 147, only logs/history are written; the library is not saved until the next loop iteration. If the process stops between those points, accepted edits disappear from `LIBP`. If it stops after the episode result but before reflection, reflection has not happened at all. Both cases are skipped on restart by `done.exists()` at 120–121; the log is not replayed and no reflection-complete marker is checked.

The final episode's edits go only into `ee_pg_easy_v2_frozen.json` (154), leaving the named current working library behind. Resume also rewrites version labels starting at `ee_pg_v0` and mixes old results with the current library without a content-hash match. These are correctness/provenance problems even if all original episodes were successful.

**Suggested correction:** persist accepted edits immediately with an atomic library commit and an episode/reflection commit record containing input/output hashes. Resume independently at episode execution, reflection, and committed update stages. Keep current and frozen artifact provenance explicit.

### B07 — P1: an episode subprocess timeout can abort the whole playground

**Location:** Orchestrator `run_one` 36–44; Playground `main` 122–126.

`subprocess.run(..., timeout=2400)` is not wrapped in a `TimeoutExpired` handler. `B.run_one(...)` and `B.summarize(...)` occur before the playground's reflection `try`. A timeout therefore exits the overnight loop, bypassing remaining tasks and final freeze. Start perception and environment setup also occur outside the runner's main exception-to-result handler, so a crash can leave no ordinary result.

There is a related resume mismatch: Playground treats `end_reason='crash'` as rerunnable, while `run_one()`'s existing-result shortcut only excludes `harness_error`, returning a cached crash unchanged if that JSON exists.

**Suggested correction:** handle process timeout/failure at the per-episode boundary, record a structured failure, preserve logs, and use one retry/cache policy in both layers. This review did not exercise the timeout path.

### B08 — P1: held-object relook can deterministically wipe effective yaw

**Location:** Runner `_new_keypoints_explicit_yaw` 238–245; `_restore_approach_pose` 247–260; `_commit_keypoints` 262–276; `set_perc` 308–317.

`_new_keypoints_explicit_yaw()` compares new yaw to **old PER yaw**, not the effective commanded yaw. It treats an ordinary numeric difference of at least 15 degrees as explicit evidence, then drops all patched yaw/tilt fields.

Static example: `PER['yaw']=0`, effective `PATCH['yaw_deg']=45`, gripper closed and held; relook returns yaw 45 and `object_held_by_gripper=true`. `set_perc(..., held_now=True)` skips updating `PER['yaw']`. The explicit-yaw test sees `abs(45-0)>=15`, clears the patch, and does not restore it. Effective yaw becomes 0, although perception repeated the current 45-degree approach.

Even when not held, repeating the effective yaw can unnecessarily discard a useful tilt. The difference test also lacks angle wrapping or parallel-gripper symmetry handling. There is no stored evidence that the old patch was actually successful despite comments calling it “confirmed.”

**Suggested correction:** compare against effective approach pose; distinguish identity change, keypoint refresh, and explicit pose replacement. Update or preserve pose atomically, including the held-object branch. Use angular distance and an explicit change decision, not difference from an obsolete base as a proxy for evidence.

### B09 — P2: `RSHAPE` bypasses the single-owner/reset contract

**Location:** Runner 219, `_reset_patch` 227–236, `namespace` 410–412, `servo` 438–440, recovery application 1499–1509.

The effective grasp Z is `(PATCH['grasp_dz_objh'] + RSHAPE['dz']) * obj_h`. Yaw priority is `PATCH`, then `RSHAPE`, then intent/perception. `_reset_patch()` never resets `RSHAPE`, including on wrong identity. Thus an old conditional height adjustment survives a new base/identity, and an old learned yaw can override an explicitly refreshed perception yaw after PATCH is cleared. `yaw_delta_deg` even stores the value in both layers.

**Scope:** the current five recoveries do not have `shape` entries, so this is latent for accepted learned shape recoveries, not proof that today's library always triggers it.

**Suggested correction:** bind every override to a target/stage revision with explicit persistence rules. A reset must invalidate all target-specific layers; a fresh target should not inherit geometry merely because it came from a library rule.

### B10 — P2: a `relook` plan step can do nothing while being reported as done

**Location:** Runner `_apply_episode_plan` 905–956; `do_fallback` 1404–1439.

In `_apply_episode_plan`, `{'step':'relook'}` only appends `'relook'` to `done_steps`. Without keypoints, it does not call `do_relook()` or queue it. The comment that “a later do_relook may run” has no corresponding scheduling here. A wrong-identity reply with missing keypoints can therefore consume a slow intervention and report relook without obtaining a new observation.

`has_kp = any(...)` is also weaker than a valid complete keypoint update. Commit exceptions are swallowed, but a later relook step with `has_kp=True` marks `target_updated=True` anyway. An empty `adjust` similarly counts as an update. In fallback, `target_updated=True` and `has_kp=True` are initialized unconditionally after its initial commit, making the forced-update guard unreachable in normal execution. Identical keypoints can satisfy the claimed “must change target” requirement.

**Suggested correction:** distinguish a requested observation, a validated refresh, and an effective control change. Report an operation as completed only when it occurred. Enforce measurable target/trajectory change or a genuinely new observation, with bounded retries.

### B11 — P2: keypoint updates are non-atomic and incompletely validated

**Location:** Runner `bp` 154–156, `set_perc` 308–317, `_commit_keypoints` 266–276, `_apply_episode_plan` 907–913.

`set_perc()` writes `seen_held` and `place` before accessing required grasp fields and other values. A later malformed/missing field can leave a partially changed PER while `_commit_keypoints` never reaches its patch reset/revision update. Several callers swallow that exception. `bp()` clips the depth-sampling pixel but uses the original unbounded `uv` in the projection ray, so out-of-frame coordinates combine edge depth with an out-of-frame ray. Numeric finiteness and complete payload shape are not checked before mutation.

**Suggested correction:** validate and normalize a candidate perception record, project into a temporary structure, then commit all fields and revision together. Reject incomplete/non-finite geometry rather than leaving mixed old/new state.

### B12 — P2: orientation-only changes are never required to complete

**Location:** Runner `execute` 791–795; `servo` 436–449; `_apply_relative_approach_change` 529–533.

`execute()` returns `at_target` whenever translation error is below 6 mm, regardless of yaw/tilt error. `servo()` computes rotational control but returns `reached` based only on the pre-step translation error; its stall detector also measures only translation. A new yaw at an already reached XYZ can therefore receive no servo step, or be declared reached before orientation converges.

In addition, `_apply_relative_approach_change()` sets the new yaw **before** `_raise_tcp()`, so its “raise then rotate” description is inaccurate: the raise servo rotates while lifting out of the obstruction.

**Suggested correction:** require position and orientation convergence for pose goals; separate clearance motion with the old pose from rotation/reapproach where necessary. Record orientation failure separately from translation stall.

### B13 — P2: contact-directed clearing is expressed in the wrong frame

**Location:** Runner `_apply_stall_feedback`, `clear_reapproach`, 598–611.

Left-pad contact produces `_nudge_grasp_xy(0.4, 0.0)` and right-pad contact produces its negative. Those offsets are added to world X/Y in `namespace()`. Finger left/right rotates with end-effector yaw; it is not a fixed world-X direction. After an approach rotation, the purported motion “away from left pad” can be sideways to, or toward, the obstruction.

`lateral_recenter` also normalizes the TCP-to-object-center direction into a fixed width-scaled step; it does not cap the shift by the remaining offset from the current **grasp target** to the center. It can overshoot or erase an intentional rim grasp.

**Suggested correction:** transform contact-side retreat through measured gripper orientation and project onto the support plane. Use target geometry and bounded displacement, with a separate clearance phase; preserve intended grasp features such as rims.

### B14 — P2: deadlock handling still leaves exhausted no-op actions selectable

**Location:** Runner `execute` 754, 761–777; `do_ask_astra` 1166–1169; main decision menu 1513–1524; `_escape_identity_deadlock` 652–678.

Once identity is unconfirmed and relook budget is gone, the menu excludes only the name `close_gripper`. It explicitly recommends `relook` and `ask_astra`, even though those can return `relook_budget_exhausted` or `ask_astra_cap` without moving. The escape path is attached to a refused **close** or fallback; choosing exhausted relook/help does not directly invoke it. Repetition fallback can rescue some sequences, but is itself limited by `MAX_FB`.

Other fingers-close intents remain available, and forced library closes bypass menu filtering (though they still reach the execute gate). After `MAX_ESCAPE`, repeated raise-only behavior and forced retreat have no informative terminal result and can continue toward horizon/decision cap.

**Suggested correction:** filter by action capability, budget, and current preconditions rather than one intent name. Route identity-budget exhaustion to a bounded evidence-seeking escape or explicit unresolved-identity termination. Do not confuse refused closure with empty capture.

### B15 — P2: validator accepts recovery controls that runtime ignores

**Location:** Validator 24–34; Runner recovery dispatch 1486–1511.

- Validator permits `shape.grasp_dxy_objw` and explicitly skips its checks with “applied in harness.” The runner's learned-shape dispatch never applies this key. A valid lateral recovery is silently ineffective. Current library recoveries do not yet contain this field.
- `stall_feedback` executes and immediately `break`s before `shape` or `approach_change` processing. Validator and prompts allow combinations (“and/or”), but a recovery containing both does not execute the full accepted specification.
- Presence of `shape`, including `{}`, satisfies `has_ctrl`; no actual control action is required. Duplicate IDs/names and unreasonable `max_uses` are not rejected. Runtime exceptions are swallowed after a use may already be consumed.

**Suggested correction:** define one supported recovery contract, validate it strictly, and either implement composition in a documented order or reject conflicting combinations. Reject accepted-but-unused fields and empty controls.

### B16 — P1: dimensional validation can admit hard-coded lengths

**Location:** DSL `dim` 82–94; Validator 11–15, 24–34; Playground `apply` 103.

`dim()` computes argument dimensions for distance functions but returns `'L'` without validating those dimensions:

```python
if f in ('dist', 'dist_xy', 'max_abs_xy'): return 'L'
```

Static counterexample: `dist(1, 0)` is classified as a length even though both arguments are numeric constants. Runtime `dist` converts both scalars to arrays and takes their norm, producing 1. Consequently a refpoint such as `vec(dist(1, 0), 0, 0)` has an accepted length dimension while introducing a fixed one-unit coordinate. This bypasses the playground's only machine gate for its central no-hard-coded-length rule.

The same checker does not distinguish scalar length from vector length, validate function arity consistently, require boolean operands, or validate indexing shape. An expression such as scalar `opening` can pass the refpoint dimensional check yet fail runtime `.reshape(3)`.

**Suggested correction:** track scalar/vector shape along with units, check each function's signature, and require observed provenance for dimensional values. Add explicit semantic checks at the library boundary. These examples were reasoned from code, not run through the validator.

### B17 — P2: new recovery results are incompletely represented in reflection evidence

**Location:** Orchestrator `summarize` 59–79; Runner logging 1525–1533; Playground prompts 63–83.

The summary preserves only the first 25 key-frame events, collapses repeated-intent facts, and does not directly include standalone `stall_feedback`, goal-commit, approach-preservation, escape, or shape-rule logs. It also omits the result's rich `start_identity`, `identity_events`, `goal_target`, and `close_refusals` fields. Some details survive inside slow events, but library-fired stall feedback lacks equivalent coverage.

Facts attached to an intent are from before execution—and sometimes before recovery (B05)—while `result` is afterward. The older Orchestrator `make_prompt` explicitly labels this timing, but Playground's own reflection prompt does not. The learner can mistake pre-close opening/contact for the outcome of that close, and late placement evidence may be cut off.

**Suggested correction:** retain compact event-linked before/after state, the actual recovery and target revision, refusal reasons, and terminal placement outcomes. Summarize by causal event/stage, not the first N log entries.

### B18 — P2: “clean success” and “verified” do not guarantee the advertised evidence

**Location:** Playground `needs_full_video` 30–43, prompts 45–83, `apply` 89–103; Experience 89–103.

`needs_full_video()` checks slow count, fallback events/lessons, and close refusals, but not direct library stall-feedback or escape counters. A successful episode repaired by a library stall recovery can be sent to a prompt claiming “SUCCEEDED cleanly (no slow-think/fallback intervention)” without its recovery sequence being explicit in the summary. Conversely `slow_interventions` includes help calls that returned `ok`, so it is not solely an intervention count.

For new entries, `apply()` requires a valid status and merely truthy evidence. It does not enforce a list of episode IDs or two distinct reproductions for `verified`. Revisions do not require new evidence, and predicate status validation is weaker than lessons/recoveries. A recurrent failure observation can therefore be labeled verified while its proposed remedy remains untested; Experience gives that status a ranking bonus.

**Suggested correction:** distinguish verified observation from verified recovery effect; retain explicit intervention provenance and validate evidence references/types. Keep untested recovery advice marked as such even when the underlying failure repeats.

## 3. Risky interactions

### Relook versus goal/patch ownership

The recent `_commit_keypoints()` abstraction improves consistency: ordinary relook, escape look, slow keypoints, and fallback keypoints now pass through one base-update function, and position PATCH is cleared on refresh. However, precedence is currently “last writer wins,” not the advertised semantic priority. Async snapshots carry no ownership revision (B02); RSHAPE survives (B09); partial updates can escape the transaction (B11); and execution can retain an old `ns` (B05).

Relook while holding deserves a separate contract. `set_perc(..., held_now=True)` intentionally preserves grasp/object geometry, but `_commit_keypoints()` still clears position PATCH. The carry formula uses `grasp - objc`; removing the offset used for the successful grasp changes the inferred object-to-TCP relationship. A place-only refresh should preserve the measured grasp transform, not recompute it from a mutable approach target. If vision falsely says the held object is not held, grasp/center can instead be recomputed from the airborne object, contaminating placement height calculations.

Identity consistency also weakens after startup. Startup requires both `matched_language_goal` and marked-pixel verification. Ordinary and escape relooks require only `identity_ok(p)`, with no independent mark verification or prior-image reconciliation. The stronger startup check can be overturned by a later occluded self-assertion. This is a risk of inconsistent evidence standards, not proof that every later perception is wrong.

### Yaw preservation versus yaw execution

The pre-goal-ownership comparison confirms that pose preservation and relook approach context were added. Current relook/escape prompts include effective yaw/tilt and ask Astra not to revert to defaults. These are useful changes. B08 shows why prompt language alone is insufficient.

Slow context, by contrast, does not include the current effective yaw, tilt, or full target revision/patch (993–1008), while it still requests fresh keypoints and adjustments. Fallback includes PATCH but duplicates plan logic. Different advice paths thus make pose decisions from different context.

Even a correctly preserved command is not a successfully executed pose: B12 allows a reached XYZ to terminate before yaw convergence. Pose evidence should separately record requested, commanded, achieved, and capture-associated orientation.

### Stall recovery versus the empty-relook ban

The ban is narrower than its name suggests. A fresh, valid observation after object motion is useful; an unchanged keypoint payload merely satisfying `has_kp` is not necessarily progress. The current guards conflate the two, and ordinary library `relook` is not subject to the plan guard. `adjust` with no effective change also bypasses the forced feedback path (B10).

Hard stall handling runs `_sync_slow()` immediately after a failed intent. The original `last_result`, `hist`, and result history still describe that failure afterward. On the next loop, the library may fire a second recovery or fallback may fire from unchanged history, stacking interventions before the first is evaluated. A slow plan's forced next intent sometimes suppresses a library recovery, but there is no general recovery-epoch/debounce mechanism. Conversely `_stage_fail_grip_unchanged(consume=True)` is shared by slow and fallback, so one path can consume escalation evidence before the other sees it.

The menu has five named options, but its selection is partly round-robin. Nearly every non-close branch increases target height; repeated failure eventually saturates positive `grasp_dz_objh`, potentially moving the target above useful capture depth. `_raise_tcp()` suppresses exceptions and ignores its servo result, while feedback can still be reported as completed. A useful trigger/principle recovery must verify clearance, changed approach, and subsequent capture—not count the selected menu item as success.

### Identity gate versus close-gripper deadlock

The new escape path prevents some former frozen-close loops, but it still conflates two different problems: uncertainty about *which object* and inability to *reach a confirmed object*. `_escape_identity_deadlock()` applies geometric stall feedback before looking even when identity is unconfirmed. Such offsets are not evidence that the candidate is correct, and the subsequent keypoint refresh clears those offsets anyway.

The comment “never force another immediate close” is not an invariant: `_apply_stall_feedback(..., confirm_contact)` can queue `close_gripper` when both pads touch, and `_escape_identity_deadlock()` retains a nonempty queued intent. Execution still enforces the identity gate, but the intended escape ordering is not guaranteed. B03 and B14 are the more serious closure and availability failures.

Acting Jev's regular `facts()` omits `goal_id_confirmed`, close-refusal count, remaining slow/relook/escape budgets, and target revision; `namespace()` has `n_relook`, but `facts()` does not expose it. The spectator gets some identity/budget fields, though potentially stale facts. Asking Jev to avoid exhausted or unconfirmed actions while hiding their availability makes the deadlock menu less reliable.

### Async slow-think versus synchronous fallback and `xhigh`

All runner Astra call paths read `ASTRA_EFFORT`, default `xhigh`; playground reflection does likewise. The local client passes that string through. No claim is made here about remote-model support or measured latency; neither was queried.

Higher-cost/longer-latency calls would magnify existing overlap problems: sync critical checks ignore the normal async time gap, abandon in-flight work, and fallback is a separate sync path without equivalent invalidation. `MAX_SLOW` bounds interventions and selected help calls, **not all ordinary watches**: an ordinary sync/async `ok` does not increment `P['slow']`. Thus “shares MAX_SLOW” is not a global request-count cap. Start verification, relooks, fallback, and escape looks have separate accounting.

There is also a local client budget risk: `astra_client._reserve()` (30–38) checks completed ledger spend while holding a lock, but records no outstanding reservation. Concurrent requests can both pass the check before either completes. Orphan requests can be missing from episode totals even if the shared ledger eventually records them. Keep episode control budgets, request reservations, and completed spend distinct.

## 4. Experience-library gaps

### What is already fixed

The current `descent_stalled` expression (Library 226–232) includes:

```python
prev == 'descend_to_grasp' and not closed_cmd and not contact
and dist(tcp, grasp) < 1.5*obj_h
and (n_same >= 2 or last_result == 'stalled' or last_result == 'timeout')
```

The older backup required repeated descent and a smaller distance. `last_result` is now present in DSL `STR_VARS`, runner namespace, and facts. It is inaccurate to report the former `n_same=1` mismatch as universally unfixed.

Lessons are also now delivered: both acting choice sites call `JX.jev_choose_state`, and the spectator calls `build_jev_experience_brief`. The latter provides active recoveries, a short catalog, up to five selected lessons for acting Jev, and episode reflections. The pre-ask-Astra backup already contains this connection.

### Remaining predicate and observability gaps

- **Descent failure still has blind spots.** A first stall outside `1.5*obj_h`, a stall in `move_above_object`, or alternating approach/descent failures does not satisfy the library predicate. The repeated-contactless fallback predicate still requires `n_same>=2`. Hard Astra may cover these while budget remains; the executable library does not fully cover them afterward.
- **Evidence is overwritten by support actions.** `last_result` is one string. A relook, help call, or successful raising action can replace the failure result before a learned trigger evaluates. Retain the unresolved failure event until a recovery outcome resolves it.
- **No identity/availability predicates.** DSL state does not include confirmed target identity, refused-close count, slow budget, escape budget, or stage. The learner is told to write recoveries for identity refusal after budget exhaustion, but cannot directly express that general trigger with the current exposed vocabulary. String matching against `last_result` is brittle and loses history.
- **No held-state predicate with freshness.** `lifted` means closed command plus height, not capture. `seen_held` persists from perception without a timestamp; `empty_close` uses opening recorded at the previous close, not a general slip detector. There is no held-loss-after-lift or empty-carry recovery.
- **Bilateral contact is not object identity.** `contacts()` only reports pad booleans; they can refer to different nearby bodies or an articulated fixture. The policy correctly does not see simulator body identity. Any “secure grasp” rule needs temporal/visual/proprioceptive corroboration rather than silently strengthening what those booleans mean.
- **No executable placement completion vocabulary.** `above_place` measures estimated XY alignment. There is no `at_place`, verified support/release-height state, placement-miss predicate, or bounded settle/reobserve/retry recovery.

### Lessons are often useful observations without executable remedies

The five current recoveries all concern approach, empty closure, or relaxed closure. There are no corresponding library recoveries for wrong-target identity persistence, refused-close exhaustion, dropped object, wrong place, high release, or articulated interaction. `hp_place_target_check`, `hp_wrong_object`, and `hp_wait_official_success` remain prose advice; the latter partially overlaps the runner's final settle loop, but does not create a repair strategy if settling fails.

`hp_a` and `hp_b` still say to look again after contact/empty close, while current recoveries prioritize physical feedback and the slow prompt discourages relook-only plans. Resolve this into an explicit principle: clear safely when blocked; refresh when evidence suggests the object moved or identity changed; require an outcome check before repeating. Neither an unconditional relook nor an unconditional height increase captures that principle.

`hp_start_identity` asks to reconcile earlier informative views, but later perception receives current images and limited textual context, not an explicit identity-evidence memory. `hp_c` and `hp_push_not_goal` suggest moving blockers; `push_aside` exists only as an episode plan step, not an ordinary library intent or typed recovery parameterization. The advice is not directly executable by Jev alone.

The brief is connected but can cut away the action: Experience `_lesson_line` truncates `instead` to 110 characters, often mid-sentence. For example, the long `hp_contact_not_held` advice includes the useful lift/refresh decision later in the paragraph; truncation can omit that branch. Retrieval is keyword-based, includes a status bonus even without a state match, and keeps the first three episode reflections rather than the most recent/relevant ones. Store a concise executable trigger and complete short instruction separately from explanatory evidence.

Finally, prompt/schema drift remains: `B.SCHEMA` still says max three relooks, exposes only `prev` as a string, and describes move-with-close as closing then servoing, whereas the current runner permits four ordinary relooks, exposes `last_result`, and returns immediately after a newly issued close. Update the learner's contract with the actual implementation.

## 5. Long-horizon gaps

### Multi-stage goals and multiple objects

`LANG` is fixed; `PER` holds one `grasp`, one `objc`, and one `place`; the prompt asks for one object to move and one destination. There is no ordered subgoal representation, stage completion state, active-object transition, or reset policy for per-stage `n_close`, releases, recovery uses, and identity. Even a perfect Jev choice cannot represent “open container, place object, close container” through the current intent menu and state alone.

Add a stage representation grounded in language and policy-visible observations, explicit completion/transition checks, and target-bound memory. Global official success should remain the final arbiter; stage completion must not pretend to be simulator ground truth.

### Drawer and microwave joints

Current motion is point-target Cartesian servo with a mostly top-down approach, limited tilt, and binary fingers. There are no perceived handle frames, prismatic pull directions, revolute hinge arcs, joint-progress observations, or contact-maintaining articulation primitives. Articulated fixture contacts can enter the same generic pad booleans as grasp contact. These semantics need separate stages and feedback.

The Orchestrator's TRAIN includes a spatial task annotated “bowl in top drawer”; that annotation alone is not evidence that the task requires actuating the drawer. The narrower conclusion is that the available controller cannot generally execute opening/closing drawers or microwaves merely by changing its place pixel. Do not infer successful articulation coverage from a task label.

### Placement beyond basket/plate centers

Perception produces a single place pixel and scalar clearance. Carry/lower formulas assume a fixed translation from grasp to object center and an object-height estimate. They do not represent support-region boundaries, aperture clearance, target orientation, shelf entry paths, insertion direction, stacking stability, relative spatial predicates, or constrained regions behind doors.

Even basket/plate placement has a current limitation: `grasp - objc` is recomputed from mutable target estimates rather than a stored transform captured at successful grasp. Yaw or relook changes can invalidate the estimated carried-object center. Release-height reasoning should use the held-object geometry and support surface, with a post-release check and recovery budget.

### Horizon and wall-clock budgets

Defaults are 600 environment steps and 60 decisions; ordinary servo can consume 60 steps, `_raise_tcp` 40, and `clear_reapproach` can invoke two raises before the next intent. `push_aside` contains several servo segments plus closing/opening. There is no per-stage allocation, reserved placement/settle budget, or admission check that a recovery fits the remaining horizon. Motion functions usually stop at horizon, but callers can still queue intents and report logical recovery completion after no physical work remained.

At decision cap, the runner spends the remaining horizon on fixed-command settling. This is useful after release, but cannot recover a robot still stalled in approach. Slow/relook/reflection calls consume wall time independently of environment steps, and the outer 2400-second timeout is a separate limit. Use stage progress, bounded recovery attempts, and separate simulation-step, decision, request, and wall-time budgets; do not simply raise every cap.

### Learning over an overnight sequence

The loop updates immediately and freezes without retesting accepted behavioral changes; this is explicit in its prompt and not a hidden defect. It means a frozen hash proves artifact identity, not generalization or recovery efficacy. A summary combining old cached episodes and later policies is not a frozen-policy evaluation. Label the sequential learning history accordingly and preserve enough provenance for a separately authorized evaluation later.

Video reflection is sampled evidence: Playground requests ten frames at maximum side 640, and the local client's default `call_video` implementation samples evenly spaced frames, optionally adding the start still. It does not supply continuous motion by default. Brief contact/closure events can fall between samples; the final terminal image is not guaranteed by midpoint sampling. Event-aligned pre/post-contact, release, and terminal frames would better support causal lessons. When video is missing, the same prompt still claims chronological frames despite only a start image being supplied; tell the learner what evidence is actually available.

## 6. Prioritized fix list

These are proposed changes only; none were implemented or run during this review. New state concepts below are design suggestions, not claims that such APIs already exist.

| Order | Fix | Expected impact | Suggested approach |
|---|---|---|---|
| 1 | Isolate async jobs and gate application by revision/state (B01–B02) | Prevent wrong-image plans from overriding current control | Unique immutable snapshots and per-job output slots; cancel eligibility on target/grasp/stage change; account every completed request independently. |
| 2 | Make close authorization depend on the current target and current geometry (B03) | Prevent wrong-location and empty closes after corrected perception | Invalidate pending close after refresh; reacquire alignment/orientation; enforce executable preconditions and distinguish initial capture from later hold verification. |
| 3 | Unify slow/fallback identity and release semantics (B04, B10–B11) | Avoid false identity resets and dropping valid grasps | One validated transactional plan path; separate target/held/place identity; preserve holds during transport repair; execute or explicitly defer requested observations. |
| 4 | Recompute facts after interventions and track recovery outcomes (B05) | Make feedback affect the next action and make logs interpretable | Refresh namespace before execution/spectator; track unresolved failure and recovery epochs; avoid double recovery before evaluating the first attempt. |
| 5 | Commit playground learning durably and repair resume boundaries (B06–B07) | Preserve overnight work and prevent whole-run loss on one timeout | Atomic post-reflection commits with input/output hashes; independent execution/reflection checkpoints; structured timeout/crash handling and unified retry policy. |
| 6 | Unify effective pose/offset ownership and verify orientation (B08–B09, B12) | Preserve useful approach geometry across refreshes | Compare against effective pose; explicit target/stage ownership; atomic held-object pose updates; retain successful grasp transform; position-plus-orientation termination. |
| 7 | Strengthen validator/runtime parity (B15–B16) | Keep learned controls both executable and within declared constraints | Unit-and-shape typing, function signature checks, two-element dimensionless XY validation plus runtime support, unique IDs, nonempty controls, explicit composition rules. |
| 8 | Expose action availability and bound identity escapes (B14) | Replace capped no-op loops with useful evidence or clear failure | Make budgets/identity/refusal state visible; filter exhausted actions; trigger view improvement for unresolved identity; stop when evidence routes are exhausted. |
| 9 | Refine stall recoveries into trigger + principle + outcome | Improve transfer without fixed angles or object-specific rules | Unilateral obstruction → clear along measured gripper geometry; displaced-object evidence → refresh alignment; empty capture → verify/recenter; no-progress after a changed approach → choose a different intervention or escalate. Check outcome before repeating. |
| 10 | Add hold-loss and placement recoveries with fresh evidence | Improve completion after the first successful grasp | Track capture/hold confidence and age, detect slip, verify target region before release, assess support/settling, and reserve a bounded correction attempt. |
| 11 | Improve reflection provenance and concise actionable lessons (B17–B18) | Reduce incorrect causal learning and prose-only growth | Event-linked before/after evidence and final frames; separate observation status from remedy efficacy; complete brief instructions; actual evidence-availability labels; current schema documentation. |
| 12 | Add staged goals/articulation and stage budgets | Enable tasks beyond single pick-and-place | Explicit subgoals, handle/path/region representations, contact-maintaining motion, stage-bound identity/patches, and progress-based horizon allocation. |

For recovery design, prefer a general failure trigger plus a geometry-based principle and a measurable exit condition. Avoid universal absolute yaw, object names, a monotonic “always shallower” rule, or merely converting every failure into another vision call.

## 7. Suggested smoke tests

**Not run.** Start with deterministic mocked perception/control and isolated pure helpers; importing the current runner directly starts episode setup and is unsuitable for a harmless unit-test import. Simulator smoke episodes should only be run under separate authorization.

1. **Async ownership:** delay worker A, abandon it for sync, start/finish B, then finish A before polling. Only B's own reply may be applied; both requests must be accounted for.
2. **Async invalidation:** finish a reply within one decision but after relook, close, release, push, or target-revision change. It must not mutate the new state.
3. **Relook-before-close:** TCP is over A; relook confirms target B elsewhere. Assert no close is issued until reapproach/alignment to B completes.
4. **Fallback identity/hold matrix:** cover open+no held object+matched target; closed+verified goal object+path stall; wrong held object; and held identity true but target/place identity false. Check gate and release behavior separately.
5. **Post-recovery freshness:** a recovery changes XY/Z and physically raises. The forced intent target, logged execution facts, and spectator inputs must use the updated state.
6. **Yaw preservation:** test patched yaw/tilt with unchanged effective yaw, deliberate replacement, held-object refresh, identity change, and angle wrap. Verify achieved orientation even when XYZ was already reached.
7. **Plan completion:** missing/partial keypoints, empty `adjust`, and relook-without-keypoints must not be recorded as a successful target refresh. Invalid payloads must leave all state unchanged.
8. **Recovery contract/DSL:** laterally shifting shape must execute; mixed controls must execute in defined order or be rejected; scalar/vector misuse and `vec(dist(1, 0), 0, 0)` must be rejected as proposed constraints require.
9. **Identity exhaustion:** exhausted relook/help with unresolved identity must produce a bounded escape or explicit terminal failure, not repeated capped selections; refused closure must not count as empty capture.
10. **Resume durability:** interrupt after result creation, after reflection acceptance, and after the last update. Resume must complete exactly the missing phase and preserve the same library hashes; a subprocess timeout must not abort unrelated remaining episodes.
11. **Reflection evidence:** an episode repaired only by a library stall recovery must be labeled as such; late release/failure evidence and before/after close facts must survive summarization.
12. **Later simulator smoke scope:** single pick/place, held-object place refresh, post-lift slip, blocked placement, and a staged articulation task; inspect official success separately from stage/finish claims and record remaining budget at each transition.

## 中文摘要（给用户）

- 最优先的问题是旧的异步请求还能写入后来复用的结果区，可能把旧画面的建议应用到当前动作。
- 重新识别出正确目标后，代码可能直接在原来的位置夹爪闭合；应先重新对准，再允许抓取。
- 慢思考和备用处理对目标身份的判断不一致，备用处理还会无条件松开已闭合的夹爪，可能丢掉已抓住的物体。
- 恢复动作改变位置和抓取点后，下一动作仍可能使用旧状态；保留抓取角度的逻辑也有持物时被清空的漏洞。
- 每轮学到的经验没有立即保存，程序中断后可能丢失，而恢复运行又会跳过对应回合。
- 经验已经会传给 Jev，单次下降卡住也已加入判断；但身份不确定、抓取后滑落和放置失败等情况仍缺少完整恢复规则。
- 当前仍主要支持单物体抓放，抽屉、微波炉和多阶段任务需要明确的阶段、运动方式和预算安排。本次只读代码并写了审查文件，没有启动实验。
