# Call-chain audit — 2026-10-18

Audited the PR branch starting at `e6ad561`, then integrated the concurrent removal of mock backends at `4474c76` and reran the tests. This is a code/interface and offline execution audit, not a LIBERO performance result.

## Current invocation chain

| Stage | Actual entry and result |
|---|---|
| Startup | Validate the library, create LIBERO, settle the arm, obtain initial observation. |
| Visual perception | `perceive` → `harness_call_perceive` → `call_perceive` → `call_codex` with `gpt-6-astra`; image keypoints are backprojected using local depth/calibration. |
| Identity check | `verify_grasp_identity` marks the grasp pixel → `call_identity_check` → `call_codex` with `gpt-6-astra`. |
| Start planning | `_run_episode_start_plan` → `harness_call_start_plan` → `call_start_plan` → `call_codex`; the planning model is configurable, defaulting to `gpt-6-astra`. |
| Intent selection | Library rules or queued recovery intents may force an action; otherwise `jev_choice.choose` selects an intent from the current state/experience. A separate Jev call may request help. |
| Sync slow-think | `_sync_slow` → captured context/image → `call_monitor` → Codex → `_finish_slow` → `_apply_episode_plan`. |
| Async slow-think | `_launch_async_slow` → private snapshot/worker/result box → freshness/ownership checks → application on main thread. |
| Early recovery | Revoke pending advice → Astra perception through Codex → synchronous Codex planning. |
| Shutdown | Finish environment settling, join pending bounded CLI calls, collect records without applying plans, write result, close environment. |

No production `astra_client`, `AC.call`, `dual`, or alternate planner transport remains. The names `ask_astra`, `consult_astra`, and `defer_to_astra` remain in the existing library/log vocabulary; their actual planning path goes through Codex. The start plan's phase order is advisory context for Jev and slow-think, not a deterministic phase executor.

## Confirmed problems fixed in this audit

- A `relook` plan could append a log marker without perception. It now calls the visual path. Further steps based on the old image are not executed after that refresh. Planner-supplied keypoints also request perception instead of directly replacing its base.
- Async advice could apply after target/grasp/identity changes within one decision. Snapshot revision, gripper command, close-attempt count and identity checks now discard it. Patch resets and shape-rule changes advance the target revision.
- Start-plan errors were recorded but ignored by the decision loop. They now produce `harness_error` and nonzero exit before Jev decisions. The decision counter is initialized before start-plan recovery can call slow-think.
- CLI nonzero exits could be accepted if stdout contained JSON. They are now errors. Missing requested image files no longer silently turn a visual call into a text-only call.
- Invalid object-center coordinates and nonfinite geometry could pass perception validation. Numeric plan errors could also be discarded by start-plan validation. Both routes now reject those values.
- Slow-think omitted the initial plan and episode lessons from its experience brief. Both now travel with current facts. The monitor prompt no longer cuts raw serialized context mid-JSON, which could remove current state after a long history.
- End-of-episode polling could apply a pending intervention after a policy finish or official success. Shutdown now only harvests pending results and waits for active/orphan CLI workers.
- Planning replies were included in `astra_calls`, while start-plan calls were omitted. `astra_calls` now counts visual replies; `planning_calls` counts start/slow-planning replies. CLI cost fields remain unmeasured placeholders, not billing evidence.
- A legacy smoke reproduced the old no-op relook logic. The relevant checks now execute the production applicator.

## Evidence

- Before fixes, six targeted regression tests exposed nine failing assertions/subcases in the actual functions. They covered CLI failure handling, geometry validation, start-plan validation, relook execution, async freshness, and monitor context.
- All 20 smoke scripts passed after the fixes, including 11 review regression cases, five Codex visual-routing cases, the call-chain audit cases, and two full-entrypoint cases using fake services.
- The full-entrypoint tests run the actual top-level script with fake LIBERO/MuJoCo observations, fake rotation math, a Jev fixture, and model replies. They cover normal startup, sync/async planning, a relook, finish/result writing, and a failed start plan that prevents Jev execution.
- Python syntax and module-global symbol checks passed. These checks do not prove numerical control behavior.
- The locally installed `codex exec --help` confirms the used CLI image/model/stdin options. No real model request was made during this audit.

## Remaining runtime requirements and limits

The current local Python environment does not provide `jev_choice`, LIBERO, MuJoCo, robosuite, or SciPy. NumPy, Pillow, and the Codex executable are present. Therefore this checkout still cannot run a real episode on this machine without installing/providing those dependencies and the LIBERO assets.

Real Jev adapter behavior, Codex authentication/model availability, image interpretation, rendering, physics, grasp/place success and latency were not tested. The external Jev client is not in the repository, so its return-value contract can only be checked against a test fixture here. Existing pure logic/mirror tests are supplementary; they are not simulation evidence.

The runner remains a top-level script with episode-global state. This audit fixes reproduced call-chain failures; it does not claim that all controller behavior or every external failure path is exhaustively verified.
