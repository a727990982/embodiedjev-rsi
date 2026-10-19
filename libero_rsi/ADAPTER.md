# LIBERO harness runtime contract

Run from the repository root:

```bash
python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIBRARY.json OUTDIR [nofallback]
```

`SUITE` is a LIBERO benchmark registry key; `TID` and `INIT` select the task and initial state. The entry point is a top-level script and should not be imported to inspect its helpers.

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

Neither acting nor spectator Jev gets cached grasp/place coordinates, `seen_held`,
geometry-derived predicates, predicate meanings or active recovery matches.
Acting additionally receives `execution_history`: the last eight completed
intent/result pairs, the last intent, and full trailing counts for the same
intent and the same intent/result pair. Forced high-level actions are included
because they were executed. These records describe past executor outcomes;
they do not assert that an object is currently held or that a goal is complete.
Spectator additionally gets actual recent intent/result events and help budgets.
Controller geometry and predicates stay in the executor and planner context;
they are not deleted from the system.

The start planner must return `task_todo.items`, each with unique `id`, `target`,
`goal`, and `completion_condition`. All start as pending, regardless of a planner
claim of completion. Each item describes an object's desired final outcome,
not a fixed sequence of motor commands. The first non-done item is active.

After release, relook, retreat, help, failure, or a premature finish request,
Astra reviews every item using a new same-frame image/sensor snapshot. A complete
review returns each id with `pending`, `done`, `uncertain` or `unobserved` and an
evidence explanation. `unobserved` retains its previous status/evidence/timestamp;
it cannot create a new completion. Explicit contradictory evidence can reset a
done item. Malformed reviews stop the episode rather than fabricate progress.

The main thread owns todo updates. Each review abandons outstanding slow advice.
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
