# LIBERO harness runtime contract

Run from the repository root:

```bash
python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIBRARY.json OUTDIR [nofallback]
```

`SUITE` is a LIBERO benchmark registry key; `TID` and `INIT` select the task and initial state. The entry point is a top-level script and should not be imported to inspect its helpers.

## External dependencies

The rollout needs a working LIBERO installation (BDDL files, initial states and simulation assets), its compatible MuJoCo/robosuite stack, NumPy, SciPy, Pillow, and a working offscreen rendering backend. The script defaults to EGL. This repository does not pin or provide a validated simulator installation.

The external Jev adapter is not included. Supply its directory via `PYTHONPATH`, or place the modules under `libero_rsi/bench_common/`. Its actual service configuration and credentials belong to that external installation.

| Adapter API used by the harness | Required behavior |
|---|---|
| `jev_choice.choose(state, instructions, criteria, meta=...)` | Return `(choice, probabilities, latency)`, where `choice` is a key of `criteria`. Used for both acting and spectator decisions. |

No `astra_client` module is required. Planning, slow-think, visual perception and identity verification all invoke Codex CLI. Perception uses the Astra model (`gpt-6-astra`); Jev decisions still use `jev_choice`. Async Codex planning can overlap main-thread Codex perception calls. External Jev implementation and service budgets are not supplied or verified here.

Production roles are fixed: Codex performs start planning and slow-think, Astra via Codex CLI provides visual perception and identity checks, and Jev chooses intents. The old `CODING_AGENT_SLOW`, `EPISODE_START_PLAN`, and `CODING_AGENT_PERCEIVE` environment switches are no longer used. There is no dual mode or direct Astra-client fallback.

Codex requires its CLI and configured authentication. Configure the planning model and effort with `CODING_AGENT_CODEX_MODEL` and `CODING_AGENT_CODEX_EFFORT`, and the executable with `CODING_AGENT_CODEX_BIN`. Perception and marked-grasp identity verification explicitly use `gpt-6-astra` with `ASTRA_EFFORT` (default `xhigh`), independently of the planning-model override. There is no mock backend. Offline tests inject fixed replies at the CLI boundary; they do not demonstrate visual capability.

For early recovery the harness abandons pending advice, refreshes keypoints/identity using Astra through Codex CLI, and calls synchronous Codex planning. A perception failure stops that recovery attempt; a Codex failure is recorded without invoking a separate client or substituting a mock result. Existing help-intent labels (`ask_astra`, `consult_astra`, `defer_to_astra`) keep their names in libraries/logs but route to Codex planning.

## Policy-visible inputs

- End-effector position, finger joint opening, last gripper command and action history.
- One contact boolean per finger, with no contacted-object identity, pose or force.
- RGB-D perception: image keypoints backprojected through the camera calibration and depth; language-target identity assertions from perception/verification.
- Official binary success, which terminates the episode.
- Library predicates, recoveries and qualitative lessons, plus episode-local corrections.
- Recovery-menu facts, only when `JEV_RECOVERY_MENU` is `jev` or `both`: `last_recovery_outcome`, `recent_recovery_outcomes`, push/clear counters, `push_aside_capped`, `need_reapproach`, `stall_feedback_count`, and `last_stall_issue`. Rules mode (the default) does not add these keys. See the README section "Jev recovery menu".

The start planner cannot elevate the identity gate. An unverified target queues `relook`; planning statements such as “blocked by neighbour” do not authorize closing or pushing.

## Diagnostics and asynchronous execution

`DIAG` writes privileged simulator state, including object poses, for offline analysis. These files and the final diagnostic result are not monitor inputs. The monitor receives only policy decision history, the captured policy state, experience, and captured images.

Each async job owns detached nested state, an image file, and a private result box. The worker never reads live `EP`, `PATCH`, `GOAL_TARGET` or decision files. Only the main thread applies results and writes decision log events. Job IDs reject abandoned jobs. Decision age, target revision, gripper command, close-attempt count, and identity state are checked before applying async advice. Shutdown joins active/orphan CLI calls and harvests their records without applying further actions. Each CLI subprocess has a timeout.

Libraries must pass `validate.py` before a simulator is created. A recovery exception is logged with its rule ID and terminates the episode as `harness_error`; there is no rollback for partially executed robot actions.

The monitor receives the initial plan, current facts and episode lessons. A plan's `relook` step invokes visual perception; keypoints included in a planning reply trigger a visual refresh rather than replacing the perception base directly. Start-planning failures prevent entry into the Jev loop.

The adapter rejects nonzero CLI exit codes even if stdout contains JSON. Missing requested images, malformed/nonfinite geometry, and invalid plan adjustments are errors. Planning calls and visual calls are counted separately; zero cost from the CLI adapter is an unmeasured placeholder.
