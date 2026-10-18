# embodiedjev-rsi

Public snapshot of the LIBERO RSI / EmbodiedJev harness. This repo had no prior tree, so the agent module, episode runner, experience brief, libraries, logic smokes, and review notes all live under `libero_rsi/`.

| Path | Role |
|---|---|
| `libero_rsi/coding_agent_slow.py` | Codex CLI transport for start planning, slow-think, Astra perception and identity checks; explicit mock calls for offline tests. |
| `libero_rsi/run_ep_fbpatch.py` | Episode harness: Jev intents, stall feedback, grasp confirm, local-deadend relook, place force-release. Needs LIBERO, MuJoCo, Codex CLI, and the Jev client to actually roll out. |
| `libero_rsi/jev_experience.py` | Builds `experience_brief` (recoveries, lessons, and the start-plan prior) for Jev. |
| `libero_rsi/validate.py` | Library validator. Imports sibling `dsl_libero.py`. |
| `libero_rsi/dsl_libero.py` | Expression DSL and dimensional checks used by the validator and logic smokes. |
| `libero_rsi/libs/*.json` | Experience libraries, including `ee_pg_easy_v2_lib.json`. |
| `libero_rsi/smoke/test_*.py` | Logic tests. No simulator and no network on the mock path. |
| `libero_rsi/reviews/*.md` | Design notes for the behaviors below. |

## Episode-start global plan

The harness always uses Codex for episode-start planning.

Before the first approach the planner reads `experience_brief` and writes one ordered skeleton (`start_plan.json`). `phase_order` is a full cycle per language-goal object: approach → grasp → lift → place → **retreat**, then the next object, ending with a retreat. Retreat after every place is filled in if the model omits it. Mid-episode slow-think only fine-tunes that skeleton.

`mock` uses explicit start observations: `neighbour_observed=true` plus confirmed identity permits `push_aside`; conditional lessons and neighbour priors alone do not. Unknown or rejected identity leads to `relook`. Production calls the Codex CLI. The mock planner is only used by explicit offline test calls.

The start planner cannot confirm target identity. Its proposed plan and the gated executable plan are both saved in `start_plan.json`. When the identity gate is unconfirmed, the harness queues a real `relook` intent before contact, even if the planner claims a match.

## Local dead-end → synchronous global relook

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
| Episode-start planning and sync/async slow-think | `CODING_AGENT_CODEX_MODEL` (default `gpt-6-astra`) | Codex CLI |
| Visual perception, keypoints and identity verification | Astra (`gpt-6-astra`) | Codex CLI |
| Per-step intent selection and spectator decision | Jev | `jev_choice.choose` |

Codex is the invocation mechanism; Astra is the visual model. Perception and marked-grasp identity verification both use `codex exec -m gpt-6-astra`. There is no dual/comparison mode or direct Astra-client transport. The former `CODING_AGENT_SLOW`, `EPISODE_START_PLAN`, and `CODING_AGENT_PERCEIVE` role switches are no longer read. `mode='mock'` remains available only when explicitly calling the standalone planning/perception helpers in offline tests.

Early recovery refreshes perception with Astra through Codex CLI, then requests Codex planning. CLI failures do not switch to a direct Astra client or to mock results. The existing `ask_astra`, `consult_astra`, and `defer_to_astra` intent labels are retained for the shipped libraries and logs, but their help path invokes Codex planning.

Codex looks up `codex` on `PATH`, or `CODING_AGENT_CODEX_BIN`. Planning uses `CODING_AGENT_CODEX_MODEL` and `CODING_AGENT_CODEX_EFFORT`; vision uses `gpt-6-astra` with `ASTRA_EFFORT` (default `xhigh`). A proxy is used only when `CODING_AGENT_HTTP_PROXY` or `CODING_AGENT_SOCKS` is set. Do not commit proxy URLs, keys, or host details.

Async slow-think captures a detached copy of policy state and a unique image before starting its worker. Workers return results and log events in private job boxes; the main thread writes the decision log. Simulator diagnostic files are never passed to the coding monitor, including when `DIAG` is enabled.

## Running an episode

This snapshot does **not** include `jev_choice`. Actual rollouts require that external Jev client, an authenticated Codex CLI, LIBERO with its assets, MuJoCo/robosuite, and rendering dependencies. `astra_client` is no longer needed. The client interfaces and policy/diagnostic boundary are documented in [libero_rsi/ADAPTER.md](libero_rsi/ADAPTER.md). Offline mock tests do not replace the services required by the full harness.

Once the external dependencies are installed:

```bash
PYTHONPATH=/path/to/bench/common python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIB.json OUTDIR
```

Libraries are validated before environment creation. Recovery-rule errors are logged and end the episode with `harness_error`, rather than silently continuing with potentially partial state changes.

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

The regression suite executes selected production functions with environment/model doubles, covering identity authorization, observed versus prior clutter, fixed backend routing and Codex failure handling, event timestamps, async snapshots, diagnostic isolation, recovery failures, and validation of every shipped library. These tests do not establish simulation success.

`libs/smoke_copies/` contains duplicate library fixtures retained from the original snapshot; these are not independent policy variants or experimental results.
