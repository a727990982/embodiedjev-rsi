# embodiedjev-rsi

Public snapshot of the LIBERO RSI / EmbodiedJev harness. This repo had no prior tree, so the agent module, episode runner, experience brief, libraries, logic smokes, and review notes all live under `libero_rsi/`.

| Path | Role |
|---|---|
| `libero_rsi/coding_agent_slow.py` | Episode-start plan, mid-episode slow-think, and perceive. Backend: `codex` (`codex exec`); unsupported values such as `mock` raise an error at startup. |
| `libero_rsi/run_ep_fbpatch.py` | Episode harness: Jev intents, stall feedback, grasp confirm, local-deadend relook, place force-release. Needs LIBERO, MuJoCo, and the Astra/Jev clients to actually roll out. |
| `libero_rsi/jev_experience.py` | Builds `experience_brief` (recoveries, lessons, and the start-plan prior) for Jev. |
| `libero_rsi/validate.py` | Library validator. Imports sibling `dsl_libero.py`. |
| `libero_rsi/dsl_libero.py` | Expression DSL and dimensional checks used by the validator and logic smokes. |
| `libero_rsi/libs/*.json` | Experience libraries, including `ee_pg_easy_v2_lib.json`. |
| `libero_rsi/smoke/test_*.py` | Logic tests of validators, gating, prompts, and harness hooks. No simulator and no network. |
| `libero_rsi/reviews/*.md` | Design notes for the behaviors below. |

## Episode-start global plan

`EPISODE_START_PLAN` is `0` (off), `codex`, `astra`, or `auto` (empty/`auto` uses `codex` when `CODING_AGENT_SLOW` is `codex`/`dual` or `CODING_AGENT_PERCEIVE` is `codex`, else off).

Before the first approach the planner reads `experience_brief` and writes one ordered skeleton (`start_plan.json`). `phase_order` is a full cycle per language-goal object: approach → grasp → lift → place → **retreat**, then the next object, ending with a retreat. Retreat after every place is filled in if the model omits it. Mid-episode slow-think only fine-tunes that skeleton.

`codex` calls the Codex CLI. `astra` is applied by the harness client.

The start planner cannot confirm target identity. Its proposed plan and the gated executable plan are both saved in `start_plan.json`. When the identity gate is unconfirmed, the harness queues a real `relook` intent before contact, even if the planner claims a match.

## Local dead-end → synchronous global relook

Repeated local stall-family actions (`push_aside`, `clear_reapproach`, `raise_redescend`, lateral recenter) without a secured grasp, or hitting the push-aside caps, abort the local loop:

1. Retreat for a clear front (agentview) look.
2. Global relook of keypoints.
3. **Synchronous** slow-think (`CODING_AGENT_SLOW`, or the Astra hybrid) before Jev moves again.

Caps: `LOCAL_DEADEND_N` (default 3), `MAX_LOCAL_DEADEND_GLOBAL` (default 2), `MAX_PUSH_ASIDE` (default 2), `MAX_PUSH_ASIDE_PER_APPROACH` (default 1).

## Grasp confirm and place force-release

After `close_gripper`, `_confirm_grasp_after_close` checks proprio (opening and pads) plus a short closed lift probe. `grasp_confirm=true` forces lift and suppresses `push_aside`, local-deadend retreat, and stall undo while the object is held. A failed confirm reopens and reapproaches.

During place, consecutive `lower_to_place` stalls (`MAX_PLACE_LOWER_STALL`, default 3) or lift↔lower cycles (`MAX_PLACE_LIFT_LOWER_CYCLES`, default 2), while the held object is near the place XY, set `place_force_release` and force `open_gripper`.

## Backends

| Env | `0` / unset | `codex` | other |
|---|---|---|---|
| `CODING_AGENT_SLOW` | Astra slow-think | Codex monitor | `dual` calls Codex for logging and still applies Astra; Codex errors are logged and Astra continues |
| `CODING_AGENT_PERCEIVE` | Astra keypoints | Codex keypoints | `astra` is the same as off |
| `EPISODE_START_PLAN` | inherit, else off | Codex start plan | `astra` uses the harness client |

Unsupported values, including the removed `mock` backend and bare `1`/`on`, raise an error at startup before the simulator is created.

Codex looks up `codex` on `PATH`, or `CODING_AGENT_CODEX_BIN`. A proxy is used only when `CODING_AGENT_HTTP_PROXY` or `CODING_AGENT_SOCKS` is set. Do not commit proxy URLs, keys, or host details.

Async slow-think captures a detached copy of policy state and a unique image before starting its worker. Workers return results and log events in private job boxes; the main thread writes the decision log. Simulator diagnostic files are never passed to the coding monitor, including when `DIAG` is enabled.

## Running an episode

This snapshot does **not** include `astra_client` or `jev_choice`. Actual rollouts require those external clients, LIBERO with its assets, MuJoCo/robosuite, and rendering dependencies. The client interfaces and policy/diagnostic boundary are documented in [libero_rsi/ADAPTER.md](libero_rsi/ADAPTER.md). Identity verification and Jev execution always need the clients.

Once the external dependencies are installed:

```bash
PYTHONPATH=/path/to/bench/common python libero_rsi/run_ep_fbpatch.py SUITE TID INIT LIB.json OUTDIR
```

Libraries are validated before environment creation. Recovery-rule errors are logged and end the episode with `harness_error`, rather than silently continuing with potentially partial state changes.

## Smoke tests

From the repo root (Python 3; `numpy` for the tests that import it):

```bash
failed=0
for t in libero_rsi/smoke/test_*.py; do
  python "$t" || failed=1
done
test "$failed" -eq 0
```

Each test adds `libero_rsi/` to `sys.path`, so `validate` and `dsl_libero` resolve as siblings. `numpy` is required for the tests that import it. No simulator and no network.

The regression suite executes selected production functions with environment/model doubles, covering identity authorization, dual routing, rejection of unsupported backend values, event timestamps, async snapshots, diagnostic isolation, recovery failures, and validation of every shipped library. These tests do not establish simulation success.

`libs/smoke_copies/` contains duplicate library fixtures retained from the original snapshot; these are not independent policy variants or experimental results.
