# embodiedjev-rsi

Public snapshot of the LIBERO RSI / EmbodiedJev harness. This repo had no prior tree, so the agent module, episode runner, experience brief, libraries, logic smokes, and review notes all live under `libero_rsi/`.

| Path | Role |
|---|---|
| `libero_rsi/coding_agent_slow.py` | Episode-start plan, mid-episode slow-think, and perceive. Backends: `mock` (local, no network) and `codex` (`codex exec`). |
| `libero_rsi/run_ep_fbpatch.py` | Episode harness: Jev intents, stall feedback, grasp confirm, local-deadend relook, place force-release. Needs LIBERO, MuJoCo, and the Astra/Jev clients to actually roll out. |
| `libero_rsi/jev_experience.py` | Builds `experience_brief` (recoveries, lessons, and the start-plan prior) for Jev. |
| `libero_rsi/libs/*.json` | Experience libraries, including `ee_pg_easy_v2_lib.json`. |
| `libero_rsi/smoke/test_*.py` | Logic tests. No simulator and no network on the mock path. |
| `libero_rsi/reviews/*.md` | Design notes for the behaviors below. |

## Episode-start global plan

`EPISODE_START_PLAN` is `0` (off), `mock`, `codex`, `astra`, or `auto` (empty/`auto` inherits `CODING_AGENT_SLOW`, or `CODING_AGENT_PERCEIVE` when slow-think is off).

Before the first approach the planner reads `experience_brief` and writes one ordered skeleton (`start_plan.json`). `phase_order` is a full cycle per language-goal object: approach → grasp → lift → place → **retreat**, then the next object, ending with a retreat. Retreat after every place is filled in if the model omits it. Mid-episode slow-think only fine-tunes that skeleton.

`mock` is deterministic from the brief and start hints (neighbour prior → `push_aside` first). `codex` calls the Codex CLI. `astra` is applied by the harness client, not by this module.

## Local dead-end → synchronous global relook

Repeated local stall-family actions (`push_aside`, `clear_reapproach`, `raise_redescend`, lateral recenter) without a secured grasp, or hitting the push-aside caps, abort the local loop:

1. Retreat for a clear front (agentview) look.
2. Global relook of keypoints.
3. **Synchronous** slow-think (`CODING_AGENT_SLOW`, or the Astra hybrid) before Jev moves again.

Caps: `LOCAL_DEADEND_N` (default 3), `MAX_LOCAL_DEADEND_GLOBAL` (default 2), `MAX_PUSH_ASIDE` (default 2), `MAX_PUSH_ASIDE_PER_APPROACH` (default 1).

## Grasp confirm and place force-release

After `close_gripper`, `_confirm_grasp_after_close` checks proprio (opening and pads) plus a short closed lift probe. `grasp_confirm=true` forces lift and suppresses `push_aside`, local-deadend retreat, and stall undo while the object is held. A failed confirm reopens and reapproaches.

During place, consecutive `lower_to_place` stalls (`MAX_PLACE_LOWER_STALL`, default 3) or lift↔lower cycles (`MAX_PLACE_LIFT_LOWER_CYCLES`, default 2), while the held object is near the place XY, set `place_force_release` and force `open_gripper`.

## Mock vs Codex

| Env | `0` / unset | `mock` | `codex` | other |
|---|---|---|---|---|
| `CODING_AGENT_SLOW` | Astra slow-think | local monitor | Codex monitor | `dual` logs Codex (or mock) and still applies Astra |
| `CODING_AGENT_PERCEIVE` | Astra keypoints | local keypoints | Codex keypoints | off |
| `EPISODE_START_PLAN` | inherit, else off | local start plan | Codex start plan | `astra` uses the harness client |

Codex looks up `codex` on `PATH`, or `CODING_AGENT_CODEX_BIN`. A proxy is used only when `CODING_AGENT_HTTP_PROXY` or `CODING_AGENT_SOCKS` is set. Do not commit proxy URLs, keys, or host details.

## Smoke tests

From the repo root (Python 3; `numpy` for the tests that import it):

```bash
for t in libero_rsi/smoke/test_*.py; do
  python "$t" || echo "FAIL $t"
done
```

These run with no extra modules:

- `test_episode_start_plan.py`
- `test_coding_agent_slow_pilot.py`
- `test_coding_agent_perceive.py`
- `test_grasp_confirm_logic.py`
- `test_place_force_release_logic.py`
- `test_goal_ownership_logic.py` (numpy)
- `test_async_slow_ownership.py`
- `test_clear_reapproach_yaw.py`
- `test_post_recovery_ns_freshness.py` (numpy)
- `test_reapproach_before_close.py` (numpy)
- `test_slow_prompt_issue_enum.py`
- `test_spectator_jev_gate.py`

These five also import `validate` and/or `dsl_libero` from the private bench (not in this snapshot). Put those modules on `PYTHONPATH` before running them:

- `test_ask_astra_help_logic.py`
- `test_deadlock_escape_logic.py`
- `test_jev_experience_brief.py`
- `test_push_aside_neighbour.py`
- `test_stall_feedback_logic.py`
