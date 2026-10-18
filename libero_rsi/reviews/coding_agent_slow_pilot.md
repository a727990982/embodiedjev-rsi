# Coding-agent slow-think pilot (EmbodiedEvo-inspired)

Date: 2026-10-18 (America/New_York). Status: **scaffolded** — offline validator smoke green; real 1-ep A/B not started (wait for `ee_pg_retest_push`).

## EmbodiedEvo here (what exists)

Local tree: an EmbodiedEvo checkout (plus `embodiedevo_rules_notes.md`). Neither is part of this public snapshot.

| Piece | What the coding agent sees | What it writes |
|---|---|---|
| Online loop | `observations/` (stdout/stderr, camera stills/videos, robot state, success) after each `scripts/env.py exec` | `submission/solution.py` (next-stage control code) |
| Playground evolution | Same + prior submissions | `skills/*.py` + markdown, `lessons/*.md` (Signature / Instead / Evidence / Status) |
| Test | Frozen read-only skills/lessons + primitives | Fresh `solution.py` only; no reset across scenes |
| Prior | Thin: action API + “don’t hard-code scene numbers; derive from observation” (prompt). **No** hand-written recovery menu like ours |

There is **no separate failure summarizer** — the same coding agent reads raw observations and decides. Recipes under `tasks/RoboDojo/recipe/` are supplied task wiki, not learned skills.

**Our mapping:** EmbodiedEvo’s “write control code” ≈ our “Jev selects intents.” This pilot puts a coding agent in EmbodiedEvo’s *observe → revise* role, but the revision target is **episode plan JSON** (already accepted by `_apply_episode_plan`), not Python controllers.

## Scope (minimal)

- **In:** Mid-episode monitor hook on the same triggers as Astra slow-think (stall / spectator-defer / hard identity / ask_astra). Inputs: recent diag + decisions lines, last 2–4 keyframe jpgs, `experience_brief`, PATCH + GOAL_ID summary. Output: structured intervention (`raise_redescend`, `push_aside`, `adjust`, `relook`, …) or intervene=false.
- **Out:** Full overnight playground rewrite; replacing Jev; library-wide RSI edits from the monitor (episode-only only); killing `ee_pg_retest_push`.

## Scaffolding landed

| Path | Role |
|---|---|
| `libero_rsi/coding_agent_slow.py` | Context pack, prompt, Codex backend, validate, applicator gate |
| `libero_rsi/run_ep_fbpatch.py` | Env `CODING_AGENT_SLOW` (default off). `codex` replaces Astra on sync/async slow; `dual` runs Codex then still applies Astra (log both) |
| `libero_rsi/smoke/test_coding_agent_slow_pilot.py` | Mode resolver + Codex-shaped plan validation → applicator accepts (no sim) |

## A/B protocol (ONE hard ep)

Prefer **libero_object cream `t1_i0`** (or ketchup `t5_i0` if cream already saturated). Same lib snapshot as push retest (`ee_pg_easy_v2` / `lib_used.json`), `ASTRA_EFFORT=xhigh`, `GATE_JEV=1`, `HORIZON=600`.

1. **Baseline A:** `CODING_AGENT_SLOW=0` (current Astra slow-think). Reuse a finished push-retest ep if seed/lib match; else re-run once.
2. **Pilot B:** `CODING_AGENT_SLOW=codex`. Same suite/tid/init/lib/out root under `runs/coding_agent_slow_ab/`.
3. Optional **dual:** `CODING_AGENT_SLOW=dual` — coding-agent proposal logged; Astra still applied (compare plans without changing control yet).

Do **not** start B until `ee_pg_retest_push` finishes (GPU + Astra budget). Push retest remains the right baseline to wait for: it measures the new `push_aside` auto menu; coding-agent monitor is the next layer on top of that recovery vocabulary.

### How to run (when ready)

```bash
# Optional Codex proxy (unset = direct). Set in the environment; do not commit values.
# export CODING_AGENT_HTTP_PROXY=...
# export CODING_AGENT_SOCKS=...
# export CODING_AGENT_CODEX_BIN=codex

# Smoke (anytime):
python libero_rsi/smoke/test_coding_agent_slow_pilot.py
```

## Success metrics (per ep)

| Metric | Why |
|---|---|
| Official success (BDDL) | Primary |
| `push_aside_n` / stall_feedback mix | Did monitor choose neighbour shove vs yaw churn? |
| Steps / decisions to first secure grasp (`closed` + bilateral / held) | Efficiency vs baseline |
| Slow interventions used (`P['slow']`) + wall time of monitor call | Latency / budget |
| Plan disagreement under `dual` (coding vs Astra step set) | Signal before cutting over |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Latency (Codex >> Astra JPEG call) | Sync-only on critical triggers first; keep async Astra or skip async for pilot |
| Cost | Cap `MAX_SLOW`; one ep; check wiring with the offline smokes first |
| Overwrite races | Reuse existing `job_id` + private `result_box` (B01); coding worker writes only into that box |
| Bad JSON / illegal steps | `validate_intervention` before `_finish_slow`; on failure fall back to Astra |
| Proxy required for Codex | Set `CODING_AGENT_HTTP_PROXY` or `CODING_AGENT_SOCKS` in the environment. Nothing is baked in. |
| Scope creep | Episode-only plans; no playground freeze from this hook |

## Recommendation

The mock backend referred to in earlier versions of this note has been removed; unsupported `CODING_AGENT_SLOW` / `CODING_AGENT_PERCEIVE` values now raise at startup.


1. Let **`ee_pg_retest_push`** finish → score push_aside auto path.  
2. Run offline smokes (already).  
3. One cream/ketchup ep with `CODING_AGENT_SLOW=dual` (log-only disagreement).  
4. If dual plans look saner on neighbour blocks, cut over that ep to `codex` replace and compare success / steps-to-grasp.

---

## Perceive via coding agent (2026-10-18)

Env **`CODING_AGENT_PERCEIVE=0|codex`** (default off). When `codex`, harness `perceive()` (episode-start **and** mid-episode relook / escape_look) asks Codex for the **same keypoint JSON** Astra returns (`grasp_px`, `place_px`, yaw/depth, `matched_language_goal`, …) so `_commit_keypoints` is unchanged. On failure → Astra fallback. **`verify_grasp_identity` stays on Astra.**

| Path | Role |
|---|---|
| `coding_agent_slow.py` | `perceive_backend_mode`, `validate_keypoints`, `call_perceive`, `harness_call_perceive` |
| `run_ep_fbpatch.py` | `CODING_AGENT_PERCEIVE` hook inside `perceive()`; saves front (+wrist) stills under `perc_<phase>_*.jpg` |
| `smoke/test_coding_agent_perceive.py` | Codex-shaped keypoints validate (no sim) |

### Run both flags (cream t1_i0)

```bash
CODING_AGENT_SLOW=codex CODING_AGENT_PERCEIVE=codex \
  python libero_rsi/run_ep_fbpatch.py libero_object 1 0 \
  libero_rsi/libs/ee_pg_easy_v2_lib.json /tmp/libero_object_t1_i0_codex
```

The one-episode shell helper from the private bench is not in this snapshot. The runner needs LIBERO, MuJoCo, and the Astra/Jev clients.

Smoke:

```bash
python libero_rsi/smoke/test_coding_agent_perceive.py
```

### Codex CLI note (prompt + images)

`codex exec -i FILE...` treats `-i` as **variadic**. Passing the prompt as the last argv after `-i` makes Codex eat the prompt as another image and then fail with `No prompt provided via stdin`. Fix: always pass prompt on **stdin** with argv `-` (`call_codex` does this for both slow-think and perceive).
