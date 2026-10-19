# Episode-start Great Wall plan (2026-10-18)

## Current execution

Astra via Codex generates the start plan and task todo; Codex runs slow-think, and Astra via Codex provides perception/identity. Jev reviews todo completion and then chooses intents from updated todo, current images and proprioception. Start phase_order is historical planner advice, not Jev state. See [todo runtime contract](../ADAPTER.md). The former role-selection environment switches and mock backends are removed; see [current setup](../../README.md#fixed-model-roles).

The runs below are historical records. In particular, the old proposal to confirm identity from a start-plan assertion has been superseded by `gate_start_plan`: planning cannot elevate the identity gate.

## Smoke (libero_object cream t1_i0)
- Mock path (executed push): `runs/ep_start_plan_cream/libero_object_t1_i0_mock_spmock/libero_object_t1_i0/`
  - `start_plan.json` step1=push_aside, applied `push_aside:ok` (pushed=true)
  - `result.json` success=false end=horizon astra~$0.28 wall~110s
  - video: `.../videos/ep.mp4`
- Codex planner evidence (attempt1): `.../libero_object_t1_i0_codex_spcodex_attempt1_push_refused/`
  - Codex chose push_aside + blocker_px=[130,318] (~15s); apply initially refused identity
  - Fixed: confirm GOAL_ID from start-plan matched_language_goal before push
- Note: the mock-path run used the mock start-plan and slow-think backends, which have since been removed from this tree, so it cannot be reproduced as-is.

## Files
- `libero_rsi/coding_agent_slow.py` — START_PLAN_PROMPT, codex backend, validate, gate_start_plan, harness_call_start_plan
- `libero_rsi/run_ep_fbpatch.py` — `_run_episode_start_plan`, persist start_plan.json/task_todo.json, active-target handoff and Jev image input
- `libero_rsi/jev_experience.py` — experience_brief.episode_start_plan for the planner
- `libero_rsi/task_todo.py` / `jev_observation.py` / `jev_choice.py` / `jev_todo.py` — progress and TypeSafe image decision chain
- `libero_rsi/libs/ee_pg_easy_v2_lib.json` — v14 + hp_start_predict_neighbour lesson
