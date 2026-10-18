# Episode-start Great Wall plan (2026-10-18)

## Enable
```bash
export EPISODE_START_PLAN=codex   # or mock|astra|auto
# auto inherits CODING_AGENT_SLOW / CODING_AGENT_PERCEIVE when those are on
export CODING_AGENT_SLOW=codex    # optional mid-episode monitor
```

Helper: `runs/coding_agent_slow_ab/run_one.sh SUITE TID INIT [slow] [perceive] [start_plan]`

## Smoke (libero_object cream t1_i0)
- Mock path (executed push): `runs/ep_start_plan_cream/libero_object_t1_i0_mock_spmock/libero_object_t1_i0/`
  - `start_plan.json` step1=push_aside, applied `push_aside:ok` (pushed=true)
  - `result.json` success=false end=horizon astra~$0.28 wall~110s
  - video: `.../videos/ep.mp4`
- Codex planner evidence (attempt1): `.../libero_object_t1_i0_codex_spcodex_attempt1_push_refused/`
  - Codex chose push_aside + blocker_px=[130,318] (~15s); apply initially refused identity
  - Fixed: confirm GOAL_ID from start-plan matched_language_goal before push

## Files
- `libero_rsi/coding_agent_slow.py` — START_PLAN_PROMPT, mock/codex, validate, harness_call_start_plan
- `libero_rsi/run_ep_fbpatch.py` — `_run_episode_start_plan`, persist start_plan.json, Jev inject
- `libero_rsi/jev_experience.py` — experience_brief.episode_start_plan
- `libero_rsi/libs/ee_pg_easy_v2_lib.json` — v14 + hp_start_predict_neighbour lesson
