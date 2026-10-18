# Local-deadend → global relook → sync slow-think (2026-10-18)

## Rule
When local/wrist near-grasp stall family thrash has no useful solution:
1. Abort local push_aside / move_above / descend loop
2. Retreat upward for a clear **front (agentview)** look
3. Global relook keypoints
4. **Synchronous** `_sync_slow` (Codex planner) — Jev held until plan applied
5. Cap `push_aside` per episode / per approach

## Triggers
- `push_aside_n >= MAX_PUSH_ASIDE` (default 2)
- `push_aside_approach_n >= MAX_PUSH_ASIDE_PER_APPROACH` (default 1) on another push proposal
- `LOCAL_DEADEND_N` (default 3) consecutive local-loop stall actions without grasp secured
- Budget: `MAX_LOCAL_DEADEND_GLOBAL` (default 2)

## Log
`local_deadend_global_relook=true` with `reason`, `counts`, `slow_result`, relook identity.

## Cream t1_i0 retest
- Path: `runs/ep_start_plan_cream/libero_object_t1_i0_mock_spmock_deadend2/`
- `EPISODE_START_PLAN=mock` `CODING_AGENT_SLOW=mock` `VIDEO=.../videos/ep.mp4`
- **success=true** env_steps=423; deadend fired **2×** (approach_cap then episode_cap); both sync_cas_mock intervened with raise_redescend (not another push)
- Video: `.../videos/ep.mp4` (~1.1MB)
- Note: this run used the mock start-plan and slow-think backends, which have since been removed from this tree, so it cannot be reproduced as-is.

## Files
- `libero_rsi/run_ep_fbpatch.py` — detection, retreat, global relook, sync slow hook, push caps
- `libero_rsi/coding_agent_slow.py` — slow-think context and plan validation (the backend decides the replan; `local_deadend_global_relook` is passed in `slow_ctx`)
