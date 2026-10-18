# Place-phase force release (2026-10-18)

## Rule
After `carry_to_place` / while held XY is over the place region:
- Track consecutive `lower_to_place` `stalled`/`timeout` (`MAX_PLACE_LOWER_STALL`, default 3)
- Track lift↔lower cycles (`MAX_PLACE_LIFT_LOWER_CYCLES`, default 2)
- If either hits and held XY near place → **force `open_gripper`**, log `place_force_release=true` + reason
- Default `place_clearance_m` floor raised to 0.05 (`PLACE_CLEARANCE_DEFAULT`) when perception omits it

## Why
Cream graspconfirm ep: `carry_to_place` reached, then ~21× `lower_to_place` stalled + lift↔lower thrash until open ~t=496.

## Files
- `libero_rsi/run_ep_fbpatch.py` — hard gate + counters
- `libero_rsi/libs/ee_pg_easy_v2_lib.json` — recovery `hp_place_lower_stall_release`
- `libero_rsi/smoke/test_place_force_release_logic.py`

## Cream t1_i0 retest (`mock_spmock_placeforce`)
- `EPISODE_START_PLAN=mock` `CODING_AGENT_SLOW=mock` + grasp_confirm + deadend + place force
- **success=true** `finish_then_success` env_steps=**251** (prior graspconfirm thrash ep: 496)
- Place: `carry` reached → `lower_to_place` **reached once** → `open_gripper` ~t=243 (0 lower stalls; force gate not needed this seed)
- Video: `runs/ep_start_plan_cream/libero_object_t1_i0_mock_spmock_placeforce/videos/ep.mp4`
- Note: this run used the mock start-plan and slow-think backends, which have since been removed from this tree, so it cannot be reproduced as-is.
