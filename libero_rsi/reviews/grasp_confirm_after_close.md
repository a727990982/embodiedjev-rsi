# Grasp confirm after close (2026-10-18)

## Rule
After `close_gripper`: proprio (opening + pads) + short closed **lift probe** (+ optional `seen_held`).
- Log `grasp_confirm=true/false`
- If secured → `grasp_secured_once`, force `lift`; suppress `push_aside` / local-deadend retreat / stall undo while holding
- If not → reopen + reapproach (existing recovery)

## Cream t1_i0 retest
- Path: `runs/ep_start_plan_cream/libero_object_t1_i0_mock_spmock_graspconfirm/`
- `EPISODE_START_PLAN=mock` `CODING_AGENT_SLOW=mock` + deadend rules
- **success=true** env_steps=496
- First close t≈182: `grasp_confirm=true` → `grasp_confirm_skip` → **lift** (no push_aside / no deadend after close)
- Video: `.../videos/ep.mp4`

## Files
- `libero_rsi/run_ep_fbpatch.py` — `_confirm_grasp_after_close`, gates
- `libero_rsi/coding_agent_slow.py` — mock respects `grasp_secured`
- `libero_rsi/smoke/test_grasp_confirm_logic.py`
