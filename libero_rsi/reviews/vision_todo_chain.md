# Main harness: task todo and current-image decisions

The production runner now follows this chain:

1. Astra via Codex writes the initial plan and one todo outcome per requested object.
2. The harness initializes all items as pending, selects the active object, refreshes
   controller keypoints and verifies the marked grasp identity for that object.
3. Acting Jev reads the todo, current front/wrist images, same-frame proprioception
   and official success. The bundled TypeSafe client pins `jev-1.14.0` and sends
   actual image assets. Cached perception/predicate/experience state is absent.
4. Jev selects a candidate described by its applicability and executor behavior.
   Reopening an empty/wrong grasp happens in place; re-alignment is a later choice.
5. Astra reviews todo progress at release/relook/retreat/help/failure checkpoints.
   Visible evidence can advance or reopen goals. Occluded goals preserve their
   previous evidence timestamp; they are not newly confirmed. A changed active
   object invalidates prior slow advice and grasp/approach state before relocalizing.
6. Spectator Jev uses the same image/proprioception/todo boundary, plus actual
   execution events. Codex slow-think receives a detached todo and active goal.
7. Only the official environment success check establishes full success. It also
   completes remaining todo items with explicitly identified official evidence.

The original `phase_order` is retained as historical planner advice, not task
progress or Jev input. The old text-only Jev state/instruction builders were
removed. Controller geometry, library DSL recoveries and deterministic safety
gates remain; forced recoveries are still distinguishable in decision logs.

## Verification on 2026-10-19

- All 21 offline smoke scripts passed, including the full top-level harness with
  fake environment/model services and a start-plan failure scenario.
- `test_vision_todo_chain.py` adds 12 targeted cases covering todo initialization,
  visual progress/rollback, occlusion history, official completion, Astra model
  routing, actual image encoding and field isolation, required observations,
  TypeSafe request/response contracts, target identity and stationary reopening.
- The runtime chain verifies real image assets, todo and complete proprioception
  reach the Jev call, with no `observed_facts` or `experience_brief` payload.
- Syntax/import checks and whitespace validation pass.

These are code and offline execution results. No live Astra/Jev call or full
LIBERO episode was launched for this change. They do not establish improved
decisions, real rendering/physics compatibility or recovery/task success rates.
See [ADAPTER.md](../ADAPTER.md) for credentials, runtime requirements and input fields.
