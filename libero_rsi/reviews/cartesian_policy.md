# Jev-as-Policy Cartesian control migration

Base: `028b807` (PR #18, post-release finalization). Source design:
[YuanKJing/Jev-as-Policy](https://github.com/YuanKJing/Jev-as-Policy),
`jev_policy.py:56-108` and `app.py:194-218`, inspected 2026-10-20.
The MIT attribution is in the repository's `THIRD_PARTY_NOTICES.md`.

## Acting chain

1. The existing image-based acting request chooses an applicable library intent.
2. `executor.execute_policy` refreshes proprioception, images and the library
   reference point, and asks `jev_choice.choose_questions` once for four Choices:
   X/Y/Z each select positive, negative or stay; fingers select open, close or stay.
3. The motor reply drives `executor.execute` and `robot.servo`. A direction cannot
   move away from the current reference; each axis stops at its reference endpoint.
   A gripper reply that conflicts with the selected intent is refused, not silently
   converted to a different command. Identity, supported-grasp and release checks
   still apply. The close/confirm and open/settle procedures retain their existing
   behavior and step budgets.
4. Translation lasts at most eight environment control steps per decision. The
   controller uses the environment's `control_timestep`, with a nominal commanded
   per-axis cap of 0.14 m/s, or 0.06 m/s for descent/lowering. Existing gain scales
   this downward. This is a command cap, not a measured physical speed guarantee;
   conversion retains the harness's existing 0.05 m normalized OSC action scale.
   Orientation remains the existing commanded grasp yaw/tilt.
5. `reached`/`at_target` mean the full waypoint is within the existing 6 mm position
   tolerance. A progressing but unfinished block returns `in_progress`, never
   `timeout` or a completed task checkpoint. A motionless eight-step block still
   returns `stalled`. Official terminal/horizon guards remain in effect.
6. Another intent decision follows on fresh images. Partial blocks do not request
   routine outcome review or slow-think merely because their block ended. Existing
   measured no-progress recovery still applies. Completed actions and failures
   retain normal review/help behavior.

The previous CLI invocation now uses this path by default; no new feature flag is
required. Explicit recovery operations, optional pose-menu/fine-step commands,
relook, help and finish retain their own executors. Forced movement intents are
logged as forced and also pass through motor Choices when they are ordinary moves.
The complete system therefore still includes host recovery decisions.

## Deliberate differences from upstream

| Aspect | Upstream | This migration |
|---|---|---|
| Observations | MuJoCo object geometry/contact truth in text | Existing same-frame images, own-robot sensors and visual-estimate waypoints |
| Model/transport | Jev 1.13 SDK text | Existing Jev 1.14 image-capable multi-question transport |
| Intent names and targets | Cube-specific phases, fixed scene heights | Existing LIBERO library names/formulas and task todo |
| Motor timing | Physics runs concurrently with network inference | Synchronous bounded simulation blocks; physics pauses during API requests |
| Low-level control | Panda DLS IK and joint-target filtering | Existing LIBERO OSC controller and orientation handling |
| Completion | Demo-specific placement test | Official LIBERO success only |

This does not make Jev output arbitrary centimetres or free-form paths. The library
still computes the endpoint; Jev selects axis directions and the grip command.
No simulator object/goal pose or contact-object identity is added to policy input.
No upstream object height, barrier coordinate or fixed goal position is copied.

## Records and evaluation

`decisions.jsonl` records the motor context/questions, same-frame observation
serial, replies/probabilities/latency, and actual motion residual. Subsequent
observations include `motion_context.last_motor_decision`; target switches clear
that history. `result.json` separates `jev_intent_calls` and `jev_motor_calls`;
`jev_calls` is their sum. Todo-review and spectator counts remain separate.

Run with the same library, task/init, model, HORIZON and MAX_DEC as the baseline:

```bash
python libero_rsi/run_ep_fbpatch.py libero_object 1 0 \
  libero_rsi/libs/ee_pg_easy_v2_lib.json runs/cartesian
```

Compare full official success, environment steps, total calls including reviews,
wall time and forced recovery events. The existing 60-decision/600-step defaults
are unchanged; shorter motor blocks can hit the decision cap sooner. Report that
limit rather than interpreting a partial placement as success.

Offline verification command:

```bash
python -m unittest discover -s libero_rsi/smoke -q
```

New tests exercise combined-axis output, endpoint/sign limits, actual servo
feedback with deterministic sensor doubles, terminal/stall distinctions,
transport-grasp protection, API failure behavior and the complete episode entry
point across successive short motor blocks. These tests do not use real Jev,
LIBERO physics or paid services. Real rollout quality remains unmeasured.
