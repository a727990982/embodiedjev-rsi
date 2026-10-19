# Codex planning integration

The current harness uses one production assignment:

- Codex CLI: invocation transport and synchronous/asynchronous slow-think.
- Astra via Codex CLI: start planning/todo, visual perception, RGB-D keypoints, identity verification and todo progress review.
- TypeSafe Jev 1.14: image/todo/proprioception action selection and spectator decisions.

`coding_agent_slow.py` contains the planning/perception schemas and shared Codex CLI transport. Visual perception and marked-grasp verification explicitly select `gpt-6-astra`; the slow-think model can be configured separately. Start/todo calls also pin Astra. Mock backends are removed; offline tests use literal replies and fake CLI processes. The runner does not read environment variables to change model roles, and it does not compare two planners or fall back to a direct Astra client when Codex fails.

Early recovery follows Astra perception with synchronous Codex planning. The help-intent names in existing libraries remain stable, but invoke Codex. Workers use detached state/image snapshots and publish log events for the main thread.

Run the smoke scripts under `libero_rsi/smoke/`. A real rollout also requires LIBERO, Codex CLI, and TypeSafe credentials for the bundled Jev client; see [ADAPTER.md](../ADAPTER.md). Model-call failures and offline tests do not establish simulator performance.

The [original pilot note](history/coding_agent_slow_pilot.md) is retained as historical design context, not current setup instructions.
