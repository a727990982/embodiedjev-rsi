# Codex planning integration

The current harness uses one production assignment:

- Codex CLI: episode-start planning and synchronous/asynchronous slow-think.
- Astra: visual perception, RGB-D keypoints, and identity verification.
- Jev: action-intent selection and spectator decisions.

`coding_agent_slow.py` contains the planning schemas and the Codex CLI transport. Explicit mock calls support offline tests only. The runner does not read environment variables to change model roles, and it does not compare two planners or fall back to an Astra planner when Codex fails.

Early recovery follows Astra perception with synchronous Codex planning. The help-intent names in existing libraries remain stable, but invoke Codex. Workers use detached state/image snapshots and publish log events for the main thread.

Run the smoke scripts under `libero_rsi/smoke/`. A real rollout also requires LIBERO and the external Astra/Jev clients; see [ADAPTER.md](../ADAPTER.md). Model-call failures and offline tests do not establish simulator performance.

The [original pilot note](history/coding_agent_slow_pilot.md) is retained as historical design context, not current setup instructions.
