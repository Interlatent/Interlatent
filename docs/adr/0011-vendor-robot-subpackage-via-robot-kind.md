# 0011 — Vendor robot support as a subpackage selected by robot kind

- Status: Accepted
- Date: 2026-06-24

## Context

The node daemon drives a robot through a `control_loop(client, session,
should_stop, **_)` function. Until now there were two ways to pick that loop:

1. The bundled LeRobot wrapper (`lerobot_control_loop`), selected implicitly
   whenever a robot kind is set.
2. A fully custom loop, selected with `--loop module:function`, which the daemon
   resolves with `import_callable`.

Some robots are neither: they ship their own native (non-LeRobot) SDK, expose a
different motor/camera stack, and should **not** drag the `lerobot` package (and
its multi-GB transitive deps) into the node's import graph. The Almond Axol
dual-arm robot is the first such case — it talks to `almond_axol` over CAN and
opens GMSL ZED cameras by serial onboard a Jetson.

We needed a way to give a vendor robot a first-class, discoverable selector
without (a) importing lerobot for it, (b) forcing users to know an internal
`module:function` path, or (c) making the vendor's heavy, sometimes
Python-3.13-only dependency part of the base install.

## Decision

Vendor robots live in an optional subpackage under
`interlatent.adapters.<vendor>` and are selected by **robot kind** — the same
`--robot <name>` flag users already pass. The daemon holds a single registry
mapping a robot kind to the loop it imports lazily:

```python
# interlatent/node/daemon.py
_NATIVE_LOOPS = {
    "axol": "interlatent.adapters.axol:control_loop",
}
```

Resolution precedence in `_resolve_loop_fn`: an explicit `--loop module:fn`
override wins; else a robot kind registered in `_NATIVE_LOOPS` uses its native
loop; else the bundled LeRobot wrapper. The native loop reuses only the
LeRobot-free DRTC wire helpers from `interlatent.node.control` (so the recorded
observation payload is byte-identical to the built-in loop), and imports its
vendor SDK lazily inside the loop body. The vendor's dependencies are declared
as an optional extra (`pip install 'interlatent[axol]'`), never in the base
dependency set.

"Adapter" here means a **robot adapter** — distinct from a server-side policy
backend, a collection `--loop` adapter, or a LoRA adapter.

## Consequences

- A vendor robot is selected the same way as any built-in robot (`--robot axol`),
  with no internal path to memorize.
- The base `interlatent` install never imports the vendor SDK or `lerobot` on the
  native path; both are pulled only when the extra is installed and the loop
  actually runs. `import interlatent` stays adapter-free.
- The registry is the single, auditable place a vendor robot is wired in. Adding a
  vendor means adding a subpackage, a `_NATIVE_LOOPS` entry, and an extra.
- The vendor extra may carry sharper requirements than the core SDK (e.g. axol
  floors Python at 3.13 and is `uv`-friendly); these are documented on the extra,
  not imposed on the base package.
- Trade-off: `_NATIVE_LOOPS` is a hardcoded map in the daemon rather than a plugin
  entry-point discovery mechanism. We accept this — the set of first-party vendor
  robots is small and a static map is easier to read and audit than entry-point
  scanning. Third parties retain the generic `--loop module:function` escape hatch.
