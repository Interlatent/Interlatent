# Contributing to Interlatent

Thanks for helping make open robot infrastructure better. The highest-impact contributions
right now are:

- **Add a robot** — wire a new arm/platform into the node control loop and teleop driver
- **Add a policy backend** — make a new policy family servable by `interlatent-server`
- **Fix the fresh-clone experience** — anything that breaks `pip install` → first rollout
- Docs, examples, and latency/perf improvements

## Repo layout

```
packages/sdk/      pip: interlatent          import: interlatent          (robot-side client)
packages/server/   pip: interlatent-server   import: interlatent_server   (GPU inference server)
packages/teleop/   pip: interlatent-teleop   import: interlatent_teleop   (laptop ↔ Pi teleop)
proto/             gRPC wire contract (source of truth for generated stubs)
examples/          runnable examples, ordered by hardware required
docs/              user documentation
docker/            CUDA image for the inference server
```

## Dev setup

```bash
git clone https://github.com/interlatent/interlatent
cd interlatent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ./packages/sdk -e ./packages/server -e ./packages/teleop
```

The three packages use distinct import paths (`interlatent`, `interlatent_server`,
`interlatent_teleop`) and co-install cleanly.

Run the server locally (no GPU needed — the built-in `echo` backend works on CPU):

```bash
interlatent-serve --port 50051
python examples/01_loopback_no_hardware.py   # exercises the full client↔server path
```

For real policies install `pip install -e './packages/server[lerobot]'` (needs a CUDA GPU
for VLA-class policies).

## Changing the wire protocol

`proto/messages.proto` is the source of truth. The generated `*_pb2.py` stubs are committed
in both packages. After editing the proto:

```bash
./proto/gen_proto.sh
```

Protocol changes must stay backwards-compatible (the hosted cloud and older robots speak
the same contract). Add fields; don't renumber or repurpose existing ones.

## Adding a policy backend

Backends live in `packages/server/src/interlatent_server/server/`. Register one with the
`@register_backend("name")` decorator in `policy_runtime.py` and implement
`forward(observation, prior_actions, **kw) -> np.ndarray` returning a `(chunk_size,
action_dim)` action chunk. `lerobot_backend.py` is the reference implementation; keep heavy
imports lazy (inside `__init__`) so the server stays importable without your dependency.

## Adding a robot

The node's control loop lives in `packages/sdk/src/interlatent/node/control.py` and wraps
LeRobot robot classes — if your robot is supported by LeRobot, it likely already works via
`--robot <type>`. For non-LeRobot hardware, pass a custom loop with `--loop module:fn`, or
add a driver under `packages/teleop/src/interlatent_teleop/pi/` for teleop support.

## Pull requests

- Keep PRs focused; separate refactors from behavior changes.
- Match the surrounding code's style. Lint with `ruff check packages/`.
- Examples and docs count as code — if your change alters a public surface, update them.
- **Sign off your commits** (DCO): `git commit -s`. By signing off you certify the
  [Developer Certificate of Origin](https://developercertificate.org/) — that you have the
  right to submit the contribution under the project's Apache-2.0 license.

## Reporting issues

Use the issue templates. For latency/control problems, include: robot type, network path
(LAN/Tailscale/WAN), GPU, policy URI, and the server log around the session.

## Security

Do not open public issues for security vulnerabilities — email team@interlatent.com.
