# Contributing to Interlatent

The highest-impact contributions right now:

- **Add a robot** — wire a new arm/platform into the node control loop
- **Fix the fresh-clone experience** — anything that breaks `pip install` → first rollout
- Docs, examples, and latency/perf improvements

## Repo layout

```
packages/sdk/      pip: interlatent          import: interlatent          (robot-side client, node, CLI)
packages/server/   pip: interlatent-server   import: interlatent_server   (self-hosted DRTC policy server)
proto/             gRPC wire contract (source of truth for generated stubs)
examples/          runnable examples, ordered by hardware required
tests/             pytest suite for the SDK — runs with no GPU and no robot
teleop/teleop-web/ WebXR VR teleop producer (npm + vitest)
docker/            GPU server image
docs/              user documentation
```

## Dev setup

```bash
git clone https://github.com/interlatent/interlatent
cd interlatent
python3.11 -m venv .venv && source .venv/bin/activate
# Pillow + opencv give node/jpeg.py's fallback chain a CPU encoder; without one
# the node-jpeg tests fail. jsonschema is optional (its tests skip without it).
pip install -e ./packages/sdk Pillow opencv-python-headless pytest pytest-timeout ruff jsonschema
```

Run the tests (no GPU, no robot). **Both** SDK roots, as CI does —
`packages/sdk/tests/` is not a subset of `tests/`:

```bash
pytest tests/ packages/sdk/tests/
```

For the policy server: `pip install -e ./packages/server pytest pytest-timeout`, then
`pytest packages/server/tests/`. Full matrix in [TESTING.md](TESTING.md).

For real hardware and policies install `pip install -e './packages/sdk[lerobot]'`.
Inference runs on managed cloud GPU pods through the
[dashboard](https://interlatent.com) — you need an API key (`ilat_…`), not a local GPU.

## Changing the wire protocol

`proto/messages.proto` is the source of truth. The generated `*_pb2.py` stubs are committed
in both packages. After editing the proto:

```bash
pip install 'grpcio-tools==1.74.0'   # pinned: the stubs embed version stamps
./proto/gen_proto.sh
```

Protocol changes must stay backwards-compatible (the hosted cloud and older robots speak
the same contract). Add fields; don't renumber or repurpose existing ones.
`tests/test_proto_sync.py` fails the build if the mirrors drift. See
[proto/README.md](proto/README.md).

## Adding a robot

The node's control loop lives in `packages/sdk/src/interlatent/node/control.py` and wraps
LeRobot robot classes — if your robot is supported by LeRobot, it likely already works via
`--robot <type>`. For non-LeRobot hardware, write a native adapter under
`packages/sdk/src/interlatent/adapters/` (see [ROBOT.md](ROBOT.md)) or pass a custom loop
with `--loop module:fn`.

## Pull requests

- Keep PRs focused; separate refactors from behavior changes.
- Match the surrounding code's style. Lint with `ruff check .` and run
  `pytest tests/ packages/sdk/tests/` before pushing — CI runs both.
- Examples and docs count as code — if your change alters a public surface, update them.
- **Sign off your commits** (DCO): `git commit -s`. By signing off you certify the
  [Developer Certificate of Origin](https://developercertificate.org/) — that you have the
  right to submit the contribution under the project's Apache-2.0 license.

## Reporting issues

Use the issue templates. For latency/control problems, include: robot type, network path
(LAN/VPN/WAN), policy URI, and the session id (from `interlatent session ls`).

## Security

Do not open public issues for security vulnerabilities — email team@interlatent.com.
