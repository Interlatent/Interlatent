# Contributing to Interlatent

Thanks for helping make open robot infrastructure better. The highest-impact contributions
right now are:

- **Add a robot** — wire a new arm/platform into the node control loop
- **Fix the fresh-clone experience** — anything that breaks `pip install` → first rollout
- Docs, examples, and latency/perf improvements

## Repo layout

```
packages/sdk/      pip: interlatent          import: interlatent          (robot-side client, node, CLI)
proto/             gRPC wire contract (source of truth for generated stubs)
examples/          runnable examples, ordered by hardware required
tests/             pytest suite — runs with no GPU and no robot
docs/              user documentation
```

## Dev setup

```bash
git clone https://github.com/interlatent/interlatent
cd interlatent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ./packages/sdk pytest pytest-timeout ruff jsonschema
```

Run the tests (no GPU, no robot):

```bash
pytest tests/
```

For real hardware and policies install `pip install -e './packages/sdk[lerobot]'`.
Inference itself runs on managed cloud GPU pods through the
[dashboard](https://interlatent.com) — you need an API key (`ilat_…`), not a local GPU.

## Changing the wire protocol

`proto/messages.proto` is the source of truth. The generated `*_pb2.py` stubs are committed
in the SDK. After editing the proto:

```bash
./proto/gen_proto.sh
```

Protocol changes must stay backwards-compatible (the hosted cloud and older robots speak
the same contract). Add fields; don't renumber or repurpose existing ones.

## Adding a robot

This is the highest-leverage contribution. The node's control loop lives in
`packages/sdk/src/interlatent/node/control.py` and wraps LeRobot robot classes — if your
robot is supported by LeRobot, it likely already works via `--robot <type>`. For non-LeRobot
hardware, pass a custom loop with `--loop module:fn`.

## Pull requests

- Keep PRs focused; separate refactors from behavior changes.
- Match the surrounding code's style. Lint with `ruff check .` and run `pytest tests/`
  before pushing — CI runs both.
- Examples and docs count as code — if your change alters a public surface, update them.
- **Sign off your commits** (DCO): `git commit -s`. By signing off you certify the
  [Developer Certificate of Origin](https://developercertificate.org/) — that you have the
  right to submit the contribution under the project's Apache-2.0 license.

## Reporting issues

Use the issue templates. For latency/control problems, include: robot type, network path
(LAN/VPN/WAN), policy URI, and the session id (from `interlatent session ls`).

## Security

Do not open public issues for security vulnerabilities — email team@interlatent.com.
