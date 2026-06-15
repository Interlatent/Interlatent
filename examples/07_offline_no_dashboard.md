# 07 — Run sessions and record datasets with no dashboard

Drive an always-on robot **node** and collect LeRobot datasets entirely on your own
infrastructure, using the local **coordinator** instead of Interlatent Cloud. No account, no
callbacks.

Three roles (they can all be one machine for development):

- **GPU box** — runs `interlatent-serve` (the policy).
- **Coordinator host** — runs `interlatent up` (assigns sessions; holds the recording
  destination).
- **Node** — the robot machine (Pi) running `interlatent-node`.

## 1. Serve a policy (GPU box)

```bash
pip install 'interlatent-server[lerobot,s3]'   # drop [s3] if you only record locally
interlatent-serve --policy lerobot/smolvla_base
```

No API key needed — a self-hosted server trusts its network.

## 2. Start the coordinator (control host)

```bash
pip install interlatent

# Local directory destination (one growing, merged LeRobot dataset):
interlatent up --port 8900 --output-dir /data/so101-kitchen

# …or an S3-compatible bucket (AWS / Cloudflare R2 / MinIO):
# interlatent up --port 8900 \
#     --s3-uri s3://my-bucket/so101-kitchen \
#     --s3-endpoint-url https://<accountid>.r2.cloudflarestorage.com \
#     --s3-access-key $AWS_ACCESS_KEY_ID --s3-secret-key $AWS_SECRET_ACCESS_KEY

interlatent gpu add gpu0 100.x.y.z:50051       # the GPU box, reachable over LAN/tailnet
interlatent status
```

> If you start the coordinator with **no** destination, sessions run inference but are *not*
> recorded — `session start` warns you.

## 3. Pair and run the node (robot)

```bash
pip install interlatent

# No API key needed when pairing against your own coordinator.
interlatent-node pair --name arm0 --api-base http://<coordinator-host>:8900
interlatent-node run --robot so101 --port /dev/ttyACM0 --camera top=/dev/video0 \
    --api-base http://<coordinator-host>:8900
```

No SO-101 handy? Use the mock driver (`--robot so101 --robot-arg mock=true`) to exercise the
full path without hardware.

## 4. Drive a session (control host)

```bash
interlatent node ls          # confirm arm0 is "live"
interlatent session start --node arm0 --gpu gpu0 \
    --policy lerobot/smolvla_base --task "pick up the cube"
interlatent session ls
```

`session start` probes that the GPU box is reachable before assigning (use `--no-probe` to
skip). The node picks up the assignment on its next poll and starts driving the robot.

## 5. Stop — and collect the dataset

```bash
interlatent session stop <session-id>
```

Stopping **unassigns** the session: the node closes the DRTC session, and the GPU box builds
the episode and appends it into the canonical dataset at your destination
(`/data/so101-kitchen` or the S3 prefix). Run more sessions and they accumulate into the same
dataset — `meta/info.json`, `data/*.parquet`, and videos, ready for training.

```bash
interlatent down             # stop the coordinator (refuses while a session is active)
```

## How it fits together

- The coordinator is **only a control plane** — inference flows directly node↔GPU and keeps
  running even if the coordinator stops. Stopping a session is graceful by construction (it
  routes through the node's normal teardown), which is what triggers the dataset publish.
- Local/S3 destinations **merge-on-stop** into one flat LeRobot dataset; point one
  destination at one robot/policy collection.

See [docs/self-hosting.md](../docs/self-hosting.md) and
[ADR-0001](../docs/adr/0001-offline-coordinator-control-plane.md) /
[ADR-0002](../docs/adr/0002-recording-destination-via-session-metadata.md) for the design.
