# Going to cloud

The open-source runtime is complete: serve, drive, teleoperate, collect — all offline,
forever. [Interlatent Cloud](https://interlatent.com) exists for the walls you may
eventually hit:

- **"I don't own a GPU that serves Pi0 at low latency."** Hosted endpoints run on managed
  warm pools — no box to rent, no cold starts, no torch.compile babysitting.
- **"I want my data stored, versioned, and viewable."** Episodes record server-side into a
  hosted canonical LeRobot dataset per environment, with a dashboard episode viewer.
- **"I want my data labeled with rewards so I can actually train on it."** **Robometer**
  labels your episodes with dense rewards / value estimates — this is the part that doesn't
  exist in the OSS and isn't DIY-able from it.
- **"I have several robots and a team."** Nodes, session assignment, sharing, access
  control.

## The upgrade is one argument

```diff
 client = connect_drtc(
     environment="my-arm",
     policy_uri="lerobot/smolvla_base",
-    server_address="gpu-box:50051",
+    api_key="ilat_...",
     task="pick up the red cube",
 )
```

Your robot code, observation packing, and control loop are unchanged — the hosted endpoint
speaks the exact gRPC contract in [`proto/`](../proto). Datasets collected offline are
standard LeRobot v3.0 and can be imported later; datasets recorded hosted are exportable —
no lock-in in either direction.

Steps:

1. Sign up at [interlatent.com](https://interlatent.com) and create an API key.
2. Create an **environment** in the dashboard (one per robot/policy collection).
3. Pass `api_key=` (or set `INTERLATENT_API_KEY`) — see
   [examples/06_connect_hosted.py](../examples/06_connect_hosted.py).
4. Optional: pair always-on robots with `interlatent-node pair` so the dashboard can assign
   sessions to them.

## What stays true either way

- The client, server, teleop stack, and wire protocol in this repo are Apache-2.0 and work
  with zero account.
- The cloud consumes these same packages from PyPI — the OSS is the product, not a demo.
