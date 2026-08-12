# interlatent-teleop-web

A standalone, open-source **WebXR VR teleop producer** for Interlatent robot
sessions. Open it in the Meta Quest Browser, point it at your coordinator,
paste that coordinator's `ilop_…` operator key, and enter VR: the headset's
controllers drive the robot's end effector through the WebTransport/QUIC
relay — grip is the clutch/deadman, trigger is the gripper.

Two ways in:

- **Start your own recording.** Pick a node and an environment and hit *Start
  recording* — a teleop recording needs no GPU/policy choice, so those two
  dropdowns are the whole form. The app waits out provisioning, enters VR when
  the recording goes live, and offers *Stop* when you're done.
- **Join something already running.** Any active inference session is joinable
  as a live intervention (the human takes over from the running policy, and
  those steps record as DAgger corrections).

No robot data ships with the app: the token mint returns the relay URL, and
the robot's kinematic spec is served by the node itself over the relay.

## Running

```sh
npm install
npm run dev        # http://localhost:3100 — fine for desktop smoke tests
```

For a headset you need a **secure context** — both WebXR and WebTransport
require HTTPS (localhost is exempt, but the Quest is not "localhost" to your
dev machine). Build and serve `dist/` over HTTPS:

```sh
npm run build
# serve dist/ with any static file server behind HTTPS
```

Production builds register a manifest + service worker, so Quest Browser will
offer to install it as an app (PWA). Dev builds skip the service worker.

### CORS

Dev origins are not on the coordinator's global CORS allowlist, so a direct
call from `http://localhost:3100` can fail the preflight with `400 Disallowed
CORS origin`. `npm run dev` therefore proxies `/api` to the coordinator
server-side (`vite.config.ts`), and the client routes the default API base
through that proxy automatically. Override the proxy target with
`TELEOP_API_TARGET=http://localhost:8000 npm run dev`.

The built app has no proxy and needs none: the coordinator's teleop CORS policy
(`INTERLATENT_TELEOP_CORS_ORIGINS`, default `["*"]`) admits any origin on
exactly the routes this app calls — the pickers, the two token mints, and
recording create/stop. So you can serve `dist/` from any HTTPS host without
registering that host anywhere first.

That policy is *path-scoped*: the global `INTERLATENT_CORS_ORIGINS` stays tight
for everything else, and the wildcard is safe here because the origin was never
the access control — every route is authenticated by an explicit `x-api-key`
header and the policy is uncredentialed, so an allowed origin gains nothing
without your key. A deployment that wants the origin back as a second factor
sets `INTERLATENT_TELEOP_CORS_ORIGINS` to an explicit list.

## Configuration

Everything is in the in-app Settings panel (gear icon), persisted to
localStorage:

- **API base URL** (`interlatent.coordinator`) — your coordinator's origin,
  e.g. `http://10.0.0.2:8900`. Required; there is no default, so the app never
  talks to a control plane you did not name.
- **API key** (`interlatent.apiKey`) — the `ilop_…` operator key that
  coordinator printed, sent as the `x-api-key` header on every request

## Layout

The teleop engine — `src/lib/teleop/*` (IK solver, clutch pose mapping, the
QUIC pose socket, the XR scene) and `src/components/VRTeleopOverlay.tsx` — is
dependency-free by design: plain WebXR, plain WebTransport, no framework glue,
so it can be lifted into another app as files rather than as a package.

Around it: `src/lib/client.ts` (typed fetch client over the coordinator API),
`src/App.tsx` (session picker + settings shell),
`src/components/StartRecordingPanel.tsx` (the create form), and the PWA
scaffolding.

## The engine files

`VRTeleopOverlay.tsx`, `xrScene.ts`, `quicPoseSocket.ts`, `quat.ts`,
`webtransport.ts`, `dlsSolver.ts`, `wristCalibration.ts`,
`clutchPoseMapper.ts`, `teleopProfiler.ts` and `kinematics.ts` are the WebXR
teleop engine: pose capture, clutch mapping, IK, and the datagram transport to
the relay. They have no counterpart elsewhere and no sync obligation — this
repo is where they live, and a change here is the change.

They are the only part of the app with real algorithmic content, so they carry
the test suite (`src/lib/teleop/__tests__/`) and are what the `teleop-web` CI
job typechecks and builds.
