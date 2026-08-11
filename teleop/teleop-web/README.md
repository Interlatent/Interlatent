# interlatent-teleop-web

A standalone, open-source **WebXR VR teleop producer** for Interlatent robot
sessions. Open it in the Meta Quest Browser, paste your `ilat_…` API key, and
enter VR: the headset's controllers drive the robot's end effector through the
hosted WebTransport/QUIC relay — grip is the clutch/deadman, trigger is the
gripper.

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

Dev origins are not on the backend's global CORS allowlist, so a direct call
from `http://localhost:3100` can fail the preflight with `400 Disallowed CORS
origin`. `npm run dev` therefore proxies `/api` to the backend server-side
(`vite.config.ts`), and the client routes the default API base through that
proxy automatically. Override the proxy target with
`TELEOP_API_TARGET=http://localhost:8000 npm run dev`.

The built app has no proxy and needs none: the backend's teleop CORS policy
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

- **API base URL** (`interlatent.apiBase`) — default `https://interlatent.com`
- **API key** (`interlatent.apiKey`) — your Interlatent `ilat_…` key, sent as
  the `x-api-key` header on every request

## Provenance / drift

The teleop engine here is **copied, by decision, from the Interlatent
dashboard** (`site/src/lib/teleop/*` and
`site/src/components/teleop/VRTeleopOverlay.tsx` in the Interlatent-Main
repo) rather than extracted into a shared package — the dashboard deploy
and this app have different release cadences and the code is dependency-free
by design. Copied files carry a header naming their source path and commit.
**Fixes must land in both copies**; when touching one, port the change to the
other. (`src/lib/teleop/kinematics.ts` and `src/lib/teleop/quicPoseSocket.ts`
carry no such header — check upstream before editing them.)

Local pieces: `src/lib/client.ts` (typed fetch client replacing the dashboard's
react-query `api.ts`), `src/App.tsx` (session picker + settings shell),
`src/components/StartRecordingPanel.tsx` (the create form), and the PWA
scaffolding.
