# interlatent-teleop-web

A standalone, open-source **WebXR VR teleop producer** for Interlatent robot
sessions. Open it in the Meta Quest Browser, paste your `ilat_…` API key,
pick an active inference session (or teleop recording), and enter VR: the
headset's controllers drive the robot's end effector through the hosted
WebTransport/QUIC relay — grip is the clutch/deadman, trigger is the gripper.

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

Because it registers a manifest + service worker, Quest Browser will offer
to install it as an app (PWA).

## Configuration

Everything is in the in-app Settings panel (gear icon), persisted to
localStorage:

- **API base URL** — default `https://interlatent.com`
- **API key** — your Interlatent `ilat_…` key, sent as the `x-api-key`
  header on every request

## Provenance / drift

The teleop engine here is **copied, by decision, from the Interlatent
dashboard** (`site/src/lib/teleop/*` and
`site/src/components/teleop/VRTeleopOverlay.tsx` in the Interlatent-Main
repo) rather than extracted into a shared package — the dashboard deploy
and this app have different release cadences and the code is dependency-free
by design. Every copied file carries a header naming its source path and
commit. **Fixes must land in both copies**; when touching one, port the
change to the other.

Local pieces (not copies): `src/lib/client.ts` (typed fetch client replacing
the dashboard's react-query `api.ts`), `src/App.tsx` (session picker +
settings shell), and the PWA scaffolding.
