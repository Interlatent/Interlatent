import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// A coordinator's CORS allowlist (INTERLATENT_TELEOP_CORS_ORIGINS) will not
// include the dev origin, so a browser at http://localhost:3100 calling it
// directly gets `400 Disallowed CORS origin` on the preflight. Naming a target
// here proxies /api server-side instead — no preflight, no CORS:
//
//   TELEOP_API_TARGET=http://localhost:8900 npm run dev
//
// There is no default. Unset means no proxy is registered at all, and the app
// talks straight to whatever address you enter in Settings — the same rule the
// rest of the stack follows, where defaulting to somebody's control plane is
// how a self-hosted deployment ends up quietly phoning home. `client.ts` pairs
// with this: leave Settings blank and every call goes same-origin, through here.
const apiTarget = process.env.TELEOP_API_TARGET || ''

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3100,
    ...(apiTarget
      ? { proxy: { '/api': { target: apiTarget, changeOrigin: true } } }
      : {}),
  },
})
