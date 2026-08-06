import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The hosted backend keeps a CORS allowlist (INTERLATENT_CORS_ORIGINS) that
// does not include dev origins, so a browser at http://localhost:3100 gets
// `400 Disallowed CORS origin` on the preflight. In dev we therefore proxy
// /api server-side — no preflight, no CORS. `client.ts` routes the hosted
// API base through here automatically; to hit your own backend instead run
// TELEOP_API_TARGET=http://localhost:8000 npm run dev.
const apiTarget = process.env.TELEOP_API_TARGET || 'https://interlatent.com'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3100,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
