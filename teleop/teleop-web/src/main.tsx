import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// PWA: tiny no-op-ish service worker (cache-first app shell) so Quest
// Browser offers "Install". Registered only where supported and only in
// production builds — a stale SW fights the Vite dev server.
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      /* SW is a nicety; the app works without it */
    });
  });
}
