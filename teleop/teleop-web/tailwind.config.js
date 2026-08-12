/** @type {import('tailwindcss').Config} */
// Theme tokens for the app shell, so VRTeleopOverlay's classes (bg-bg-panel,
// text-text-primary, status-* …) resolve. Trimmed to the tokens the teleop
// UI uses (blog paper/ink palette, display/serif faces dropped).
export default {
  future: {
    hoverOnlyWhenSupported: true,
  },
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-dark': '#000000',
        'bg-panel': '#141414',
        'bg-elevated': '#1f1f1f',
        'border-subtle': '#21262d',
        'border-emphasis': '#30363d',

        'primary': 'rgb(var(--primary-rgb) / <alpha-value>)',
        'primary-muted': '#1f6feb',
        'secondary': '#8b949e',

        'status-active': '#3fb950',
        'status-warning': '#d29922',
        'status-critical': '#f85149',
        'status-info': '#58a6ff',

        'text-primary': '#e6edf3',
        'text-secondary': '#9ca3af',
        'text-tertiary': '#7d8590',
        'text-quaternary': '#6e7681',
        'text-link': '#58a6ff',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'Consolas', 'monospace'],
      },
      letterSpacing: {
        'wide': '0.08em',
        'kicker': '0.18em',
      },
    },
  },
  plugins: [],
}
