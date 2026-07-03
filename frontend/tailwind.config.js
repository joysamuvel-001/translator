/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: '#12161B',
          surface: '#1A2027',
          raised: '#212833',
          border: '#2A323D',
        },
        text: {
          primary: '#EDF1F4',
          muted: '#8C96A3',
          faint: '#5C6673',
        },
        vital: {
          teal: '#2FD7A8',
          tealDim: '#1C8C6E',
          amber: '#F2A75C',
          coral: '#E06464',
          indigo: '#7C8CF8',
        },
      },
      fontFamily: {
        display: ['"Spectral"', 'serif'],
        body: ['"Inter"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      boxShadow: {
        panel: '0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.5)',
      },
      borderRadius: {
        xl2: '1.1rem',
      },
    },
  },
  plugins: [],
}
