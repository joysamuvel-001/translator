import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Backend's own MedGemma/RunPod poll loop already waits up to 300s
        // (MAX_WAIT_SECONDS in medgemma_service.py) before it even decides
        // to fall back and respond. Diarization + ASR + translation happen
        // BEFORE that 300s clock starts, adding more time on top. If this
        // proxy timeout equals the backend's internal timeout, the proxy
        // kills the connection first and the frontend never sees the
        // response the backend was about to send — even on a successful
        // fallback. Give it real headroom: 300s backend timeout + 60s
        // buffer for everything upstream of MedGemma + response overhead.
        proxyTimeout: 360000, // 6 min
        timeout: 360000,
      },
    },
  },
})