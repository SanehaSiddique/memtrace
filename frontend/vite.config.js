import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const API_PREFIXES = ['/memory', '/agent', '/debug', '/evaluate', '/health', '/cost', '/demo', '/api']

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      ...Object.fromEntries(
        API_PREFIXES.map((prefix) => [prefix, { target: 'http://localhost:8000', changeOrigin: true }]),
      ),
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})

