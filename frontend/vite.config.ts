/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev: the browser talks to Vite; /api is proxied to the FastAPI backend.
const backend = process.env.VITE_DEV_PROXY_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { '/api': { target: backend, changeOrigin: true } } },
  preview: { port: 4173, proxy: { '/api': { target: backend, changeOrigin: true } } },
  test: {
    environment: 'jsdom',
    globals: true,
    css: false,
  },
})
