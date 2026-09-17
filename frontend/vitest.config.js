import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// Test-only config. The app build stays on CRA/craco; Vitest (via the Vite
// pipeline + @vitejs/plugin-react) runs the same JSX sources under jsdom.
export default defineConfig({
  plugins: [react()],
  // Mirror CRA's jsconfig.json "@/*" path alias.
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/setupTests.js',
    css: false,
    include: ['src/**/*.{test,spec}.{js,jsx}'],
  },
});
