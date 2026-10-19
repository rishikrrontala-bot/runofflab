import { defineConfig } from 'vite';

// Data generation and training write thousands of files under models/, evidence/ and data/; the dev server
// must not watch them (it serves only src/ and public/).
export default defineConfig({
  base: './',
  server: {
    proxy: { '/api': 'http://127.0.0.1:4173' },
    watch: { ignored: ['**/models/**', '**/evidence/**', '**/data/**', '**/.venv/**', '**/test-results/**', '**/.impeccable/**'] },
  },
  build: { chunkSizeWarningLimit: 650 },
});
