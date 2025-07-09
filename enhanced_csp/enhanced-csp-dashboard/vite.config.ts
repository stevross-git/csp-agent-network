import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';   // ⬅️ NEW
import path from 'path';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),                              // ⬅️ NEW
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: true,  // allows LAN access (e.g. 192.168.x.x)
  },
});
