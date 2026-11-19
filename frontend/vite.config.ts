import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // The browser calls /api on the same origin; Vite forwards it to FastAPI.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // Keep the 3D libraries in their own file so the text of the page loads first.
        manualChunks(id) {
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id) || id.includes("vite/preload-helper")) return "react";
          if (/node_modules\/(gsap|lenis)\//.test(id)) return "motion";
          if (id.includes("node_modules")) return "three";
        },
      },
    },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
