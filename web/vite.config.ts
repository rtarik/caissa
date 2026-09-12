import { defineConfig } from "vite";

export default defineConfig({
  // Relative, so the build works from a GitHub Pages project subpath
  // (user.github.io/caissa/) without knowing the repository name.
  base: "./",
  build: { outDir: "dist", target: "es2022" },
  worker: { format: "es" },
});
