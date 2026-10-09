import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    langchain: "src/langchain.ts",
    strands: "src/strands.ts",
  },
  format: ["esm", "cjs"],
  target: "node20",
  platform: "node",
  dts: true,
  clean: true,
  sourcemap: true,
  splitting: true,
  // The optional frameworks are peer dependencies and must never be bundled.
  external: [/^@langchain\//, /^@strands-agents\//, "zod"],
});
