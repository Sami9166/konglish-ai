import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  external: ["onnxruntime-node"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "es2020",
  platform: "node",
  treeshake: true,
  tsconfig: "./tsconfig.json",
});