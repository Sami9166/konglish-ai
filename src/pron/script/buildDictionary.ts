// scripts/build-dictionary-ts.ts
import fs from "node:fs/promises";
import path from "node:path";

const INPUT_JSON = path.resolve("./src/pron/script/dd.json");
const OUTPUT_TS = path.resolve("./src/pron/dictionary.ts");

function sortObjectByKey(obj: Record<string, string[]>) {
  return Object.fromEntries(
    Object.entries(obj).sort(([a], [b]) => a.localeCompare(b)),
  );
}

function normalizeDictionary(data: unknown): Record<string, string[]> {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("JSON must be an object: Record<string, string>");
  }

  const result: Record<string, string[]> = {};

  for (const [rawKey, rawValue] of Object.entries(data)) {
    const key = String(rawKey).trim().toLowerCase();
    if (!key) continue;

    const value = String(rawValue).trim();
    if (!value) continue;

    result[key] = [value];
  }

  return sortObjectByKey(result);
}

function toTypeScriptModule(dictObj: Record<string, string[]>) {
  return `export const customDictionary: Record<string, string[]> = ${JSON.stringify(dictObj, null, 2)} as const;\n`;
}

async function main() {
  const raw = await fs.readFile(INPUT_JSON, "utf-8");
  const parsed = JSON.parse(raw) as unknown;

  const normalized = normalizeDictionary(parsed);
  const tsCode = toTypeScriptModule(normalized);

  await fs.mkdir(path.dirname(OUTPUT_TS), { recursive: true });
  await fs.writeFile(OUTPUT_TS, tsCode, "utf-8");

  console.log(`OK: ${OUTPUT_TS}`);
  console.log(`entries: ${Object.keys(normalized).length}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});