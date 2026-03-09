// src/pron/latinTokorea.ts
import { customDictionary } from "./dictionary";
import { predictHangulWithOnnx } from "./onnxRuntime";

type Token = { type: "word" | "other"; text: string };

export type LatinToHangulOptions = {
  dictionary?: Record<string, string[]>;
  modelDir?: string;
};

type InternalOptions = {
  dictionaryOverride?: Record<string, string[]>;
  modelDir?: string;
};

export function latinToHangul(
  input: string,
  options?: LatinToHangulOptions,
): string {
  // sync에서는 dict만 처리
  const fullKey = input.trim().toLowerCase();
  const userFullHit = options?.dictionary?.[fullKey];
  if (userFullHit?.length) return userFullHit[0];

  const builtInFullHit = customDictionary[fullKey as keyof typeof customDictionary];
  if (builtInFullHit?.length) return builtInFullHit[0];

  const tokens = tokenizePreservingSpecialChars(input);
  return applyDictionaryOnly(tokens, options);
}

export async function latinToHangulAsync(
  input: string,
  options?: LatinToHangulOptions,
): Promise<string> {
  const fullKey = input.trim().toLowerCase();
  const userFullHit = options?.dictionary?.[fullKey];
  if (userFullHit?.length) return userFullHit[0];

  const builtInFullHit = customDictionary[fullKey as keyof typeof customDictionary];
  if (builtInFullHit?.length) return builtInFullHit[0];

  const resolved = resolveOptions(options);
  const tokens = tokenizePreservingSpecialChars(input);
  return await transliterateTokensAsync(tokens, resolved);
}

function tokenizePreservingSpecialChars(input: string): Token[] {
  const tokens: Token[] = [];
  const regex = /([a-zA-Z]+)|([^a-zA-Z]+)/g;
  while (true) {
    const match = regex.exec(input);
    if (!match) break;
    if (match[1]) tokens.push({ type: "word", text: match[1] });
    else if (match[2]) tokens.push({ type: "other", text: match[2] });
  }
  return tokens;
}

function resolveOptions(options?: LatinToHangulOptions): InternalOptions {
  return {
    dictionaryOverride: options?.dictionary,
    modelDir: options?.modelDir,
  };
}

function overrideCandidates(
  word: string,
  overrides?: Record<string, string[]>,
): string[] {
  const key = word.toLowerCase();

  const userHit = overrides?.[key];
  if (userHit?.length) return userHit.map(String);

  const builtIn = customDictionary[key as keyof typeof customDictionary];
  if (builtIn?.length) return builtIn.map(String);

  return [];
}

function applyDictionaryOnly(tokens: Token[], options?: LatinToHangulOptions): string {
  return tokens
    .map((token) => {
      if (token.type !== "word") return token.text;
      const hit = overrideCandidates(token.text, options?.dictionary);
      return hit[0] ?? token.text;
    })
    .join("");
}

async function transliterateTokensAsync(
  tokens: Token[],
  options: InternalOptions,
): Promise<string> {
  const out: string[] = [];

  for (const token of tokens) {
    if (token.type !== "word") {
      out.push(token.text);
      continue;
    }

    const dictHit = overrideCandidates(token.text, options.dictionaryOverride);
    if (dictHit.length > 0) {
      out.push(dictHit[0]);
      continue;
    }

    if (options.modelDir) {
      const pred = await predictHangulWithOnnx(token.text, options.modelDir);
      out.push(pred ?? token.text);
      continue;
    }

    out.push(token.text);
  }

  return out.join("");
}