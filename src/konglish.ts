// src/konglish.ts
import {
  type LatinToHangulOptions,
  latinToHangul,
  latinToHangulAsync,
} from "./pron/latinTokorea";

function mergeOptions(
  base?: LatinToHangulOptions,
  override?: LatinToHangulOptions,
): LatinToHangulOptions | undefined {
  if (!base) return override;
  if (!override) return base;
  return { ...base, ...override };
}

export class Konglish {
  constructor(private readonly defaultOptions: LatinToHangulOptions = {}) {}

  latinToHangul(input: string, options?: LatinToHangulOptions): string {
    return latinToHangul(input, mergeOptions(this.defaultOptions, options));
  }

  latinToHangulAsync(
    input: string,
    options?: LatinToHangulOptions,
  ): Promise<string> {
    return latinToHangulAsync(input, mergeOptions(this.defaultOptions, options));
  }
}