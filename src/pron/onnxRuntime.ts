import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as ort from "onnxruntime-node";
import { fileURLToPath } from "node:url";

type VocabJson = {
  itos: string[];
  stoi?: Record<string, number>;
  pad_id?: number;
  sos_id?: number;
  eos_id?: number;
  unk_id?: number;
};

type ConfigJson = {
  special_tokens: {
    src_pad_id: number;
    tgt_pad_id: number;
    sos_id: number;
    eos_id: number;
    src_unk_id?: number;
    tgt_unk_id?: number;
  };
  decode: {
    beam_size?: number;
    max_output_length?: number;
    length_alpha?: number;
  };
};

type RuntimeBundle = {
  encoder: ort.InferenceSession;
  decoder: ort.InferenceSession;
  srcVocab: VocabJson;
  tgtVocab: VocabJson;
  config: ConfigJson;
};

type BeamState = {
  tokens: number[];
  score: number; // cumulative log-prob
  prevH: ort.Tensor;
  ended: boolean;
};

const cachedBundles = new Map<string, RuntimeBundle>();

function buildStoi(vocab: VocabJson): Record<string, number> {
  if (vocab.stoi) return vocab.stoi;

  const stoi: Record<string, number> = {};
  vocab.itos.forEach((tok, i) => {
    stoi[tok] = i;
  });
  return stoi;
}

function normalizeWord(word: string): string {
  return word.trim().toLowerCase();
}

function encodeWord(
  word: string,
  srcVocab: VocabJson,
  config: ConfigJson,
): bigint[] {
  const stoi = buildStoi(srcVocab);
  const sos = srcVocab.sos_id ?? 1;
  const eos = srcVocab.eos_id ?? 2;
  const unk = srcVocab.unk_id ?? config.special_tokens.src_unk_id ?? 0;

  const chars = [...normalizeWord(word)];
  const ids = [sos, ...chars.map((ch) => stoi[ch] ?? unk), eos];
  return ids.map((v) => BigInt(v));
}

function decodeTokenIds(
  ids: number[],
  tgtVocab: VocabJson,
  config: ConfigJson,
): string {
  const eos = tgtVocab.eos_id ?? config.special_tokens.eos_id;
  const sos = tgtVocab.sos_id ?? config.special_tokens.sos_id;

  const chars: string[] = [];
  for (const id of ids) {
    if (id === sos) continue;
    if (id === eos) break;

    const tok = tgtVocab.itos[id];
    if (!tok) continue;
    if (tok === "<pad>" || tok === "<sos>" || tok === "<eos>") continue;

    chars.push(tok);
  }

  return chars.join("");
}

export function resolveBundledModelDir(metaUrl: string): string {
  const here = path.dirname(fileURLToPath(metaUrl));
  return path.resolve(here, "../../onnx_model");
}

function getEffectiveModelDir(modelDir?: string): string {
  return modelDir ?? resolveBundledModelDir(import.meta.url);
}

function lengthPenalty(length: number, alpha: number): number {
  if (alpha <= 0) return 1;
  return Math.pow((5 + length) / 6, alpha);
}

function decodedLength(tokens: number[], sos: number): number {
  let count = 0;
  for (const id of tokens) {
    if (id !== sos) count += 1;
  }
  return Math.max(count, 1);
}

function normalizedBeamScore(
  beam: BeamState,
  sos: number,
  alpha: number,
): number {
  const len = decodedLength(beam.tokens, sos);
  return beam.score / lengthPenalty(len, alpha);
}

function topKIndices(values: ArrayLike<number>, k: number): number[] {
  const pairs: Array<{ idx: number; value: number }> = [];

  for (let i = 0; i < values.length; i++) {
    pairs.push({ idx: i, value: Number(values[i]) });
  }

  pairs.sort((a, b) => b.value - a.value);
  return pairs.slice(0, k).map((p) => p.idx);
}

function logSoftmax(values: ArrayLike<number>): number[] {
  let maxVal = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = Number(values[i]);
    if (v > maxVal) maxVal = v;
  }

  let sumExp = 0;
  for (let i = 0; i < values.length; i++) {
    sumExp += Math.exp(Number(values[i]) - maxVal);
  }

  const logZ = maxVal + Math.log(sumExp);
  const out = new Array<number>(values.length);

  for (let i = 0; i < values.length; i++) {
    out[i] = Number(values[i]) - logZ;
  }

  return out;
}

export async function loadOnnxBundle(modelDir?: string): Promise<RuntimeBundle> {
  const effectiveModelDir = getEffectiveModelDir(modelDir);

  const cached = cachedBundles.get(effectiveModelDir);
  if (cached) return cached;

  const encoderPath = path.join(effectiveModelDir, "encoder.onnx");
  const decoderPath = path.join(effectiveModelDir, "decoder_step.onnx");
  const srcVocabPath = path.join(effectiveModelDir, "src_vocab.json");
  const tgtVocabPath = path.join(effectiveModelDir, "tgt_vocab.json");
  const configPath = path.join(effectiveModelDir, "config.json");

  const [srcText, tgtText, configText] = await Promise.all([
    fs.readFile(srcVocabPath, "utf-8"),
    fs.readFile(tgtVocabPath, "utf-8"),
    fs.readFile(configPath, "utf-8"),
  ]);

  const [encoder, decoder] = await Promise.all([
    ort.InferenceSession.create(encoderPath),
    ort.InferenceSession.create(decoderPath),
  ]);

  const bundle: RuntimeBundle = {
    encoder,
    decoder,
    srcVocab: JSON.parse(srcText),
    tgtVocab: JSON.parse(tgtText),
    config: JSON.parse(configText),
  };

  cachedBundles.set(effectiveModelDir, bundle);
  return bundle;
}

export async function predictHangulWithOnnx(
  word: string,
  modelDir?: string,
): Promise<string | null> {
  const bundle = await loadOnnxBundle(modelDir);
  const { encoder, decoder, srcVocab, tgtVocab, config } = bundle;

  const srcIds = encodeWord(word, srcVocab, config);
  const srcTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(srcIds),
    [1, srcIds.length],
  );

  const encRes = await encoder.run({
    src_ids: srcTensor,
  });

  const encOut = encRes.enc_out;
  const h0 = encRes.h0;
  const encMask = encRes.enc_mask;

  if (!encOut || !h0 || !encMask) return null;

  const sos = tgtVocab.sos_id ?? config.special_tokens.sos_id;
  const eos = tgtVocab.eos_id ?? config.special_tokens.eos_id;
  const maxLen = config.decode.max_output_length ?? 30;
  const beamSize = Math.max(1, config.decode.beam_size ?? 1);
  const alpha = config.decode.length_alpha ?? 0;

  let beams: BeamState[] = [
    {
      tokens: [sos],
      score: 0,
      prevH: h0,
      ended: false,
    },
  ];

  for (let step = 0; step < maxLen; step++) {
    const allCandidates: BeamState[] = [];

    for (const beam of beams) {
      if (beam.ended) {
        allCandidates.push(beam);
        continue;
      }

      const lastToken = beam.tokens[beam.tokens.length - 1];
      const yTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from([BigInt(lastToken)]),
        [1],
      );

      const decRes = await decoder.run({
        y_t: yTensor,
        prev_h: beam.prevH,
        enc_out: encOut,
        enc_mask: encMask,
      });

      const logits = decRes.logits;
      const nextH = decRes.next_h;
      if (!logits || !nextH) return null;

      const data = logits.data as Float32Array | number[];
      const logProbs = logSoftmax(data);
      const topIds = topKIndices(logProbs, beamSize);

      for (const tokenId of topIds) {
        allCandidates.push({
          tokens: [...beam.tokens, tokenId],
          score: beam.score + logProbs[tokenId],
          prevH: nextH,
          ended: tokenId === eos,
        });
      }
    }

    allCandidates.sort((a, b) => {
      return normalizedBeamScore(b, sos, alpha) - normalizedBeamScore(a, sos, alpha);
    });

    beams = allCandidates.slice(0, beamSize);

    if (beams.every((beam) => beam.ended)) {
      break;
    }
  }

  const finishedBeams = beams.filter((beam) => beam.ended);
  const finalCandidates = finishedBeams.length > 0 ? finishedBeams : beams;

  finalCandidates.sort((a, b) => {
    return normalizedBeamScore(b, sos, alpha) - normalizedBeamScore(a, sos, alpha);
  });

  const best = finalCandidates[0];
  const text = decodeTokenIds(best.tokens, tgtVocab, config).trim();

  return text || null;
}