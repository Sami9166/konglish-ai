import * as fs from "node:fs/promises";
import { Runtime } from "node:inspector/promises";
import * as path from "node:path";
import { RecordableHistogram } from "node:perf_hooks";
import * as ort from "onnxruntime-node";


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

let cachedBundle: RuntimeBundle | null = null;

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

function encodeWord(word: string, srcVocab: VocabJson, config: ConfigJson): bigint[] {
  const stoi = buildStoi(srcVocab);
  const sos = srcVocab.sos_id ?? 1;
  const eos = srcVocab.eos_id ?? 2;
  const unk = srcVocab.unk_id ?? config.special_tokens.src_unk_id ?? 0;

  const chars = [...normalizeWord(word)];
  const ids = [sos, ...chars.map((ch) => stoi[ch] ?? unk), eos];
  return ids.map((v) => BigInt(v));
}

function decodeTokenIds(ids: number[], tgtVocab: VocabJson, config: ConfigJson): string {
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

export async function loadOnnxBundle(modelDir: string): Promise<RuntimeBundle> {
  if (cachedBundle) return cachedBundle;

  const encoderPath = path.join(modelDir, "encoder.onnx");
  const decoderPath = path.join(modelDir, "decoder_step.onnx");
  const srcVocabPath = path.join(modelDir, "src_vocab.json");
  const tgtVocabPath = path.join(modelDir, "tgt_vocab.json");
  const configPath = path.join(modelDir, "config.json");

  const [srcText, tgtText, configText] = await Promise.all([
    fs.readFile(srcVocabPath, "utf-8"),
    fs.readFile(tgtVocabPath, "utf-8"),
    fs.readFile(configPath, "utf-8"),
  ]);

  const [encoder, decoder] = await Promise.all([
    ort.InferenceSession.create(encoderPath),
    ort.InferenceSession.create(decoderPath),
  ]);

  cachedBundle = {
    encoder,
    decoder,
    srcVocab: JSON.parse(srcText),
    tgtVocab: JSON.parse(tgtText),
    config: JSON.parse(configText),
  };

  return cachedBundle;
}

export async function predictHangulWithOnnx(
  word: string,
  modelDir: string,
): Promise<string | null> {
  const bundle = await loadOnnxBundle(modelDir);
  const { encoder, decoder, srcVocab, tgtVocab, config } = bundle;

  const srcIds = encodeWord(word, srcVocab, config);
  const srcTensor = new ort.Tensor("int64", BigInt64Array.from(srcIds), [1, srcIds.length]);

  const encFeeds: Record<string, ort.Tensor> = {
    src_ids: srcTensor,
  };

  const encRes = await encoder.run(encFeeds);
  const encOut = encRes.enc_out;
  const h0 = encRes.h0;
  const encMask = encRes.enc_mask;

  if (!encOut || !h0 || !encMask) return null;

  const sos = tgtVocab.sos_id ?? config.special_tokens.sos_id;
  const eos = tgtVocab.eos_id ?? config.special_tokens.eos_id;
  const maxLen = config.decode.max_output_length ?? 30;

  let prevH = h0;
  let currentId = sos;
  const outIds: number[] = [sos];

  for (let step = 0; step < maxLen; step++) {
    const yTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(currentId)]), [1]);

    const decRes = await decoder.run({
      y_t: yTensor,
      prev_h: prevH,
      enc_out: encOut,
      enc_mask: encMask,
    });

    const logits = decRes.logits;
    const nextH = decRes.next_h;
    if (!logits || !nextH) return null;

    const data = logits.data as Float32Array | number[];
    let bestIdx = 0;
    let bestVal = -Infinity;

    for (let i = 0; i < data.length; i++) {
      const v = Number(data[i]);
      if (v > bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    }

    currentId = bestIdx;
    outIds.push(currentId);
    prevH = nextH;

    if (currentId === eos) break;
  }

  const text = decodeTokenIds(outIds, tgtVocab, config).trim();
  return text || null;
}