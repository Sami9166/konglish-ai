from pathlib import Path
from model import Seq2Seq
import json
import torch
import torch.nn as nn
import torch.onnx

ENC_ONNX  = "encoder.onnx"
DEC_ONNX  = "decoder_step.onnx"

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.onnx


class EncoderONNX(nn.Module):
    def __init__(self, s2s):
        super().__init__()
        self.s2s = s2s

    def forward(self, src_ids):
        emb = self.s2s.encoder.emb(src_ids)
        emb = self.s2s.encoder.norm(emb)
        emb = self.s2s.encoder.dropout(emb)

        enc_out, enc_h = self.s2s.encoder.rnn(emb)
        enc_mask = (src_ids != self.s2s.src_pad_id)
        fw = enc_h[-2]
        bw = enc_h[-1]
        h_cat = torch.cat([fw, bw], dim=-1)
        h0_one = torch.tanh(self.s2s.bridge(h_cat))

        h0 = h0_one.unsqueeze(0).repeat(self.s2s.dec_layers, 1, 1)  # [L,B,H]

        return enc_out, h0, enc_mask


class DecoderStepONNX(nn.Module):
    def __init__(self, s2s):
        super().__init__()
        self.s2s = s2s

    def forward(self, y_t, prev_h, enc_out, enc_mask):
        logits, next_h, _attn = self.s2s.decoder.step(y_t, prev_h, enc_out, enc_mask)
        return logits, next_h


def dump_vocab(vocab_obj, out_path: Path):
    out_path.write_text(
        json.dumps(vocab_obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def convert_onnx(
    ckpt_path="./seq2seq_best_20260307_031335.pt",
    out_dir="./onnx_export",
    opset=17,
):
    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_onnx = out_dir / "encoder.onnx"
    dec_onnx = out_dir / "decoder_step.onnx"
    src_vocab_json = out_dir / "src_vocab.json"
    tgt_vocab_json = out_dir / "tgt_vocab.json"
    config_json = out_dir / "config.json"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hp"]
    sv = ckpt["sv"]
    tv = ckpt["tv"]
    sv['unk_id'] = 3
    tv['unk_id'] = 3

    src_vocab_size = len(sv["itos"]) if sv.get("itos") is not None else len(sv.get("stoi", {}))
    tgt_vocab_size = len(tv["itos"]) if tv.get("itos") is not None else len(tv.get("stoi", {}))

    s2s = Seq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_id=int(sv["pad_id"]),
        tgt_pad_id=int(tv["pad_id"]),
        hp=hp,
        dec_layers=int(hp["dec_layers"])
    ).eval()

    s2s.load_state_dict(ckpt["model_state"], strict=True)

    enc = EncoderONNX(s2s).eval()
    dec = DecoderStepONNX(s2s).eval()


    B, T = 1, 12
    src_ids = torch.randint(0, src_vocab_size, (B, T), dtype=torch.long)

    with torch.no_grad():
        enc_out, h0, enc_mask = enc(src_ids)

    y_t = torch.full((B,), int(tv["sos_id"]), dtype=torch.long)

    torch.onnx.export(
        enc,
        (src_ids,),
        str(enc_onnx),
        input_names=["src_ids"],
        output_names=["enc_out", "h0", "enc_mask"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "src_ids":  {0: "batch", 1: "src_len"},
            "enc_out":  {0: "batch", 1: "src_len"},
            "enc_mask": {0: "batch", 1: "src_len"},
            "h0":       {1: "batch"},
        },
        dynamo=False,
    )


    torch.onnx.export(
        dec,
        (y_t, h0, enc_out, enc_mask),
        str(dec_onnx),
        input_names=["y_t", "prev_h", "enc_out", "enc_mask"],
        output_names=["logits", "next_h"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "y_t":      {0: "batch"},
            "prev_h":   {1: "batch"},
            "enc_out":  {0: "batch", 1: "src_len"},
            "enc_mask": {0: "batch", 1: "src_len"},
            "logits":   {0: "batch"},
            "next_h":   {1: "batch"},
        },
        dynamo=False,
    )

    dump_vocab(sv, src_vocab_json)
    dump_vocab(tv, tgt_vocab_json)


    config = {
        "version": 1,
        "model_type": "seq2seq-gru-attn",
        "src_vocab_size": int(src_vocab_size),
        "tgt_vocab_size": int(tgt_vocab_size),
        "dims": {
            "emb": int(hp["emb"]),
            "hid": int(hp["hid"]),
            "enc_layers": int(hp["enc_layers"]),
            "dec_layers": int(hp["dec_layers"]),
            "dropout": float(hp["dropout"]),
        },
        "special_tokens": {
            "src_pad_id": int(sv["pad_id"]),
            "tgt_pad_id": int(tv["pad_id"]),
            "sos_id": int(tv["sos_id"]),
            "eos_id": int(tv["eos_id"]),
            "unk_id": int(tv["unk_id"]),
        },
        "decode": {
            "beam_size": int(hp.get("beam", 1)),
            "max_output_length": int(hp.get("max_out", 30)),
            "length_alpha": float(hp.get("length_alpha", 1.0)),
        },
        "onnx": {
            "encoder_file": "encoder.onnx",
            "decoder_file": "decoder_step.onnx",
            "encoder": {
                "inputs": {"src_ids": ["batch", "src_len"]},
                "outputs": {
                    "enc_out": ["batch", "src_len", int(hp["hid"] * 2)],
                    "h0": [int(hp["dec_layers"]), "batch", int(hp["hid"])],
                    "enc_mask": ["batch", "src_len"]
                }
            },
            "decoder_step": {
                "inputs": {
                    "y_t": ["batch"],
                    "prev_h": [int(hp["dec_layers"]), "batch", int(hp["hid"])],
                    "enc_out": ["batch", "src_len", int(hp["hid"] * 2)],
                    "enc_mask": ["batch", "src_len"]
                },
                "outputs": {
                    "logits": ["batch", int(tgt_vocab_size)],
                    "next_h": [int(hp["dec_layers"]), "batch", int(hp["hid"])]
                }
            }
        }
    }

    config_json.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )