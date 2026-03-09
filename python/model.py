import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb, hid, layers, dropout, pad_id):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=pad_id)
        self.norm = nn.LayerNorm(emb)   # 입력 전 정규화
        self.rnn = nn.GRU(
            emb, hid, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=(dropout if layers > 1 else 0.0)
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
        nn.init.normal_(self.emb.weight, std=0.01)
        for name, p in self.rnn.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name: nn.init.orthogonal_(p)
            elif 'bias' in name: nn.init.zeros_(p)

    def forward(self, x, xlen):
        e = self.dropout(self.norm(self.emb(x)))
        packed = nn.utils.rnn.pack_padded_sequence(e, xlen.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return out, h

class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.wa = nn.Linear(hid_dim, hid_dim * 2)

    def forward(self, dec_h, enc_out, enc_mask=None):
        wa_dec_h = self.wa(dec_h).unsqueeze(2)

        score = torch.bmm(enc_out, wa_dec_h).squeeze(2)

        if enc_mask is not None:
            score = score.masked_fill(~enc_mask, -1e10)

        attn_weights = F.softmax(score, dim=-1)

        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1) # [B, H*2]

        return ctx, attn_weights
    
class Decoder(nn.Module):
    def __init__(self, vocab_sz, emb, hid, layers, dropout, pad_id):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=pad_id)
        self.rnn = nn.GRU(emb, hid, num_layers=layers, batch_first=True, dropout=(dropout if layers > 1 else 0.0))
        self.attn = LuongAttention(hid)
        self.concat_linear = nn.Linear(hid * 3, hid)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.concat_linear.weight)
        nn.init.xavier_uniform_(self.out.weight)
        for name, p in self.rnn.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name: nn.init.orthogonal_(p)
            elif 'bias' in name: nn.init.zeros_(p)

    def step(self, y_t, h, enc_out, enc_mask):
        e = self.emb(y_t).unsqueeze(1)
        o, h = self.rnn(e, h)
        dec_h = o.squeeze(1)
        ctx, attn = self.attn(dec_h, enc_out, enc_mask)
        combined = self.tanh(self.concat_linear(torch.cat([dec_h, ctx], dim=-1)))
        logits = self.out(combined)
        return logits, h, attn
    
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_id, tgt_pad_id,
                 hp, dec_layers):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.dec_layers = dec_layers
        self.encoder = Encoder(src_vocab_size, hp["emb"], hp["hid"], hp["enc_layers"], hp["dropout"], src_pad_id)
        self.decoder = Decoder(tgt_vocab_size, hp["emb"], hp["hid"], dec_layers, hp["dropout"], tgt_pad_id)
        self.bridge = nn.Linear(hp["hid"] * 2, hp["hid"])

    def _init_dec_hidden(self, enc_h):
        fw, bw = enc_h[-2], enc_h[-1]
        h_combined = torch.tanh(self.bridge(torch.cat([fw, bw], dim=-1)))
        return h_combined.unsqueeze(0).expand(self.dec_layers, -1, -1).contiguous()

    def forward(self, src, src_len, tgt, teacher_forcing=0.9):
        enc_out, enc_h = self.encoder(src, src_len)
        enc_mask = (src != self.src_pad_id)
        dec_h = self._init_dec_hidden(enc_h)
        y_t = tgt[:, 0]
        logits_all = []
        for t in range(1, tgt.size(1)):
            logits, dec_h, _ = self.decoder.step(y_t, dec_h, enc_out, enc_mask)
            logits_all.append(logits.unsqueeze(1))
            y_t = tgt[:, t] if (torch.rand(1).item() < teacher_forcing) else torch.argmax(logits, dim=-1)
        return torch.cat(logits_all, dim=1)

    @torch.no_grad()
    def beam_decode(self, src_str, sv, tv, beam=5, max_out=30, length_alpha=0.6, device=None):
        self.eval()
        device = device or next(self.parameters()).device
        tokens = sv.encode(src_str)
        x = torch.tensor([tokens], device=device); xl = torch.tensor([len(tokens)], device=device)
        enc_out, enc_h = self.encoder(x, xl)
        enc_mask = (x != self.src_pad_id)
        h = self._init_dec_hidden(enc_h)
        beams = [([tv.sos_id], 0.0, h)]
        for _ in range(max_out):
            new_beams = []
            for seq, score, h_state in beams:
                if seq[-1] == tv.eos_id:
                    new_beams.append((seq, score, h_state)); continue
                logits, next_h, _ = self.decoder.step(torch.tensor([seq[-1]], device=device), h_state, enc_out, enc_mask)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_v, top_i = torch.topk(log_probs, beam)
                for i in range(beam):
                    new_beams.append((seq + [top_i[i].item()], score + top_v[i].item(), next_h))
            new_beams.sort(key=lambda b: b[1] / (len(b[0])**length_alpha), reverse=True)
            beams = new_beams[:beam]
            if all(b[0][-1] == tv.eos_id for b in beams): break
        best_seq = beams[0][0][1:-1] if beams[0][0][-1] == tv.eos_id else beams[0][0][1:]
        return tv.decode(best_seq)