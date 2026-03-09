from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
from utils import augment_en_text, cer
from model import Seq2Seq
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np


@dataclass
class BaseCFG:
    json_path: str = "new_dictionary.json"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    val_ratio: float = 0.1
    test_ratio: float = 0.2
    batch: int = 256

    epochs: int = 4
    lr: float = 2e-3
    clip: float = 1.0

    eval_samples: int = 500  # val에서 CER 평가할 샘플 수
    beam: int = 5
    max_out: int = 64
    length_alpha: float = 0.7

    es_patience = 5
    es_delta = 1e-3


PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


class CharVocab:
    def __init__(self, charset):
        self.itos = [PAD, SOS, EOS, UNK] + sorted(list(charset))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, s: str):
        return (
            [self.stoi[SOS]]
            + [self.stoi.get(ch, self.stoi[UNK]) for ch in s]
            + [self.stoi[EOS]]
        )

    def decode(self, ids):
        out = []
        for t in ids:
            if t == self.stoi[EOS]:
                break
            if t in (self.stoi[PAD], self.stoi[SOS], self.stoi[UNK]):
                continue
            out.append(self.itos[t])
        return "".join(out)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def sos_id(self):
        return self.stoi[SOS]

    @property
    def eos_id(self):
        return self.stoi[EOS]

    @property
    def unk_id(self):
        return self.stoi[UNK]

    def __len__(self):
        return len(self.itos)


class PairDS(Dataset):
    def __init__(self, df, sv, tv, train=False, noise_cfg=None):
        self.src = df["en"].tolist()
        self.tgt = df["kr"].tolist()
        self.sv = sv
        self.tv = tv
        self.train = train
        self.noise_cfg = noise_cfg or {}

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        src_txt = self.src[i]

        if self.train:
            src_txt = augment_en_text(
                src_txt,
                apply_prob=self.noise_cfg.get("apply_prob", 0.20),
                delete_prob=self.noise_cfg.get("delete_prob", 0.02),
                swap_prob=self.noise_cfg.get("swap_prob", 0.02),
                vowel_prob=self.noise_cfg.get("vowel_prob", 0.03),
                dup_drop_prob=self.noise_cfg.get("dup_drop_prob", 0.01),
            )

        s = self.sv.encode(src_txt)
        t = self.tv.encode(self.tgt[i])

        return (
            torch.tensor(s, dtype=torch.long),
            len(s),
            torch.tensor(t, dtype=torch.long),
            len(t),
        )


def collate(batch, src_pad, tgt_pad):
    xs, x_lens, ys, y_lens = zip(*batch)

    x_lens = torch.tensor(x_lens, dtype=torch.long)
    y_lens = torch.tensor(y_lens, dtype=torch.long)

    max_x = int(x_lens.max())
    max_y = int(y_lens.max())

    x_pad = torch.full((len(batch), max_x), src_pad, dtype=torch.long)
    y_pad = torch.full((len(batch), max_y), tgt_pad, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, : len(x)] = x
        y_pad[i, : len(y)] = y

    return x_pad, x_lens, y_pad, y_lens


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0
        self.best_state = None

    def step(self, value, model):
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        else:
            self.count += 1
            return self.count >= self.patience


def evaluate(
    model: Seq2Seq,
    val_df: PairDS,
    sv: CharVocab,
    tv: CharVocab,
    crit,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """criterion 평가 함수

    Args:
        model (Seq2Seq): _model to evaluate_
        val_df (PairDS): _dataset_
        sv (CharVocab): _source vocab_
        tv (CharVocab): _target vocab_
        crit: _criterion_
        device (torch.device): _eval device_
        batch_size (int, optional): _data batch_. Defaults to 256.

    Returns:
        float: _eval score_
    """
    loader = torch.utils.data.DataLoader(
        PairDS(val_df, sv, tv),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, sv.pad_id, tv.pad_id),
    )

    model.eval()
    total_loss = 0.0
    total_tok = 0

    for x, xl, y, _ in loader:
        x, xl, y = x.to(device), xl.to(device), y.to(device)

        logits = model(x, xl, y, teacher_forcing=1.0)  # CE 평가용: TF=1 고정
        tgt = y[:, 1:]

        loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        n_tok = (tgt != tv.pad_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tok += n_tok

    return total_loss / max(1, total_tok)


def get_tf_ratio(epoch: int, total_epochs: int) -> float:
    """epoch 별 teacher forcing ratio를 변경하는 함수

    Args:
        epoch (int): _current epoch_
        total_epochs (int): _total epoch_

    Returns:
        float: _teacher forcing ratio_
    """
    if epoch <= int(total_epochs * 0.25):
        return 0.95
    elif epoch <= int(total_epochs * 0.50):
        return 0.85
    elif epoch <= int(total_epochs * 0.75):
        return 0.80
    else:
        return 0.75


def train_with_early_stopping(
    trn_df: PairDS,
    val_df: PairDS,
    sv: CharVocab,
    tv: CharVocab,
    hp: Dict[str, Any],
    base: BaseCFG,
) -> Tuple[Seq2Seq, float, List[float]]:
    """early stopping을 활용한 train 함수

    Args:
        trn_df (PairDS): _train dataset_
        val_df (PairDS): _validation dataset_
        sv (CharVocab): _source vocab_
        tv (CharVocab): _target vocab_
        hp (Dict[str, Any]): _hyperparameter_
        base (BaseCFG): _base configuration_

    Returns:
       Tuple[Seq2Seq, float, List[float]]: _model, best cer, cer per epochs_
    """
    device = base.device

    model = Seq2Seq(
        src_vocab_size=len(sv),
        tgt_vocab_size=len(tv),
        src_pad_id=sv.pad_id,
        tgt_pad_id=tv.pad_id,
        hp=hp,
        dec_layers=hp["dec_layers"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"])

    crit = nn.CrossEntropyLoss(ignore_index=tv.pad_id, label_smoothing=0.03)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.2, patience=1, threshold=0.0001
    )

    noise_cfg = {
        "apply_prob": 0.20,
        "delete_prob": 0.02,
        "swap_prob": 0.02,
        "vowel_prob": 0.03,
        "dup_drop_prob": 0.01,
    }

    trn_loader = torch.utils.data.DataLoader(
        PairDS(trn_df, sv, tv, train=True, noise_cfg=noise_cfg),
        batch_size=hp["batch"],
        shuffle=True,
        collate_fn=lambda b: collate(b, sv.pad_id, tv.pad_id),
    )

    stopper = EarlyStopping(patience=base.es_patience, min_delta=base.es_delta)
    history = []

    for epoch in tqdm(range(1, hp["epochs"] + 1)):
        tf_ratio_epoch = get_tf_ratio(epoch, hp["epochs"])

        model.train()
        for x, xl, y, yl in trn_loader:
            x = x.to(device)
            xl = xl.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x, xl, y, teacher_forcing=tf_ratio_epoch)
            tgt = y[:, 1:]

            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip"])
            opt.step()

        model.eval()
        val_ce = evaluate(
            model, val_df, sv, tv, crit=crit, device=device, batch_size=hp["batch"]
        )

        val_sample = val_df.sample(n=min(1000, len(val_df)), random_state=42)
        cers = []

        with torch.no_grad():
            for en, kr in zip(val_sample["en"], val_sample["kr"]):
                pred = model.beam_decode(
                    en,
                    sv,
                    tv,
                    beam=hp["beam"],
                    max_out=hp["max_out"],
                    length_alpha=hp["length_alpha"],
                    device=device,
                )
                cers.append(cer(pred, kr))

        val_cer = float(np.mean(cers))
        current_lr = opt.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "val_ce": val_ce,
                "val_cer": val_cer,
            }
        )

        print(
            f"[epoch {epoch}] "
            f"LR={current_lr:.6f} | TF={tf_ratio_epoch:.2f} | "
            f"val CE={val_ce:.4f} | val CER={val_cer:.4f}"
        )

        scheduler.step(val_cer)
        if stopper.step(val_cer, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return model, stopper.best, history


@torch.no_grad()
def eval_on_test(
    model: Seq2Seq,
    test_df: PairDS,
    sv: CharVocab,
    tv: CharVocab,
    device: torch.device,
    beam: int = 3,
    max_out: int = 64,
    length_alpha: float = 0.7,
    show_samples: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """test dataset evaluation

    Args:
        model (Seq2Seq): _model to evaluate_
        test_df (PairDS): _test dataset_
        sv (CharVocab): _source vocab_
        tv (CharVocab): _target vocab_
        device (torch.device): _torch device_
        beam (int, optional): _beam search value_. Defaults to 3.
        max_out (int, optional): _max output length_. Defaults to 64.
        length_alpha (float, optional): _length alpha value_. Defaults to 0.7.
        show_samples (int, optional): _number of samples to show_. Defaults to 20.
        seed (int, optional): _seed value_. Defaults to 42.

    Returns:
        Dict[str, Any]: _mean cer, correct / wrong count, accuracy, cer list, sample list_
    """
    model.eval()
    cers = []
    rows = []
    correct_count = 0  # 완전 일치 단어의 수
    total_count = len(test_df)

    rng = np.random.default_rng(seed)
    idxs = np.arange(total_count)
    rng.shuffle(idxs)
    show_set = set(idxs[: min(show_samples, total_count)].tolist())

    for i, (en, kr) in enumerate(
        tqdm(zip(test_df["en"], test_df["kr"]), total=total_count, desc="TEST")
    ):
        pred = model.beam_decode(
            en,
            sv=sv,
            tv=tv,
            beam=beam,
            max_out=max_out,
            length_alpha=length_alpha,
            device=device,
        )

        c = cer(pred, kr)
        cers.append(c)

        if pred.strip() == kr.strip():
            correct_count += 1

        if i in show_set:
            rows.append({"en": en, "gt": kr, "pred": pred, "cer": c})

    wrong_count = total_count - correct_count
    accuracy = (correct_count / total_count) * 100
    mean_cer = float(np.mean(cers))

    print("\n" + "=" * 30)
    print(f"TEST RESULT")
    print(f"Correct Words: {correct_count} / {total_count}")
    print(f"Wrong Words  : {wrong_count}")
    print(f"Accuracy     : {accuracy:.2f}%")
    print(f"Mean CER     : {mean_cer:.4f}")
    print("=" * 30)

    # CER이 높은 순서대로 샘플 정렬
    rows = sorted(rows, key=lambda r: r["cer"], reverse=True)

    return {
        "mean_cer": mean_cer,
        "correct": correct_count,
        "wrong": wrong_count,
        "accuracy": accuracy,
        "cers": cers,
        "samples": rows,
    }
