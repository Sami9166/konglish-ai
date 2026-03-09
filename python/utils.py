from pathlib import Path
import random
import unicodedata
import re
import torch

VOWELS = "aeiou"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def set_seed(seed: int):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_kr(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    # ensure b is shorter for memory
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(pred: str, gold: str) -> float:
    gold_len = max(1, len(gold))
    return levenshtein(pred, gold) / gold_len


def augment_en_text(
    text: str,
    apply_prob=0.25,
    delete_prob=0.03,
    swap_prob=0.03,
    vowel_prob=0.04,
    dup_drop_prob=0.02,
):
    """
    영어 입력 문자열에 아주 약한 노이즈를 넣는다.
    학습 시에만 사용.
    """
    if random.random() > apply_prob:
        return text

    chars = list(text)

    # 모음 치환
    for i, ch in enumerate(chars):
        low = ch.lower()
        if low in VOWELS and random.random() < vowel_prob:
            repl = random.choice(VOWELS.replace(low, ""))
            chars[i] = repl.upper() if ch.isupper() else repl

    # 문자 삭제
    out = []
    for ch in chars:
        if ch.isalpha() and random.random() < delete_prob:
            continue
        out.append(ch)
    chars = out if len(out) > 0 else chars

    # 인접한 문자 swap
    i = 0
    while i < len(chars) - 1:
        if (
            chars[i].isalpha()
            and chars[i + 1].isalpha()
            and random.random() < swap_prob
        ):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1

    # 중복/축소
    out = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        out.append(ch)

        if ch.isalpha() and random.random() < dup_drop_prob:
            # 50% 확률로 하나 더 복제, 50% 확률로 다음 같은 문자 하나 건너뜀
            if random.random() < 0.5:
                out.append(ch)
            elif i + 1 < len(chars) and chars[i + 1].lower() == ch.lower():
                i += 1

        i += 1

    aug = "".join(out)

    # 너무 망가졌으면 원문 유지
    if len(aug.strip()) == 0:
        return text

    return aug


def save_checkpoint(model, sv, tv, hp, base, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "hp": dict(hp),
        "base": {k: getattr(base, k) for k in dir(base) if not k.startswith("_")},
        "sv": {
            "pad_id": sv.pad_id,
            "sos_id": sv.sos_id,
            "eos_id": sv.eos_id,
            "itos": getattr(sv, "itos", None),
            "stoi": getattr(sv, "stoi", None),
        },
        "tv": {
            "pad_id": tv.pad_id,
            "sos_id": tv.sos_id,
            "eos_id": tv.eos_id,
            "itos": getattr(tv, "itos", None),
            "stoi": getattr(tv, "stoi", None),
        },
    }

    torch.save(ckpt, str(path))
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt
