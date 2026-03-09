from utils import norm_en, norm_kr, save_checkpoint
from train import BaseCFG, CharVocab, train_with_early_stopping, eval_on_test
from onnx import convert_onnx
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import json


if __name__ == "__main__":
    # load base configuration
    base = BaseCFG()

    with open(base.json_path, "r", encoding="utf-8") as fr:
        data = json.load(fr)

    df = pd.DataFrame({"en": list(data.keys()), "kr": list(data.values())})

    # normalize text
    df["en"] = df["en"].astype(str).map(norm_en)
    df["kr"] = df["kr"].astype(str).map(norm_kr)

    sv = CharVocab(set("".join(df["en"].tolist())))
    tv = CharVocab(set("".join(df["kr"].tolist())))

    # split train/val
    df = df.sample(frac=1.0, random_state=base.seed).reset_index(drop=True)
    val_n = max(1, int(len(df) * base.val_ratio))
    test_n = max(1, int(len(df) * base.test_ratio))
    val_df = df.iloc[:val_n].reset_index(drop=True)
    test_df = df.iloc[val_n : val_n + test_n].reset_index(drop=True)
    trn_df = df.iloc[val_n + test_n :].reset_index(drop=True)

    best_hp = {
        "hid": 384,
        "emb": 128,
        "dropout": 0.35,
        "beam": 3,
        "length_alpha": 0.7,
        "enc_layers": 3,
        "dec_layers": 2,
        "epochs": 40,
        "lr": 0.0005,
        "batch": 128,
        "clip": 1.0,
        "max_out": 30,
    }

    model, best_cer, history = train_with_early_stopping(
        trn_df, val_df, sv, tv, best_hp, base
    )

    # best validation CER
    print(f"BEST VAL CER: {best_cer}")

    # CER plot per epoch
    plt.plot([scores["val_cer"] for scores in history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val CER")
    plt.title("Early stopping validation curve")
    plt.grid(True)
    plt.show()

    # evaluate test dataset 
    test_res = eval_on_test(
        model,
        test_df,
        sv,
        tv,
        device=base.device,
        beam=best_hp["beam"],
        max_out=best_hp["max_out"],
        length_alpha=best_hp["length_alpha"],
        show_samples=50,
    )

    samples = test_res["samples"]
    accuracy = test_res["accuracy"]
    mean_cer = test_res["mean_cer"]

    print(f"\n🔥 오답 분석 (상위 20개 / Accuracy: {accuracy:.2f}%) 🔥")
    error_samples = [r for r in samples if r["cer"] > 0]

    if not error_samples:
        print("✨ 모든 샘플을 맞췄습니다! 오답이 없습니다.")
    else:
        print(
            f"{'No':<4} | {'English':<15} | {'Ground Truth':<12} | {'Prediction':<12} | {'CER':<6}"
        )
        print("-" * 65)
        for i, row in enumerate(error_samples[:20]):
            print(
                f"[{i+1:>2}] | {row['en']:<15} | {row['gt']:<12} | {row['pred']:<12} | {row['cer']:.4f}"
            )

    # save model
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_checkpoint(
        model=model,
        sv=sv,
        tv=tv,
        hp=best_hp,
        base=base,
        path=f"python/checkpoints/seq2seq_best_{stamp}.pt",
    )

    # convert model to onnx
    convert_onnx(f"python/checkpoints/seq2seq_best_{stamp}.pt", "./onnx_model")
