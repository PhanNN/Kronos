"""Compare the pretrained and finetuned Kronos models on the held-out test split.

Rolls a forecast origin through the test period: at each origin the models see the
previous `lookback_window` rows and forecast the next `predict_window` rows, which are
compared against the actual data. A naive "last close stays flat" forecast is included
as a baseline -- a model that cannot beat it is not adding information.

Usage (from inside finetune_csv/):
    python evaluate.py --config configs/config_vn30_daily.yaml
    python evaluate.py --config configs/config_vn30_daily.yaml --stride 1 --sample-count 20 --device cpu
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from config_loader import CustomFinetuneConfig

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def resolve(path):
    """Config paths are relative to finetune_csv/; Hugging Face IDs are left as-is."""
    if os.path.isabs(path):
        return path
    local = os.path.normpath(os.path.join(HERE, path))
    return local if path.startswith(".") or os.path.exists(local) else path


def make_origins(df, config, stride):
    n = len(df)
    test_start = int(n * (config.train_ratio + config.val_ratio))
    first = max(test_start, config.lookback_window)
    last = n - config.predict_window
    if last < first:
        raise ValueError(f"Test split too short: needs at least {config.predict_window} rows, has {n - test_start}")
    return list(range(first, last + 1, stride))


def forecast(predictor, df, origins, config, args):
    lookback, pred_len = config.lookback_window, config.predict_window
    torch.manual_seed(args.seed)
    preds = []
    for i in range(0, len(origins), args.batch_size):
        batch = origins[i:i + args.batch_size]
        preds += predictor.predict_batch(
            df_list=[df.loc[t - lookback:t - 1, FEATURES].reset_index(drop=True) for t in batch],
            x_timestamp_list=[df.loc[t - lookback:t - 1, "timestamps"].reset_index(drop=True) for t in batch],
            y_timestamp_list=[df.loc[t:t + pred_len - 1, "timestamps"].reset_index(drop=True) for t in batch],
            pred_len=pred_len,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=args.sample_count,
            verbose=False,
        )
        print(f"  {min(i + args.batch_size, len(origins))}/{len(origins)} origins")
    return [p["close"].to_numpy() for p in preds]


def score(df, origins, pred_closes, pred_len):
    rows = []
    for t, pred in zip(origins, pred_closes):
        actual = df["close"].iloc[t:t + pred_len].to_numpy()
        last = df["close"].iloc[t - 1]
        err = pred - actual
        rows.append({
            "origin": df["timestamps"].iloc[t].date(),
            "mae": np.abs(err).mean(),
            "mape_%": (np.abs(err) / actual).mean() * 100,
            "end_error_%": (pred[-1] - actual[-1]) / actual[-1] * 100,
            "direction_hit": np.sign(pred[-1] - last) == np.sign(actual[-1] - last),
            "next_day_hit": np.sign(pred[0] - last) == np.sign(actual[0] - last),
        })
    return pd.DataFrame(rows)


def plot(df, origins, results, config, stride, path):
    pred_len = config.predict_window
    start = origins[0] - config.lookback_window // 2
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["timestamps"].iloc[start:], df["close"].iloc[start:], color="black", linewidth=1.5, label="Actual")
    colors = {"pretrained": "tab:blue", "finetuned": "tab:red"}
    # non-overlapping origins only, so the chart stays readable
    shown = range(0, len(origins), -(-pred_len // stride))
    for name, pred_closes in results.items():
        for k, idx in enumerate(shown):
            t = origins[idx]
            ax.plot(df["timestamps"].iloc[t:t + pred_len], pred_closes[idx],
                    color=colors.get(name), linewidth=1.2, alpha=0.8, label=name if k == 0 else None)
    ax.axvline(df["timestamps"].iloc[origins[0]], color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Close: {config.predict_window}-step forecasts over the test period")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stride", type=int, default=5, help="rows between forecast origins")
    parser.add_argument("--sample-count", type=int, default=10, help="sampled paths averaged per forecast")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda:0 / mps / cpu (auto-detected if omitted)")
    args = parser.parse_args()

    config = CustomFinetuneConfig(args.config)
    df = pd.read_csv(resolve(config.data_path), parse_dates=["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    origins = make_origins(df, config, args.stride)
    print(f"Test period {df['timestamps'].iloc[origins[0]].date()} -> {df['timestamps'].iloc[-1].date()}, "
          f"{len(origins)} forecast origins, lookback={config.lookback_window}, horizon={config.predict_window}")

    candidates = {
        "pretrained": (config.pretrained_tokenizer_path, config.pretrained_predictor_path),
        "finetuned": (resolve(config.tokenizer_best_model_path), resolve(config.basemodel_best_model_path)),
    }
    results = {}
    for name, (tok_path, model_path) in candidates.items():
        if name == "finetuned" and not (os.path.isdir(tok_path) and os.path.isdir(model_path)):
            print(f"Skipping finetuned: no checkpoints at {tok_path} / {model_path}")
            continue
        print(f"Forecasting with {name} ({model_path})")
        predictor = KronosPredictor(Kronos.from_pretrained(model_path), KronosTokenizer.from_pretrained(tok_path),
                                    device=args.device, max_context=config.max_context, clip=config.clip)
        results[name] = forecast(predictor, df, origins, config, args)

    naive = [np.full(config.predict_window, df["close"].iloc[t - 1]) for t in origins]
    scores = {name: score(df, origins, preds, config.predict_window) for name, preds in results.items()}
    scores["naive (flat)"] = score(df, origins, naive, config.predict_window)

    summary = pd.DataFrame({
        name: {
            "close MAE": s["mae"].mean(),
            "close MAPE %": s["mape_%"].mean(),
            "mean end error %": s["end_error_%"].mean(),
            "direction accuracy %": s["direction_hit"].mean() * 100 if name != "naive (flat)" else np.nan,
            "next-day direction %": s["next_day_hit"].mean() * 100 if name != "naive (flat)" else np.nan,
        }
        for name, s in scores.items()
    }).T
    print("\n" + summary.round(2).to_string())

    out_dir = os.path.join(resolve(config.base_save_path), "eval")
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(scores, names=["model"]).to_csv(os.path.join(out_dir, "per_origin.csv"))
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    plot(df, origins, results, config, args.stride, os.path.join(out_dir, "forecasts.png"))
    print(f"\nSaved summary.csv, per_origin.csv and forecasts.png to {out_dir}")


if __name__ == "__main__":
    main()
