"""Forecast the next trading day(s) with the finetuned model and keep a forward-test log.

Samples many independent paths from the latest `lookback_window` rows and reports the
share of paths that close above today's close (P(up)). Every forecast is appended to a
log; on later runs, once the target day is in the CSV, the actual close is filled in and
the running hit rate is printed. Refresh the data first (vn30/convert_to_csv.py).

Usage (from inside finetune_csv/):
    python predict_next.py --config configs/config_vn30_daily.yaml
    python predict_next.py --config configs/config_vn30_daily.yaml --horizon 5 --paths 100
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from evaluate import FEATURES, resolve  # also puts the repo root on sys.path
from model import Kronos, KronosTokenizer, KronosPredictor
from config_loader import CustomFinetuneConfig


def sample_paths(predictor, df, horizon, args):
    context = df.iloc[-args.lookback:].reset_index(drop=True)
    # Weekdays only: Vietnamese public holidays are not skipped, which only shifts the date features slightly
    future = pd.Series(pd.bdate_range(context["timestamps"].iloc[-1], periods=horizon + 1)[1:])
    torch.manual_seed(args.seed)
    closes = []
    for i in range(0, args.paths, args.batch_size):
        n = min(args.batch_size, args.paths - i)
        preds = predictor.predict_batch(
            df_list=[context[FEATURES]] * n,
            x_timestamp_list=[context["timestamps"]] * n,
            y_timestamp_list=[future] * n,
            pred_len=horizon,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=1,
            verbose=False,
        )
        closes += [p["close"].to_numpy() for p in preds]
    return future, np.array(closes)


def update_log(log_path, df, entry):
    log = pd.read_csv(log_path, parse_dates=["last_date"]) if os.path.exists(log_path) else pd.DataFrame()
    if not log.empty:
        # Re-running on the same data replaces that forecast instead of duplicating it
        log = log[~((log["last_date"] == entry["last_date"]) & (log["horizon"] == entry["horizon"]))]
    log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True)

    # Fill outcomes by trading-day position, so holidays in between don't matter
    position = {ts: i for i, ts in enumerate(df["timestamps"])}
    for i, row in log.iterrows():
        target = position.get(row["last_date"], len(df)) + int(row["horizon"])
        if target < len(df):
            actual = df["close"].iloc[target]
            log.loc[i, "actual_date"] = df["timestamps"].iloc[target].date()
            log.loc[i, "actual_close"] = actual
            log.loc[i, "hit"] = (row["pred_median_close"] > row["last_close"]) == (actual > row["last_close"])

    log.to_csv(log_path, index=False)
    return log


def write_signal(path, signal):
    """Write via a temp file + rename, so a reader never sees a half-written signal."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(signal, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--horizon", type=int, default=1, help="trading days ahead")
    parser.add_argument("--paths", type=int, default=50, help="independent sampled paths")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda:0 / mps / cpu (auto-detected if omitted)")
    parser.add_argument("--log", default=None, help="forward-test log CSV (default: <base_save_path>/forecast_log.csv)")
    parser.add_argument("--signal-out", default=None, help="also write the forecast as JSON for the trading bot")
    args = parser.parse_args()

    config = CustomFinetuneConfig(args.config)
    args.lookback = config.lookback_window
    df = pd.read_csv(resolve(config.data_path), parse_dates=["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    predictor = KronosPredictor(
        Kronos.from_pretrained(resolve(config.basemodel_best_model_path)),
        KronosTokenizer.from_pretrained(resolve(config.tokenizer_best_model_path)),
        device=args.device, max_context=config.max_context, clip=config.clip,
    )
    future, closes = sample_paths(predictor, df, args.horizon, args)

    last_date, last_close = df["timestamps"].iloc[-1], df["close"].iloc[-1]
    end = closes[:, -1]
    change = (end / last_close - 1) * 100
    p_up = (end > last_close).mean()

    print(f"Data up to {last_date.date()}, close {last_close:.2f}")
    print(f"Forecast for {args.horizon} trading day(s) ahead (~{future.iloc[-1].date()}), {args.paths} paths:")
    print(f"  median close  {np.median(end):.2f}  ({np.median(change):+.2f}%)")
    print(f"  10-90% range  {np.percentile(end, 10):.2f} .. {np.percentile(end, 90):.2f}")
    direction = "UP" if np.median(end) > last_close else "DOWN"
    print(f"  P(up)         {p_up:.0%}  -> {direction}")

    if args.signal_out:
        write_signal(args.signal_out, {
            "for_date": str(future.iloc[0].date()),
            "horizon": args.horizon,
            "direction": direction,
            "p_up": round(float(p_up), 3),
            "pred_median_close": round(float(np.median(end)), 2),
            "data_last_date": str(last_date.date()),
            "last_close": float(last_close),
            "paths": args.paths,
            "model": resolve(config.basemodel_best_model_path),
            "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        })
        print(f"Signal written to {args.signal_out}")

    log_path = args.log or os.path.join(resolve(config.base_save_path), "forecast_log.csv")
    log = update_log(log_path, df, {
        "made_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "last_date": last_date,
        "last_close": last_close,
        "horizon": args.horizon,
        "pred_median_close": round(float(np.median(end)), 2),
        "p_up": round(float(p_up), 3),
    })
    scored = log.dropna(subset=["hit"]) if "hit" in log else log.iloc[0:0]
    print(f"\nLogged to {log_path}")
    if len(scored):
        # hit is an object column after the CSV round-trip; summing numpy bools there ORs instead of counting
        hits = scored["hit"].astype(bool)
        print(f"Forward test so far: {hits.sum()}/{len(hits)} directions correct ({hits.mean():.0%})")


if __name__ == "__main__":
    main()
