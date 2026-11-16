"""Convert CafeF VN30 history (.xlsx exports and download_cafef.py's CSV) into a Kronos-style OHLCV CSV.

Output columns match tests/data/regression_input.csv:
    timestamps, open, high, low, close, volume, amount

Usage:
    python vn30/convert_to_csv.py                         # vn30/*.xlsx + vn30/CafeF_*.csv -> vn30/VN30INDEX.csv
    python vn30/convert_to_csv.py -i a.xlsx b.xlsx -o out.csv
"""
import argparse
import glob
import os

import pandas as pd

COLUMN_MAP = {
    "Ngày": "timestamps",
    "Giá mở cửa": "open",
    "Giá cao nhất": "high",
    "Giá thấp nhất": "low",
    "Giá đóng cửa": "close",
    "Khối lượng khớp lệnh": "volume",
    "Giá trị khớp lệnh": "amount",
}
OUTPUT_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def load_file(path):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["timestamps"] = pd.to_datetime(df["timestamps"], format="%d/%m/%Y")
    for col in OUTPUT_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def report_dropped(df, rejected, reason):
    """Print rejected rows whose date has no accepted copy in another source file."""
    lost = rejected & ~df["timestamps"].isin(df.loc[~rejected, "timestamps"])
    if lost.any():
        print(f"Dropping {lost.sum()} {reason} row(s):")
        print(df[lost].drop_duplicates(subset="timestamps").to_string(index=False))


def drop_spikes(df, max_deviation=0.1):
    """Drop days whose close is far from the median close of the neighbouring days (source typos)."""
    neighbours = pd.concat([df["close"].shift(k) for k in (-2, -1, 1, 2)], axis=1).median(axis=1)
    spike = (df["close"] - neighbours).abs() / neighbours > max_deviation
    report_dropped(df, spike, "spiking")
    return df[~spike].reset_index(drop=True)


def convert(input_paths, output_path):
    df = pd.concat([load_file(p) for p in input_paths], ignore_index=True).dropna()

    # Per-row checks run before de-duplication, so a bad copy of a date in one file
    # doesn't replace a good copy of it in another.
    # No matched volume/amount: a provisional row CafeF has not finalised yet
    incomplete = (df["volume"] <= 0) | (df["amount"] <= 0)
    report_dropped(df, incomplete, "incomplete (not finalised by CafeF yet, re-download later)")
    prices = df[["open", "high", "low", "close"]]
    inconsistent = ~incomplete & ((df["high"] < prices.max(axis=1)) | (df["low"] > prices.min(axis=1)))
    report_dropped(df, inconsistent, "inconsistent OHLC")

    df = (
        df[~(incomplete | inconsistent)]
        .drop_duplicates(subset="timestamps", keep="last")
        .sort_values("timestamps")
        .reset_index(drop=True)
    )
    df = drop_spikes(df)
    df[OUTPUT_COLUMNS].to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return df


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "VN30INDEX.csv")


def default_inputs():
    # The API download goes last so its rows win over older .xlsx exports of the same date
    return sorted(glob.glob(os.path.join(HERE, "*.xlsx"))) + sorted(glob.glob(os.path.join(HERE, "CafeF_*.csv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--inputs", nargs="+", default=default_inputs())
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.inputs:
        parser.error("no input .xlsx/.csv files found")

    df = convert(args.inputs, args.output)
    print(f"Wrote {len(df)} rows ({df['timestamps'].min().date()} -> {df['timestamps'].max().date()}) to {args.output}")


if __name__ == "__main__":
    main()
