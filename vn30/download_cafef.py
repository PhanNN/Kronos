"""Download VN30INDEX daily history from CafeF's price-history API.

Rows are merged into vn30/CafeF_API_VN30INDEX.csv (newest download wins per date), using the
same Vietnamese column names as the CafeF .xlsx exports, so convert_to_csv.py reads both.

Usage:
    python vn30/download_cafef.py                    # last 60 days
    python vn30/download_cafef.py --start 2020-01-01 # backfill
"""
import argparse
import datetime
import os
import time

import pandas as pd
import requests

API_URL = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
PAGE_SIZE = 30  # the API caps every page at 30 rows, whatever PageSize asks for

# API field -> column name used in the CafeF .xlsx exports
COLUMN_MAP = {
    "Ngay": "Ngày",
    "GiaDongCua": "Giá đóng cửa",
    "GiaDieuChinh": "Giá điều chỉnh",
    "ThayDoi": "Thay đổi",
    "KhoiLuongKhopLenh": "Khối lượng khớp lệnh",
    "GiaTriKhopLenh": "Giá trị khớp lệnh",
    "KLThoaThuan": "Khối lượng thỏa thuận",
    "GtThoaThuan": "Giá trị thỏa thuận",
    "GiaMoCua": "Giá mở cửa",
    "GiaCaoNhat": "Giá cao nhất",
    "GiaThapNhat": "Giá thấp nhất",
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "CafeF_API_VN30INDEX.csv")


def fetch(symbol, start, end):
    rows, page = [], 1
    while True:
        response = requests.get(
            API_URL,
            params={
                "Symbol": symbol,
                "StartDate": start.strftime("%m/%d/%Y"),
                "EndDate": end.strftime("%m/%d/%Y"),
                "PageIndex": page,
                "PageSize": PAGE_SIZE,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("Success"):
            raise RuntimeError(f"CafeF API error: {payload.get('Message')}")

        data = payload["Data"]
        rows += data["Data"]
        if not data["Data"] or len(rows) >= data["TotalCount"]:
            return rows
        page += 1
        time.sleep(0.5)


def download(symbol, start, end, output_path):
    fetched = pd.DataFrame(fetch(symbol, start, end))
    if fetched.empty:
        raise RuntimeError(f"CafeF returned no rows for {symbol} {start}..{end}")
    fetched = fetched.rename(columns=COLUMN_MAP)
    fetched.insert(0, "Mã", symbol)
    fetched = fetched[["Mã"] + list(COLUMN_MAP.values())]

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        fetched = pd.concat([existing[~existing["Ngày"].isin(fetched["Ngày"])], fetched], ignore_index=True)

    order = pd.to_datetime(fetched["Ngày"], format="%d/%m/%Y").sort_values(ascending=False).index
    fetched.loc[order].to_csv(output_path, index=False)
    return fetched.loc[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="VN30INDEX")
    parser.add_argument("--start", type=datetime.date.fromisoformat,
                        default=datetime.date.today() - datetime.timedelta(days=60))
    parser.add_argument("--end", type=datetime.date.fromisoformat, default=datetime.date.today())
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = download(args.symbol, args.start, args.end, args.output)
    latest = df.iloc[0]
    print(f"Saved {len(df)} rows to {args.output}; latest {latest['Ngày']} close {latest['Giá đóng cửa']} "
          f"amount {latest['Giá trị khớp lệnh']}")


if __name__ == "__main__":
    main()
