"""Daily pipeline: download VN30INDEX from CafeF, rebuild VN30INDEX.csv, forecast the next session.

Meant to run after the close (e.g. 17:00, see vn30/com.kronos.vn30-signal.plist). CafeF publishes
a provisional row for today first (zero matched value) and finalises it later, so the
download is retried until today's row is final. A signal is only written from data that
includes today's final candle; on a holiday (no row for today) or if the row never gets
finalised before --deadline, no signal is written and the bot does not trade tomorrow.

Usage:
    python vn30/daily_signal.py
    python vn30/daily_signal.py --retry-minutes 15 --deadline 21:00 --signal-out signals/vn30_next_day.json
"""
import argparse
import datetime
import logging
import os
import subprocess
import sys
import time

import pandas as pd

from convert_to_csv import DEFAULT_OUTPUT, convert, default_inputs
from download_cafef import DEFAULT_OUTPUT as API_OUTPUT, download

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FINETUNE_DIR = os.path.join(REPO, "finetune_csv")


def refresh_data(today):
    """Download + convert. Returns 'final', 'provisional' or 'missing' for today's row."""
    raw = download("VN30INDEX", today - datetime.timedelta(days=60), today, API_OUTPUT)
    raw_today = raw[pd.to_datetime(raw["Ngày"], format="%d/%m/%Y").dt.date == today]
    if raw_today.empty:
        return "missing"

    df = convert(default_inputs(), DEFAULT_OUTPUT)
    return "final" if df["timestamps"].iloc[-1].date() == today else "provisional"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config_vn30_daily.yaml", help="relative to finetune_csv/")
    parser.add_argument("--signal-out", default=os.path.join(REPO, "signals", "vn30_next_day.json"))
    parser.add_argument("--paths", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retry-minutes", type=float, default=15)
    parser.add_argument("--deadline", type=datetime.time.fromisoformat, default=datetime.time(21, 0))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = datetime.date.today()
    if today.weekday() >= 5:
        logging.info("Weekend, nothing to do")
        return 0

    while True:
        try:
            status = refresh_data(today)
        except Exception:
            logging.exception("Data refresh failed")
            status = "error"

        if status == "final":
            logging.info("Today's (%s) final row is in VN30INDEX.csv", today)
            break
        if status == "missing":
            logging.warning("CafeF has no row for %s (holiday?); no signal written", today)
            return 1
        if datetime.datetime.now().time() >= args.deadline:
            logging.error("Today's row still %s at the %s deadline; no signal written", status, args.deadline)
            return 1
        logging.info("Today's row is %s; retrying in %s min", status, args.retry_minutes)
        time.sleep(args.retry_minutes * 60)

    command = [
        sys.executable, "predict_next.py",
        "--config", args.config,
        "--horizon", "1",
        "--paths", str(args.paths),
        "--signal-out", os.path.abspath(args.signal_out),
    ]
    if args.device:
        command += ["--device", args.device]
    logging.info("Running %s", " ".join(command))
    result = subprocess.run(command, cwd=FINETUNE_DIR)
    if result.returncode != 0:
        logging.error("predict_next.py failed with exit code %s; no signal written", result.returncode)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
