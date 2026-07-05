#!/usr/bin/env python3
"""
Recompute the actual_*_open / actual_*_close columns in a predictions-with-candles
CSV using the close-baseline and open-baseline formulas:

    actual_upper_close = (next_candle_high - current_close)    * pip_div
    actual_lower_close = (current_close    - next_candle_low)  * pip_div
    actual_upper_open  = (next_candle_high - next_candle_open) * pip_div
    actual_lower_open  = (next_candle_open - next_candle_low)  * pip_div

The CSV must contain the columns: current_close, next_candle_open, next_candle_high,
next_candle_low, actual_upper_open, actual_lower_open, actual_upper_close,
actual_lower_close. Rows missing any candle value (e.g. the last candle, whose next
candle has not closed yet) are left untouched.

Usage:
    python3 recompute_actuals.py FILE.csv [--pip-div 10000] [--no-backup]

pip_div defaults to 100 for JPY-quoted pairs (price >= 50) and 10000 otherwise,
auto-detected from the first usable current_close; override with --pip-div.
"""
import argparse
import csv
import shutil
import sys

REQUIRED = ["current_close", "next_candle_open", "next_candle_high",
            "next_candle_low", "actual_upper_open", "actual_lower_open",
            "actual_upper_close", "actual_lower_close"]


def is_blank(row, *keys):
    return any(row.get(k) in ("", None) for k in keys)


def detect_pip_div(rows):
    for r in rows:
        if not is_blank(r, "current_close"):
            # JPY pairs trade near 100+ (e.g. 150.25); everything else near 1.
            return 100.0 if float(r["current_close"]) >= 50.0 else 10000.0
    return 10000.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to the predictions-with-candles CSV")
    ap.add_argument("--pip-div", type=float, default=None,
                    help="pip divisor (default: auto — 100 for JPY, else 10000)")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not write a .bak copy before overwriting")
    args = ap.parse_args(argv)

    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames

    if not rows:
        sys.exit(f"{args.csv}: empty file")
    missing = [c for c in REQUIRED if c not in fields]
    if missing:
        sys.exit(f"{args.csv}: missing required column(s): {', '.join(missing)}")

    pip_div = args.pip_div if args.pip_div is not None else detect_pip_div(rows)

    changed = skipped = 0
    for r in rows:
        if is_blank(r, "current_close", "next_candle_open",
                    "next_candle_high", "next_candle_low"):
            skipped += 1
            continue
        cc = float(r["current_close"])
        no = float(r["next_candle_open"])
        nh = float(r["next_candle_high"])
        nl = float(r["next_candle_low"])
        r["actual_upper_close"] = repr((nh - cc) * pip_div)
        r["actual_lower_close"] = repr((cc - nl) * pip_div)
        r["actual_upper_open"] = repr((nh - no) * pip_div)
        r["actual_lower_open"] = repr((no - nl) * pip_div)
        changed += 1

    if not args.no_backup:
        shutil.copy(args.csv, args.csv + ".bak")

    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    bak = "" if args.no_backup else f", backup={args.csv}.bak"
    print(f"{args.csv}: recomputed={changed}, skipped(empty)={skipped}, "
          f"pip_div={pip_div:g}{bak}")


if __name__ == "__main__":
    main()
