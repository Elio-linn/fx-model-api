#!/usr/bin/env python3
"""
Recompute the actual_*_open / actual_*_close columns directly in MongoDB,
reading closed candles from the ohlcv collection — no need to reset to None
and wait for the scheduler's backfill cycle.

For every prediction of the pair it recomputes, in place, from the target candle:

    actual_upper_open  = (next_high - next_open)  * pip_div
    actual_lower_open  = (next_open - next_low)   * pip_div
    actual_upper_close = (next_high - base_close)  * pip_div
    actual_lower_close = (base_close - next_low)   * pip_div
    live_close         = next_close

`base_close` = the close the prediction was made from (record.base_close, else
the ohlcv candle at current_candle_time). A prediction is only recomputed when
its target candle is fully CLOSED — i.e. a strictly-later candle exists in ohlcv.
The live/forming candle is upserted too, so mere existence is not enough; this is
exactly the guard the scheduler backfill uses. Predictions whose target candle
has not closed yet (e.g. the last one before the weekend gap) are left untouched.

Usage:
    python3 recompute_actuals_db.py [--pair EURUSD] [--dry-run]
"""
import argparse

from api import db
from api.config import PAIR, pip_divisor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default=PAIR, help=f"currency pair (default: {PAIR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args(argv)

    pair = args.pair
    pip_div = pip_divisor(pair)

    total = db.predictions.count_documents({"pair": pair})
    recomputed = skipped_open = skipped_base = 0

    for p in db.predictions.find({"pair": pair}):
        candle = db.ohlcv.find_one({"pair": pair, "time": p["target_time"]})
        if not candle:
            skipped_open += 1
            continue
        # Target candle must be CLOSED: a strictly-later candle must exist,
        # because save_ohlcv upserts the still-forming candle too.
        newer = db.ohlcv.find_one({"pair": pair, "time": {"$gt": p["target_time"]}})
        if not newer:
            skipped_open += 1
            continue
        # Close baseline = close of the candle the prediction was made from.
        base_close = p.get("base_close")
        if base_close is None:
            base_candle = db.ohlcv.find_one(
                {"pair": pair, "time": p.get("current_candle_time")})
            if not base_candle:
                skipped_base += 1
                continue
            base_close = base_candle["close"]

        no, nh, nl, nc = candle["open"], candle["high"], candle["low"], candle["close"]
        new_vals = {
            "actual_upper_open": (nh - no) * pip_div,
            "actual_lower_open": (no - nl) * pip_div,
            "actual_upper_close": (nh - base_close) * pip_div,
            "actual_lower_close": (base_close - nl) * pip_div,
            "live_close": float(nc),
        }
        if not args.dry_run:
            db.predictions.update_one({"_id": p["_id"]}, {"$set": new_vals})
        recomputed += 1

    mode = "DRY-RUN — would recompute" if args.dry_run else "recomputed"
    print(f"{pair}: {mode}={recomputed}, skipped(target not closed)={skipped_open}, "
          f"skipped(no base candle)={skipped_base}, total={total}, pip_div={pip_div:g}")


if __name__ == "__main__":
    main()