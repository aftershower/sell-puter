"""Daily pipeline: universe -> quality gate -> put chain scan -> AI risk notes -> site/data/*.json"""

import json
import logging
import os
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import ai, options, quality, universe

DATA = Path("site/data")
log = logging.getLogger("sellputer")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = tomllib.loads(Path("config.toml").read_text())
    q, o, a, workers = cfg["quality"], cfg["options"], cfg["ai"], cfg["run"]["workers"]
    now = datetime.now(ZoneInfo("America/New_York"))

    tickers = universe.load_universe(cfg["universe"]["extra_tickers"], cfg["universe"]["exclude_tickers"])
    if limit := os.environ.get("SELLPUTER_LIMIT"):  # quick local test runs
        tickers = tickers[: int(limit)]
    log.info("universe: %d tickers", len(tickers))

    prices = quality.price_stats(tickers)
    fundamentals = quality.fetch_fundamentals(list(prices), q["fundamentals_cache_days"], workers)
    rejected = Counter()
    passed = []
    for t, f in fundamentals.items():
        reason = quality.quality_check(f, prices[t], q)
        if reason:
            rejected[reason] += 1
        else:
            passed.append(t)
    log.info("quality: %d passed, rejected %s", len(passed), dict(rejected))

    # Options only trade 9:30-16:00 ET. Outside that, chain quotes are from the last close while
    # the stock may already be moving pre-market, so reprice off the latest pre/post-market trade.
    market_open = now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)
    latest = {} if market_open else quality.latest_prices(passed)
    for t in passed:
        p = prices[t]
        p["price_now"] = latest.get(t, p["price"])
        p["gap"] = p["price_now"] / p["price"] - 1
    gaps = sorted(((prices[t]["gap"], t) for t in passed if abs(prices[t]["gap"]) >= o["gap_warn"]))
    log.info("market open: %s; big pre/post-market moves: %s", market_open, [(t, f"{g:+.1%}") for g, t in gaps])

    r = options.risk_free_rate()
    scan = lambda t: options.scan_ticker(
        t, prices[t]["price"], prices[t]["price_now"], r, now.date(), fundamentals[t]["next_earnings"], o
    )
    with ThreadPoolExecutor(workers) as pool:
        chains = dict(zip(passed, pool.map(scan, passed)))
    failed = [t for t, c in chains.items() if c is None]
    if failed:
        log.info("retrying %d failed chains one at a time", len(failed))
        time.sleep(30)
        for t in failed:
            chains[t] = scan(t)
    failed = [t for t, c in chains.items() if c is None]
    if len(failed) > 0.3 * len(passed):
        # Yahoo is throttling us; keep yesterday's page rather than publish a gutted list
        raise SystemExit(f"{len(failed)}/{len(passed)} option chains failed; not writing a report")
    candidates = [{"ticker": t, **fundamentals[t], **prices[t], "contracts": c} for t, c in chains.items() if c]
    candidates.sort(key=lambda c: c["contracts"][0]["annualized"], reverse=True)
    log.info("candidates with matching puts: %d", len(candidates))

    ai_status = {"enabled": False, "model": a["model"], "top_n": a["top_n"], "error": None}
    if a["enabled"] and os.environ.get("ANTHROPIC_API_KEY") and candidates:
        ai_status["enabled"] = True
        try:
            notes = ai.risk_notes(candidates[: a["top_n"]], a)
            for c in candidates:
                c["ai"] = notes.get(c["ticker"])
        except Exception as e:
            log.exception("AI step failed")
            ai_status["error"] = str(e)

    for c in candidates:
        c.pop("fetched", None)
    report = {
        "generated_at": now.isoformat(timespec="minutes"),
        "date": now.date().isoformat(),
        "risk_free_rate": round(r, 4),
        "market_open": market_open,
        "stats": {
            "universe": len(tickers),
            "quality_passed": len(passed),
            "candidates": len(candidates),
            "rejected": dict(rejected),
            "chain_failed": failed,
        },
        "params": {"quality": q, "options": o},
        "ai": ai_status,
        "candidates": candidates,
    }
    write_report(report)


def write_report(report: dict) -> None:
    hist = DATA / "history"
    hist.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, ensure_ascii=False, indent=1, default=str)
    (DATA / "latest.json").write_text(body)
    (hist / f"{report['date']}.json").write_text(body)
    dates = sorted((p.stem for p in hist.glob("????-??-??.json")), reverse=True)
    (hist / "index.json").write_text(json.dumps(dates))
    log.info("wrote %s (%d candidates)", DATA / "latest.json", len(report["candidates"]))


if __name__ == "__main__":
    main()
