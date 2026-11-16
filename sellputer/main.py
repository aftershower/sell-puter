"""Daily pipeline: universe -> quality gate -> put chain scan -> AI risk notes -> site/data/*.json"""

import json
import logging
import os
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import ai, options, quality, tradingdays, universe

DATA = Path("site/data")
log = logging.getLogger("sellputer")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = tomllib.loads(Path("config.toml").read_text())
    q, o, pg, a, workers = cfg["quality"], cfg["options"], cfg["page"], cfg["ai"], cfg["run"]["workers"]
    now = datetime.now(ZoneInfo("America/New_York"))
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule" and not tradingdays.is_trading_day(now.date()):
        log.info("%s is not a trading day; skipping", now.date())
        return
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule" and (DATA / "history" / f"{now.date()}.json").exists():
        # a late 13:00 UTC run and the 14:00 UTC run can both land at 9:xx in winter
        log.info("already scanned today; skipping")
        return

    u = cfg["universe"]
    trending = universe.reddit_trending(u["reddit_top_n"], u["reddit_ignore"]) if u["reddit_top_n"] else []
    reddit = {r["ticker"]: r for r in trending}
    screens = universe.yahoo_screens(u["yahoo_screens"], u["yahoo_count"])
    core = universe.load_universe(u["extra_tickers"], u["exclude_tickers"])  # S&P 500 + the user's list
    tickers = sorted((set(core) | set(reddit) | set(screens)) - set(u["exclude_tickers"]))
    core = set(core)
    if limit := os.environ.get("SELLPUTER_LIMIT"):  # quick local test runs
        tickers = tickers[: int(limit)]
    log.info("universe: %d tickers", len(tickers))

    prices = quality.price_stats(tickers)
    if len(prices) < 0.7 * len(tickers):
        raise SystemExit(f"only {len(prices)}/{len(tickers)} price histories came back; not writing a report")
    fundamentals = quality.fetch_fundamentals(list(prices), q["fundamentals_cache_days"], workers)
    if len(fundamentals) < 0.7 * len(prices):
        raise SystemExit(f"only {len(fundamentals)}/{len(prices)} fundamentals came back; not writing a report")
    rejected = Counter()
    passed = []
    status = {}  # per-ticker outcome, reported for the Reddit list
    for t, f in fundamentals.items():
        reason = quality.quality_check(f, prices[t], q, core=t in core)
        status[t] = reason or "passed"
        if reason:
            rejected[reason] += 1
        else:
            passed.append(t)
    log.info("quality: %d passed, rejected %s", len(passed), dict(rejected))

    # Yahoo's option quotes run ~15 minutes late: just after the open it still serves the pre-open
    # snapshot, with no bids on most contracts. Wait for live quotes rather than publish a bare page.
    for _ in range(45):
        if options.quotes_ready():
            break
        log.info("option quotes not usable yet; waiting a minute")
        time.sleep(60)
    else:
        raise SystemExit("option quotes never became usable; keeping the previous page")

    # Options only trade 9:30-16:00 ET. Outside that, chain quotes are from the last close while
    # the stock may already be moving pre-market, so reprice off the latest pre/post-market trade.
    market_open = options.market_open_now()
    latest = {} if market_open else quality.latest_prices(passed, now.date())
    for t in passed:
        p = prices[t]
        p["price_now"] = latest.get(t, p["price"])
        p["gap"] = p["price_now"] / p["price"] - 1
    gaps = sorted(((prices[t]["gap"], t) for t in passed if abs(prices[t]["gap"]) >= o["gap_warn"]))
    log.info("market open: %s; big pre/post-market moves: %s", market_open, [(t, f"{g:+.1%}") for g, t in gaps])

    r = options.risk_free_rate()
    scan = lambda t: options.scan_ticker(
        t, prices[t]["price"], date.fromisoformat(prices[t]["price_date"]), prices[t]["price_now"],
        r, now.date(), fundamentals[t]["next_earnings"], o,
    )
    with ThreadPoolExecutor(workers) as pool:
        chains = dict(zip(passed, pool.map(scan, passed)))
    failed = [t for t, c in chains.items() if c is None]
    if failed and len(failed) <= 0.3 * len(passed):
        # a capped slow pass; each retry can take ~20s and the job has a 60-minute limit
        log.info("retrying %d failed chains one at a time", min(len(failed), 40))
        time.sleep(30)
        for t in failed[:40]:
            chains[t] = scan(t)
    failed = [t for t, c in chains.items() if c is None]
    if len(failed) > 0.3 * len(passed):
        # Yahoo is throttling us; keep yesterday's page rather than publish a gutted list
        raise SystemExit(f"{len(failed)}/{len(passed)} option chains failed; not writing a report")
    candidates = [
        {
            "ticker": t, **fundamentals[t], **prices[t],
            "core": t in core, "reddit": reddit.get(t), "screens": screens.get(t, []), "contracts": c,
        }
        for t, c in chains.items()
        if c
    ]
    for t, c in chains.items():
        status[t] = "candidate" if c else "no_contract" if c is not None else "chain_failed"
    candidates.sort(key=lambda c: c["contracts"][0]["annualized"], reverse=True)
    log.info("candidates with matching puts: %d", len(candidates))

    ai_status = {"enabled": False, "model": a["model"], "top_n": a["top_n"], "error": None, "spend": None}
    if a["enabled"] and os.environ.get("ANTHROPIC_API_KEY") and candidates:
        ai_status["enabled"] = True
        try:
            # page defaults first, so what the user sees on open has notes
            ranked = sorted(candidates, key=lambda c: not any(options.matches_page(k, pg) for k in c["contracts"]))
            notes, ai_status["spend"] = ai.notes(ranked[: a["top_n"]], a, pg)
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
            "reddit_trending": len(trending),
            "yahoo_screened": len(screens),
            "added_outside_core": len(set(tickers) - core),
        },
        "reddit": [{**r, "status": status.get(r["ticker"], "data_failed")} for r in trending],
        "params": {"quality": q, "options": o, "page": pg},
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
