"""Fundamentals (cached) and price history stats, plus the "good quality" gate."""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = Path("cache/fundamentals.json")
INFO_FIELDS = {
    "name": "shortName",
    "sector": "sector",
    "industry": "industry",
    "market_cap": "marketCap",
    "roe": "returnOnEquity",
    "profit_margin": "profitMargins",
    "fcf": "freeCashflow",
    "debt_to_equity": "debtToEquity",
    "forward_pe": "forwardPE",
}

log = logging.getLogger(__name__)


def _next_earnings(info: dict) -> str | None:
    now = time.time()
    for key in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
        ts = info.get(key)
        if ts and ts > now:
            return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
    return None


def _fetch_one(ticker: str) -> dict | None:
    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info
            if not info or info.get("quoteType") not in (None, "EQUITY"):
                return None
            rec = {k: info.get(src) for k, src in INFO_FIELDS.items()}
            rec["next_earnings"] = _next_earnings(info)
            rec["fetched"] = datetime.now(timezone.utc).isoformat()
            return rec
        except Exception as e:
            log.debug("info %s attempt %d failed: %s", ticker, attempt, e)
            time.sleep(2 * (attempt + 1))
    log.warning("info %s failed after retries", ticker)
    return None


def _stale(rec: dict, max_age_days: int, today: str) -> bool:
    age = datetime.now(timezone.utc) - datetime.fromisoformat(rec["fetched"])
    # earnings date already passed -> refetch to get the next one
    passed = rec.get("next_earnings") is not None and rec["next_earnings"] < today
    return age > timedelta(days=max_age_days) or passed


def fetch_fundamentals(tickers: list[str], max_age_days: int, workers: int) -> dict[str, dict]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    today = datetime.now(timezone.utc).date().isoformat()
    todo = [t for t in tickers if t not in cache or _stale(cache[t], max_age_days, today)]
    log.info("fundamentals: %d cached, %d to fetch", len(tickers) - len(todo), len(todo))
    with ThreadPoolExecutor(workers) as pool:
        for ticker, rec in zip(todo, pool.map(_fetch_one, todo)):
            if rec:
                cache[ticker] = rec
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=0))
    return {t: cache[t] for t in tickers if t in cache}


def price_stats(tickers: list[str]) -> dict[str, dict]:
    closes = yf.download(tickers, period="1y", auto_adjust=True, progress=False, threads=True)["Close"]
    stats = {}
    for t in tickers:
        if t not in closes:
            continue
        s = closes[t].dropna()
        if len(s) < 60:
            continue
        rets = np.log(s).diff().dropna()
        price = float(s.iloc[-1])
        stats[t] = {
            "price": price,
            "price_date": s.index[-1].date().isoformat(),
            "hv20": float(rets.tail(20).std() * math.sqrt(252)),
            "hv252": float(rets.std() * math.sqrt(252)),
            "drawdown_52w": float(1 - price / s.max()),
            "above_sma200": bool(len(s) >= 200 and price > s.tail(200).mean()),
            "sma50": float(s.tail(50).mean()),
            "sma200": float(s.tail(200).mean()) if len(s) >= 200 else None,
            "low_52w": float(s.min()),
            "high_52w": float(s.max()),
        }
    return stats


def latest_prices(tickers: list[str]) -> dict[str, float]:
    """Most recent trade including pre/post-market, from today's 5-minute bars."""
    bars = yf.download(tickers, period="1d", interval="5m", prepost=True, auto_adjust=True, progress=False, threads=True)
    closes = bars["Close"] if not bars.empty else pd.DataFrame()
    out = {}
    for t in tickers:
        if t in closes:
            s = closes[t].dropna()
            if len(s):
                out[t] = float(s.iloc[-1])
    return out


def quality_check(f: dict, p: dict, q: dict, core: bool) -> str | None:
    """Return the reason a ticker fails the quality gate, or None if it passes.

    `core` tickers (S&P 500 + the user's own list) skip the fundamentals checks; every
    threshold is optional so config can switch it off by leaving it out.
    """
    def below(value, floor):
        return value is not None and not pd.isna(value) and value < floor

    if not core:
        if below(f.get("market_cap"), q.get("min_market_cap", 0)):
            return "market_cap"
        if "min_roe" in q and below(f.get("roe"), q["min_roe"]):
            return "roe"
        margin = f.get("profit_margin")
        if q.get("require_profitable") and margin is not None and not pd.isna(margin) and margin <= 0:
            return "unprofitable"
        if "min_profit_margin" in q and below(margin, q["min_profit_margin"]):
            return "profit_margin"
        if q.get("require_positive_fcf") and below(f.get("fcf"), 0):
            return "fcf"
        dte = f.get("debt_to_equity")
        if "max_debt_to_equity" in q and dte is not None and not pd.isna(dte) and dte > q["max_debt_to_equity"]:
            return "debt"
        if "max_drawdown_52w" in q and p["drawdown_52w"] > q["max_drawdown_52w"]:
            return "drawdown"
    if p["price"] < q.get("min_price", 0):
        return "price"
    if q.get("require_above_sma200") and not p["above_sma200"]:
        return "sma200"
    return None
