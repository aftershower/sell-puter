"""Scan universe: S&P 500 constituents from Wikipedia plus extra tickers from config."""

import io
import json
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# ApeWisdom counts ticker mentions across stock subreddits over a rolling 24h window
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
UA = {"User-Agent": "sell-puter/0.1 (personal screener)"}
CACHE = Path("cache/universe.json")

log = logging.getLogger(__name__)


def _fetch_sp500() -> list[str]:
    resp = requests.get(SP500_URL, headers=UA, timeout=30)
    resp.raise_for_status()
    for table in pd.read_html(io.StringIO(resp.text)):
        if "Symbol" in table.columns and len(table) > 400:
            # yfinance spells class shares with a dash: BRK.B -> BRK-B
            return table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    raise ValueError("S&P 500 table not found")


def load_universe(extra: list[str], exclude: list[str]) -> list[str]:
    try:
        tickers = _fetch_sp500()
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(tickers))
    except Exception as e:
        if not CACHE.exists():
            raise
        log.warning("S&P 500 fetch failed (%s); using cached list", e)
        tickers = json.loads(CACHE.read_text())
    return sorted((set(tickers) | set(extra)) - set(exclude))


def reddit_trending(n: int, ignore: list[str]) -> list[dict]:
    """Top-n most-mentioned tickers on Reddit stock subs in the last 24h (empty list on failure).

    `ignore` drops words that collide with tickers ("DTE", "IT", "ALL", ...).
    """
    out = []
    try:
        for page in range(1, 6):
            resp = requests.get(APEWISDOM_URL.format(page=page), headers=UA, timeout=30)
            resp.raise_for_status()
            out += [r for r in resp.json()["results"] if r["ticker"] not in ignore]
            if len(out) >= n:
                break
    except Exception as e:
        log.warning("Reddit trending fetch failed: %s", e)
    return [
        {
            "ticker": r["ticker"].replace(".", "-"),
            "rank": r["rank"],
            "mentions": r["mentions"],
            "mentions_24h_ago": r.get("mentions_24h_ago"),
            "upvotes": r.get("upvotes"),
        }
        for r in out[:n]
    ]


def yahoo_screens(names: list[str], count: int) -> dict[str, list[str]]:
    """Tickers from Yahoo's predefined screeners (e.g. most_actives, day_losers) -> screener names."""
    out: dict[str, list[str]] = {}
    for name in names:
        try:
            for q in yf.screen(name, count=count)["quotes"]:
                if q.get("quoteType") == "EQUITY":
                    out.setdefault(q["symbol"], []).append(name)
        except Exception as e:
            log.warning("Yahoo screen %s failed: %s", name, e)
    return out
