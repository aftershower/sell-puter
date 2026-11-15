"""Scan universe: S&P 500 constituents from Wikipedia plus extra tickers from config."""

import io
import json
import logging
from pathlib import Path

import pandas as pd
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE = Path("cache/universe.json")

log = logging.getLogger(__name__)


def _fetch_sp500() -> list[str]:
    resp = requests.get(SP500_URL, headers={"User-Agent": "sell-puter/0.1 (personal screener)"}, timeout=30)
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
