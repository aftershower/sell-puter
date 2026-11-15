"""Track record: score every past recommendation against what the stock actually did.

A "recommendation" is the contract the page showed by default that day: the highest-annualized
contract per ticker that matched that day's [page] defaults. Repeats of the same contract on
later days count once, on the first day it appeared.
"""

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from .options import matches_page

HIST = Path("site/data/history")
OUT = Path("site/data/review.json")

log = logging.getLogger(__name__)


def _source(c: dict) -> str:
    if c.get("reddit"):
        return "Reddit"
    if c.get("screens"):
        return "Yahoo 榜单"
    return "S&P 500 / 自选"


def load_recs() -> list[dict]:
    recs, seen = [], set()
    for f in sorted(HIST.glob("????-??-??.json")):
        rep = json.loads(f.read_text())
        pg = rep["params"].get("page")
        if not pg:
            continue
        for c in rep["candidates"]:
            ks = [k for k in c["contracts"] if matches_page(k, pg)]
            if not ks:
                continue
            k = max(ks, key=lambda k: k["annualized"])
            if k["contract"] in seen:
                continue
            seen.add(k["contract"])
            ai = c.get("ai") or {}
            recs.append({
                "date": rep["date"],
                "ticker": c["ticker"],
                "contract": k["contract"],
                "expiry": k["expiry"],
                "dte": k["dte"],
                "strike": k["strike"],
                "premium": k["premium"],
                "breakeven": k["breakeven"],
                "collateral": k["collateral"],
                "iv": k["iv"],
                "delta": k["delta"],
                "annualized": k["annualized"],
                "crosses_earnings": k["crosses_earnings"],
                "price_at_rec": c.get("price_now", c["price"]),
                "gap": c.get("gap", 0),
                "source": _source(c),
                "ai_risk": ai.get("risk_level"),
                "ai_grade": ai.get("quality_grade"),
                "ai_deep": bool(ai.get("deep")),
            })
    return recs


def _closes(tickers: list[str], start: str) -> pd.DataFrame:
    # raw (unadjusted) closes: strikes are in unadjusted prices
    df = yf.download(tickers, start=start, auto_adjust=False, progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df.index = df.index.date
    return df


def score(recs: list[dict], today: date) -> list[dict]:
    if not recs:
        return []
    closes = _closes(sorted({r["ticker"] for r in recs}), min(r["date"] for r in recs))
    for r in recs:
        start, expiry = date.fromisoformat(r["date"]), date.fromisoformat(r["expiry"])
        path = closes[r["ticker"]].dropna() if r["ticker"] in closes else pd.Series(dtype=float)
        path = path[(path.index >= start) & (path.index <= min(expiry, today))]
        if path.empty:
            r["status"] = "no_data"
            continue
        last, K, prem = float(path.iloc[-1]), r["strike"], r["premium"]
        r["last_close"] = round(last, 2)
        r["min_close"] = round(float(path.min()), 2)
        r["touched_strike"] = bool(path.min() < K)
        r["worst_move"] = round(float(path.min()) / r["price_at_rec"] - 1, 4)
        if path.index[-1] >= expiry:
            r["status"] = "expired_worthless" if last >= K else "assigned"
            r["pnl"] = round((prem - max(0.0, K - last)) * 100, 2)
            r["return"] = round(r["pnl"] / r["collateral"], 4)
        else:
            r["status"] = "below_breakeven" if last <= r["breakeven"] else "below_strike" if last < K else "otm"
            # intrinsic-only mark: ignores remaining time value, so it flatters open losers a bit
            r["pnl_est"] = round((prem - max(0.0, K - last)) * 100, 2)
    return recs


def _bucket_stats(rows: list[dict]) -> dict:
    closed = [r for r in rows if "pnl" in r]
    return {
        "n": len(rows),
        "closed": len(closed),
        "win_rate": round(sum(r["pnl"] > 0 for r in closed) / len(closed), 3) if closed else None,
        "assigned_rate": round(sum(r["status"] == "assigned" for r in closed) / len(closed), 3) if closed else None,
        "avg_return": round(sum(r["return"] for r in closed) / len(closed), 4) if closed else None,
        "touched_rate": round(sum(r.get("touched_strike", False) for r in rows) / len(rows), 3) if rows else None,
        "open_below_strike": sum(r["status"] in ("below_strike", "below_breakeven") for r in rows),
    }


BUCKETS = {
    "AI 风险": lambda r: r["ai_risk"] or "无笔记",
    "AI 质地": lambda r: r["ai_grade"] or "无笔记",
    "来源": lambda r: r["source"],
    "IV": lambda r: "<60%" if r["iv"] < 0.6 else "60–80%" if r["iv"] < 0.8 else "80–100%" if r["iv"] < 1 else "≥100%",
    "DTE": lambda r: "≤14 天" if r["dte"] <= 14 else "15–28 天" if r["dte"] <= 28 else "29–42 天",
    "|Delta|": lambda r: "≤0.25" if -r["delta"] <= 0.25 else ">0.25",
    "跨财报": lambda r: "是" if r["crosses_earnings"] else "否",
    "盘前跳空": lambda r: "跌 ≥3%" if r["gap"] <= -0.03 else "涨 ≥3%" if r["gap"] >= 0.03 else "±3% 内",
}


def build(today: date) -> dict:
    recs = [r for r in score(load_recs(), today) if r.get("status") != "no_data"]
    review = {
        "as_of": today.isoformat(),
        "overall": _bucket_stats(recs),
        "buckets": {
            name: [{"key": k, **_bucket_stats([r for r in recs if fn(r) == k])} for k in sorted({fn(r) for r in recs})]
            for name, fn in BUCKETS.items()
        },
        "recs": sorted(recs, key=lambda r: (r["date"], r["ticker"]), reverse=True)[:200],
    }
    OUT.write_text(json.dumps(review, ensure_ascii=False, indent=1))
    log.info("review: %s", review["overall"])
    return review
