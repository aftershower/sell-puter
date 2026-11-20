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

from .options import bs_put, is_thin, lower_alt, matches_page, page_for

RATE = 0.04  # for marking open puts; a rough rate is fine here
HIST = Path("site/data/history")
OUT = Path("site/data/review.json")

log = logging.getLogger(__name__)


def _source(c: dict) -> str:
    if c.get("watch"):
        return "自选"
    if c.get("reddit"):
        return "Reddit"
    if c.get("screens"):
        return "Yahoo 榜单"
    return "S&P 500 / 额外名单"


def load_recs() -> list[dict]:
    recs, seen = [], set()
    for f in sorted(HIST.glob("????-??-??.json")):
        rep = json.loads(f.read_text())
        pg = rep["params"].get("page")
        if not pg:
            continue
        for c in rep["candidates"]:
            ks = [k for k in c["contracts"] if matches_page(k, page_for(c, pg))]
            if not ks:
                continue
            k = max(ks, key=lambda k: k["annualized"])
            if k["contract"] in seen:
                continue
            seen.add(k["contract"])
            ai = c.get("ai") or {}
            alt = lower_alt(c["contracts"], k)
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
                "buffer_sd": k.get("buffer_sd"),
                "thin": is_thin(k, ai.get("risk_level"), pg),
                # the same-expiry next lower strike, scored alongside as the safer counterfactual
                "alt": {x: alt[x] for x in ("contract", "expiry", "strike", "premium", "breakeven", "collateral", "iv",
                                            "delta", "annualized", "buffer_sd")} if alt else None,
            })
    return recs


def closes(tickers: list[str], start: str) -> pd.DataFrame:
    # raw (unadjusted) closes: strikes are in unadjusted prices
    df = yf.download(tickers, start=start, auto_adjust=False, progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df.index = df.index.date
    return df


def outcome(k: dict, path: pd.Series, today: date, price: float | None = None) -> dict:
    """What happened to a sold put, given the stock's closes from the day it was sold (path)."""
    expiry = date.fromisoformat(k["expiry"])
    path = path.dropna()
    path = path[path.index <= min(expiry, today)]
    if path.empty:
        return {"status": "no_data"}
    last, K, prem = float(path.iloc[-1]), k["strike"], k["premium"]
    o = {"sessions": len(path), "last_close": round(last, 2), "min_close": round(float(path.min()), 2),
         "touched_strike": bool(path.min() < K)}
    if price:
        o["move"] = round(last / price - 1, 4)
        o["worst_move"] = round(float(path.min()) / price - 1, 4)
    if path.index[-1] >= expiry:
        o["status"] = "expired_worthless" if last >= K else "assigned"
        o["pnl"] = round((prem - max(0.0, K - last)) * 100, 2)
        o["return"] = round(o["pnl"] / k["collateral"], 4)
    else:
        o["status"] = "below_breakeven" if last <= k["breakeven"] else "below_strike" if last < K else "otm"
        # mark the put at the latest close with the IV it was sold at (no stored quotes for later days)
        value = bs_put(last, K, max((expiry - today).days, 1) / 365, RATE, k["iv"])[0]
        o["pnl_est"] = round((prem - value) * 100, 2)
    return o


def score(recs: list[dict], today: date) -> list[dict]:
    if not recs:
        return []
    cl = closes(sorted({r["ticker"] for r in recs}), min(r["date"] for r in recs))
    for r in recs:
        path = cl[r["ticker"]] if r["ticker"] in cl else pd.Series(dtype=float)
        path = path[path.index >= date.fromisoformat(r["date"])]
        r.update(outcome(r, path, today, r["price_at_rec"]))
        if r["alt"]:
            o = outcome(r["alt"], path, today)
            r["alt"].update({x: o.get(x) for x in ("status", "touched_strike")}, pnl=o.get("pnl", o.get("pnl_est")))
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
    "缓冲 σ": lambda r: "无数据" if r.get("buffer_sd") is None else "<0.6σ" if r["buffer_sd"] < 0.6 else "≥0.6σ",
    "跨财报": lambda r: "是" if r["crosses_earnings"] else "否",
    "盘前跳空": lambda r: "跌 ≥3%" if r["gap"] <= -0.03 else "涨 ≥3%" if r["gap"] >= 0.03 else "±3% 内",
}


def _vs_alt(rows: list[dict]) -> dict:
    """The pick against its same-expiry lower strike, over picks that had one."""
    rows = [r for r in rows if r.get("alt") and r["alt"].get("pnl") is not None and r.get("status") != "no_data"]
    pnl = lambda r: r.get("pnl", r.get("pnl_est"))
    return {
        "n": len(rows),
        "closed": sum("pnl" in r for r in rows),
        "touched": sum(bool(r.get("touched_strike")) for r in rows),
        "alt_touched": sum(bool(r["alt"].get("touched_strike")) for r in rows),
        "pnl": round(sum(pnl(r) for r in rows), 2),
        "alt_pnl": round(sum(r["alt"]["pnl"] for r in rows), 2),
        "alt_better": sum(r["alt"]["pnl"] > pnl(r) for r in rows),
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
        "vs_alt": {"全部": _vs_alt(recs), "高风险且缓冲薄": _vs_alt([r for r in recs if r.get("thin")])},
        "recs": sorted(recs, key=lambda r: (r["date"], r["ticker"]), reverse=True)[:200],
    }
    OUT.write_text(json.dumps(review, ensure_ascii=False, indent=1))
    log.info("review: %s", review["overall"])
    return review
