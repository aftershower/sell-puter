"""Put option chain scan: Black-Scholes IV/delta from the mid price, then the sell-put filters."""

import logging
import math
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

NY = ZoneInfo("America/New_York")

log = logging.getLogger(__name__)


def _num(x) -> float:
    return 0.0 if x is None or math.isnan(x) else float(x)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Black-Scholes European put (price, delta); dividends ignored."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return price, _norm_cdf(d1) - 1


def implied_vol(price: float, S: float, K: float, T: float, r: float) -> float | None:
    lo, hi = 0.01, 5.0
    if not bs_put(S, K, T, r, lo)[0] < price < bs_put(S, K, T, r, hi)[0]:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_put(S, K, T, r, mid)[0] < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def risk_free_rate() -> float:
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(irx.iloc[-1]) / 100
    except Exception as e:
        log.warning("^IRX fetch failed (%s); using 4%%", e)
        return 0.04


def market_open_now() -> bool:
    now = datetime.now(NY)
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)


PROBE = ["AAPL", "MSFT", "NVDA", "AMZN", "AMD", "META"]


def quotes_ready(min_quoted: float = 0.95) -> bool:
    """Whether Yahoo's put quotes are usable. They run ~15 minutes late, so for a while after the
    9:30 open Yahoo still serves the pre-open snapshot, where most contracts have no bid."""
    total = quoted = 0
    for t in PROBE:
        try:
            tk = yf.Ticker(t)
            s = float(tk.fast_info["last_price"])
            puts = tk.option_chain(tk.options[min(3, len(tk.options) - 1)]).puts
        except Exception as e:
            log.debug("quote probe %s failed: %s", t, e)
            continue
        near = puts[(puts.strike > s * 0.92) & (puts.strike < s)]
        total += len(near)
        quoted += int((near.bid > 0).sum())
    log.info("quote probe: %d/%d near-the-money puts have a bid", quoted, total)
    return total > 0 and quoted / total >= min_quoted


def _pick_expiries(expiries: list[str], today: date, o: dict) -> list[tuple[str, int]]:
    inside = [(e, (date.fromisoformat(e) - today).days) for e in expiries]
    inside = [(e, d) for e, d in inside if o["min_dte"] <= d <= o["max_dte"]]
    n = o["max_expiries_per_ticker"]
    if len(inside) <= n:
        return inside
    if n == 1:
        return inside[:1]
    # spread the picks evenly across the DTE window
    step = (len(inside) - 1) / (n - 1)
    return [inside[round(i * step)] for i in range(n)]


def scan_ticker(
    ticker: str, s_ref: float, ref_date: date, s_now: float, r: float, today: date, next_earnings: str | None, o: dict
) -> list[dict] | None:
    """Matching contracts, or None if Yahoo kept failing (usually rate limiting).

    Before the open, option quotes are stale: they were struck against s_ref (the close on
    ref_date). IV is solved there, then delta and premium are re-estimated at s_now (the latest
    pre-market trade) so a gap shows up before the open. Once the market is open, quotes are live
    and both prices are the live one.
    """
    for attempt in range(3):
        try:
            return _scan(ticker, s_ref, ref_date, s_now, r, today, next_earnings, o)
        except Exception as e:
            log.debug("chain %s attempt %d failed: %s", ticker, attempt, e)
            if attempt < 2:
                time.sleep(5 * 2**attempt)
    log.warning("chain %s failed after retries", ticker)
    return None


def _scan(
    ticker: str, s_ref: float, ref_date: date, s_now: float, r: float, today: date, next_earnings: str | None, o: dict
) -> list[dict]:
    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    if not expiries:
        raise RuntimeError("no expiries (rate limited?)")
    if market_open_now():  # the job can start late; decide per ticker, at fetch time
        s_ref = s_now = float(tk.fast_info["last_price"])
        ref_date = today
    found = []
    for expiry, dte in _pick_expiries(expiries, today, o):
        T = dte / 365
        # stale quotes carry the time value as of their own close, not as of today
        T_ref = max((date.fromisoformat(expiry) - ref_date).days, 1) / 365
        puts = tk.option_chain(expiry).puts
        puts = puts[(puts.strike < s_now) & (puts.strike * 100 <= o["max_collateral"])]
        for row in puts.itertuples():
            bid, ask = _num(row.bid), _num(row.ask)
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            oi = int(_num(row.openInterest))
            spread = (ask - bid) / mid
            if mid < o["min_premium"] or oi < o["min_open_interest"] or spread > o["max_spread_pct"]:
                continue
            last_trade = pd.Timestamp(row.lastTradeDate).date() if not pd.isna(row.lastTradeDate) else None
            if last_trade is None or (ref_date - last_trade).days > o["max_quote_age_days"]:
                continue
            K = float(row.strike)
            iv = implied_vol(mid, s_ref, K, T_ref, r)
            if iv is None or iv < o["min_iv"]:
                continue
            est, delta = bs_put(s_now, K, T, r, iv)
            repriced = s_now != s_ref
            # a realistic fill for a seller is below the mid; take a quarter of the spread off
            premium = (est if repriced else mid) - 0.25 * (ask - bid)
            annualized = premium / K * 365 / dte
            if not (o["delta_min"] <= -delta <= o["delta_max"]) or annualized < o["min_annualized"]:
                continue
            found.append({
                "contract": row.contractSymbol,
                "expiry": expiry,
                "dte": dte,
                "strike": K,
                "collateral": round(K * 100),
                "bid": bid,
                "ask": ask,
                "mid": round(mid, 3),
                "premium": round(premium, 3),
                "repriced": repriced,
                "spread_pct": round(spread, 4),
                "open_interest": oi,
                "volume": int(_num(row.volume)),
                "last_trade": last_trade.isoformat(),
                "iv": round(iv, 4),
                "delta": round(delta, 4),
                "otm_pct": round(1 - K / s_now, 4),
                # strike distance in 1-sd moves to expiry: more = more cushion
                "buffer_sd": round((s_now - K) / (s_now * iv * math.sqrt(T)), 2),
                "breakeven": round(K - premium, 2),
                "annualized": round(annualized, 4),
                "crosses_earnings": bool(next_earnings and next_earnings <= expiry),
                "earnings_unknown": next_earnings is None,
            })
    # keep the best contract per (expiry, 0.05-wide delta band) so safer strikes survive the cut
    best: dict[tuple, dict] = {}
    for c in found:
        key = (c["expiry"], int(-c["delta"] / 0.05))
        if key not in best or c["annualized"] > best[key]["annualized"]:
            best[key] = c
    picked = sorted(best.values(), key=lambda c: c["annualized"], reverse=True)
    return picked[: o["contracts_per_ticker"]]


def matches_page(k: dict, pg: dict) -> bool:
    return (
        k["iv"] >= pg["min_iv"]
        and k["annualized"] >= pg["min_annualized"]
        and pg["delta_min"] <= -k["delta"] <= pg["delta_max"]
        and k["dte"] <= pg["max_dte"]
        and k["collateral"] <= pg["max_collateral"]
        and not (pg["hide_earnings"] and k["crosses_earnings"])
    )
