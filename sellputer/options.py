"""Put option chain scan: Black-Scholes IV/delta from the mid price, then the sell-put filters."""

import logging
import math
import time
from datetime import date

import yfinance as yf

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


def _pick_expiries(expiries: list[str], today: date, o: dict) -> list[tuple[str, int]]:
    inside = [(e, (date.fromisoformat(e) - today).days) for e in expiries]
    inside = [(e, d) for e, d in inside if o["min_dte"] <= d <= o["max_dte"]]
    n = o["max_expiries_per_ticker"]
    if len(inside) <= n:
        return inside
    # spread the picks evenly across the DTE window
    step = (len(inside) - 1) / (n - 1)
    return [inside[round(i * step)] for i in range(n)]


def scan_ticker(ticker: str, s_ref: float, s_now: float, r: float, today: date, next_earnings: str | None, o: dict) -> list[dict] | None:
    """Matching contracts, or None if Yahoo kept failing (usually rate limiting).

    Option quotes are priced off s_ref (the last close when the market is shut, since options
    don't trade pre-market). IV is solved at s_ref, then delta and premium are re-estimated at
    s_now so a pre-market gap shows up before the open.
    """
    for attempt in range(3):
        try:
            return _scan(ticker, s_ref, s_now, r, today, next_earnings, o)
        except Exception as e:
            log.debug("chain %s attempt %d failed: %s", ticker, attempt, e)
            time.sleep(5 * 2**attempt)
    log.warning("chain %s failed after retries", ticker)
    return None


def _scan(ticker: str, s_ref: float, s_now: float, r: float, today: date, next_earnings: str | None, o: dict) -> list[dict]:
    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    if not expiries:
        raise RuntimeError("no expiries (rate limited?)")
    found = []
    for expiry, dte in _pick_expiries(expiries, today, o):
        T = dte / 365
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
            K = float(row.strike)
            iv = implied_vol(mid, s_ref, K, T, r)
            if iv is None or iv < o["min_iv"]:
                continue
            est, delta = bs_put(s_now, K, T, r, iv)
            premium = mid if s_now == s_ref else est
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
                "spread_pct": round(spread, 4),
                "open_interest": oi,
                "volume": int(_num(row.volume)),
                "iv": round(iv, 4),
                "delta": round(delta, 4),
                "otm_pct": round(1 - K / s_now, 4),
                "breakeven": round(K - premium, 2),
                "annualized": round(annualized, 4),
                "crosses_earnings": bool(next_earnings and next_earnings <= expiry),
            })
    found.sort(key=lambda c: c["annualized"], reverse=True)
    # best contract of each expiry first, then fill with the next-best overall
    firsts = list({c["expiry"]: c for c in reversed(found)}.values())
    firsts.sort(key=lambda c: c["annualized"], reverse=True)
    picked = firsts + [c for c in found if c not in firsts]
    return picked[: o["contracts_per_ticker"]]
