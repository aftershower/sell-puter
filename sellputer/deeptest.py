"""Re-run one ticker's deep dive on the latest scan's data and print it.

    python -m sellputer.deeptest TQQQ          # print only
    python -m sellputer.deeptest TQQQ --save   # also replace that ticker's note on the latest page

For checking prompt changes without paying for a whole scan (one deep dive is ~$0.1-0.2).
"""

import json
import logging
import sys
import tomllib
from pathlib import Path

import anthropic

from . import ai
from .main import DATA, write_report

BUDGET_USD = 0.4

log = logging.getLogger("sellputer.deeptest")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ticker, save = sys.argv[1].upper(), "--save" in sys.argv[2:]
    a = tomllib.loads(Path("config.toml").read_text())["ai"]
    report = json.loads((DATA / "latest.json").read_text())
    cand = next((c for c in report["candidates"] if c["ticker"] == ticker), None)
    if cand is None:
        raise SystemExit(f"{ticker} is not in the latest scan ({report['date']})")

    brief = ai._brief(cand, a["news_per_ticker"], report["params"]["page"])
    if calls := ai._previous_calls().get(ticker):
        brief["previous_calls"] = calls
    budget = ai.Budget(BUDGET_USD)
    note = ai._deep(anthropic.Anthropic(), ticker, brief, a, budget)
    log.info("spent $%.2f, %d searches", budget.spent, budget.searches)
    if note is None:
        raise SystemExit("deep dive failed")
    print(json.dumps(note, ensure_ascii=False, indent=1))

    if save:
        cand["ai"] = note
        if spend := report["ai"].get("spend"):
            spend["usd"] = round(spend["usd"] + budget.spent, 2)
        write_report(report)


if __name__ == "__main__":
    main()
