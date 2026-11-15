# sell-puter

每个交易日早上 9 点（美东）自动扫描 S&P 500 + 自选名单，按规则筛出适合卖现金担保看跌期权（cash-secured put）的合约，用 Claude 写质地点评和风险笔记，发布到 GitHub Pages。

> 这是按你自己定的规则筛出来的数据，加上 AI 生成的摘要，不构成投资建议。卖 put 在股价大跌时可能被行权并产生较大亏损。

## 流程

1. **股票池**：从维基百科抓 S&P 500 成分股，加上 `config.toml` 里的 `extra_tickers`
2. **质地筛选**：市值、ROE、净利率、自由现金流、负债率、距 52 周高点回撤（数据来自 yfinance，基本面缓存 7 天）
3. **期权扫描**：7–35 天到期，|Delta| 0.20–0.30，年化 ≥ 20%，一手担保 ≤ $40,000，OI 和买卖价差过滤
4. **盘前处理**：9 点时期权还没开盘，报价是昨收。程序用昨收价反推 IV，再用盘前价重新估算 Delta 和权利金；盘前涨跌超过 3% 会标红
5. **AI 笔记**（`claude-sonnet-5-5`）：对年化最高的前 20 只写质地评级（A/B/C）、到期前风险（低/中/高）和新闻摘要。会结合 yfinance 新闻，必要时联网搜索、读取网页（包括 Reddit 讨论）
6. **发布**：结果 commit 到 `site/data/`（`latest.json` + 每日历史），整个 `site/` 部署到 GitHub Pages

## 一次性设置

1. 仓库 Settings → Secrets and variables → Actions → 新建 `ANTHROPIC_API_KEY`
2. 仓库 Settings → Pages → Source 选 **GitHub Actions**
3. Actions → daily-scan → Run workflow，手动跑一次

## 改规则

所有参数都在 [config.toml](config.toml)，改完 push 即可。

## 本地运行

```bash
uv run python -m sellputer.main                   # 全量扫描，写 site/data/
SELLPUTER_LIMIT=50 uv run python -m sellputer.main   # 只扫前 50 只，快速测试
python3 -m http.server 8765 --directory site         # 本地看页面
```

没有 `ANTHROPIC_API_KEY` 时会跳过 AI 步骤。

## 注意

- 数据来自 Yahoo Finance（非官方接口），偶尔会限流。超过 30% 的期权链拉取失败时，这次运行会报错退出，不覆盖前一天的结果。
- GitHub 的定时任务是 UTC 时间，没有夏令时。workflow 在 13:00 和 14:00 UTC 各触发一次，只有落在纽约时间 9 点的那次会真正运行。GitHub 的定时任务有时会延迟几分钟到几十分钟。
- 仓库是 public，所以页面和数据都是公开的。
