# sell-puter

每个交易日早上 9 点（美东）自动扫描 S&P 500 + 自选名单，按规则筛出适合卖现金担保看跌期权（cash-secured put）的合约，用 Claude 写质地点评和风险笔记，发布到 GitHub Pages。

> 这是按你自己定的规则筛出来的数据，加上 AI 生成的摘要，不构成投资建议。卖 put 在股价大跌时可能被行权并产生较大亏损。

## 流程

**每个交易日早上 9 点（美东）**，`daily.yml` → `sellputer.main`：

1. **股票池**：S&P 500 + `extra_tickers` + Reddit 过去 24 小时热门（ApeWisdom）+ Yahoo 成交最活跃 / 跌幅榜
2. **质地**：S&P 500 和自选名单直接通过；其他来源要求市值 ≥ $100 亿且盈利；所有股票股价 ≥ $20
3. **期权扫描**：后端抓 7–42 天到期、|Delta| 0.15–0.35、IV ≥ 40%、年化 ≥ 15%、一手担保 ≤ $40,000 的合约；页面上再按 IV、年化、Delta、DTE、担保金自由筛选（默认 IV ≥ 60%、年化 ≥ 30%、Delta 0.20–0.30）
4. **盘前处理**：9 点时期权还没开盘，报价是昨收。用昨收价反推 IV，再用盘前价重新估算 Delta 和权利金；盘前涨跌超过 3% 会标红
5. **保守估价**：权利金按"中间价 − ¼ 价差"估成交价，年化也按它算；每只股票按（到期日 × Delta 档位）各留最好的一张，安全些的行权价不会被挤掉；页面显示"缓冲 σ"（行权价离现价几个标准差）。公司还没公布下次财报日时，按上次财报日 + 75 天估算（标 ≈）
6. **AI 笔记**（`claude-sonnet-5-5`，预算 $1.8）：先给前 30 只每 8 只一批写简要笔记，保证都有；再用剩下的预算给前 10 只做单独深度分析（至少联网搜索一次、读原文、期权视角、接盘视角、催化剂；来源链接只保留搜索真实返回的）；预算用完自动停
7. **发布**：结果 commit 到 `site/data/`，`site/` 部署到 GitHub Pages

**每个交易日晚上 8 点（美东）**，`dream.yml` → `sellputer.dream`（预算 $0.7）：

1. **打分**：把过往每天页面默认显示的合约，和之后的真实收盘价对比：到期作废 / 被行权 / 进行中、是否触及行权价、盈亏；按 AI 风险评级、质地、来源、IV、DTE、Delta、财报、盘前跳空分组统计 → `site/data/review.json`
2. **复盘（dreaming）**：AI 对照"当天怎么说"和"后来发生了什么"，更新自己的经验库 `memory/lessons.json`（新增、修正、合并、推翻，每条带依据和状态）
3. **注入**：经验渲染成 `memory/MEMORY.md`，第二天早上的 AI 分析会读到
4. **建议**：筛选规则、AI 分析、页面的改进建议显示在复盘页，不会自动生效，由你决定

两个 workflow 都会跳过美股休市日。每个交易日 AI 总花费约 $2.5 以内。

## 一次性设置

1. 仓库 Settings → Secrets and variables → Actions → 新建 `ANTHROPIC_API_KEY`
2. 仓库 Settings → Pages → Source 选 **GitHub Actions**
3. Actions → daily-scan → Run workflow，手动跑一次

## 改规则

所有参数都在 [config.toml](config.toml)，改完 push 即可。

## 本地运行

```bash
uv run python -m sellputer.main                   # 全量扫描，写 site/data/
uv run python -m sellputer.dream                  # 复盘 + 更新经验库
SELLPUTER_LIMIT=50 uv run python -m sellputer.main   # 只扫前 50 只，快速测试
python3 -m http.server 8765 --directory site         # 本地看页面
```

没有 `ANTHROPIC_API_KEY` 时会跳过 AI 步骤。

## 注意

- 数据来自 Yahoo Finance（非官方接口），偶尔会限流。超过 30% 的期权链拉取失败时，这次运行会报错退出，不覆盖前一天的结果。
- GitHub 的定时任务是 UTC 时间，没有夏令时。workflow 在 13:00 和 14:00 UTC 各触发一次，只有落在纽约时间 9 点的那次会真正运行。GitHub 的定时任务有时会延迟几分钟到几十分钟。
- 仓库是 public，所以页面和数据都是公开的。
