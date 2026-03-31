# Portfolio Multi-Agent Analysis System

A multi-agent workflow that analyses an investor's equity portfolio and produces
a structured trade-decision report. Agents are wired into a directed graph using
**LangGraph**, with conditional routing based on real-time volatility signals.

---

## Table of Contents

- [Overview](#overview)
- [Deterministic Layer](#deterministic-layer)
- [LLM Layer](#llm-layer)
- [Orchestration](#orchestration)
- [Guardrails](#guardrails)
- [Architecture](#architecture)
  - [Agent Graph](#agent-graph)
  - [Conditional Routing](#conditional-routing)
- [Folder Structure](#folder-structure)
- [Shared State](#shared-state)
- [Agents (Subagents)](#agents-subagents)
  - [PortfolioAgent](#portfolioagent)
  - [RiskAgent](#riskagent)
  - [MarketAgent](#marketagent)
  - [NewsAgent](#newsagent)
  - [DecisionAgent](#decisionagent)
  - [CriticAgent](#criticagent)
  - [FormatterAgent](#formatteragent)
- [Tools](#tools)
  - [portfolio_tools](#portfolio_tools)
  - [market_tools](#market_tools)
  - [news_tools](#news_tools)
  - [risk_tools](#risk_tools)
- [LangGraph Integration](#langgraph-integration)
- [Running the Workflow](#running-the-workflow)
- [Sample Output](#sample-output)
- [Extending the System](#extending-the-system)

---

## Overview

The portfolio system answers the question:

> _"Given my current holdings, what should I do — hold, exit, or increase each position?"_

It does this by passing a single shared `PortfolioState` object through a chain
of specialised agents, each enriching the state with new data or decisions.
A critic agent validates the output for consistency before a formatter agent
renders the final human-readable report.

**Key design goals:**

| Goal                | Implementation                                                                        |
| ------------------- | ------------------------------------------------------------------------------------- |
| Single shared state | `PortfolioState` dataclass passed through every LangGraph node                        |
| Decoupled agents    | Each agent only reads/writes specific state fields                                    |
| Conditional logic   | Volatility-driven routing — skip or include `NewsAgent`                               |
| Real LLM decisions  | `DecisionAgent` calls `get_llm()` — provider is auto-inferred from the model name     |
| Live market data    | `market_tools` and `news_tools` fetch live data via `yfinance` + VADER + RSS fallback |

---

## Deterministic Layer

These components produce the same output every time for the same input.
No LLM is involved. They are the foundation the LLM layer builds on top of.

| Component                 | File                       | What it computes                                                                              |
| ------------------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| **Quantitative score**    | `tools/scoring.py`         | Integer score (−5 to +5) from P/E trend, 52-week range, gain/loss, volatility, news sentiment |
| **News score**            | `tools/news_tools.py`      | Aggregates VADER-scored headlines → `−1 / 0 / +1` signal                                      |
| **Risk metrics**          | `tools/risk_tools.py`      | Portfolio value, P&L, weighted volatility, sector breakdown, concentration label              |
| **Portfolio action**      | `tools/rebalance_tools.py` | Sector overweight / diversification recommendation (threshold-based, no LLM)                  |
| **Action floor**          | `decision_agent.py`        | Downgrades `DOUBLE_DOWN → HOLD` when score `< 2`                                              |
| **Allocation sizing**     | `decision_agent.py`        | Computes `allocation_change` from score + weight formula; discards LLM value                  |
| **Confidence override**   | `decision_agent.py`        | `abs(score) ≥ 2 → high`; `≤ 1 → moderate`; LLM value ignored                                  |
| **Structural validation** | `tools/validation.py`      | Rejects malformed `action`, `confidence`, `reason`, `allocation_change` fields                |
| **Stage-1 critic rules**  | `critic_agent.py`          | Low-confidence rejection, exit-rate flag, risky double-down flag                              |

---

## LLM Layer

These components call the language model. Their output is **always
post-processed by the deterministic layer** before being used.

| Component                    | Agent                   | Role                                                                                                                         |
| ---------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Per-ticker decision**      | `DecisionAgent`         | Chooses `action` and writes a human-readable `reason` given the quantitative score, market data, news, and portfolio context |
| **Portfolio-level critique** | `CriticAgent` (Stage 2) | Reviews all decisions holistically — news overreaction, score alignment, risk coherence, cross-ticker inconsistency          |

**Provider / model selection:**

| Env var / CLI                | Purpose                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| `--model MODEL_NAME`         | Decision agent model. Provider is **inferred automatically** from the model name.                 |
| `PORTFOLIO_LLM_MODEL`        | Same as `--model`, env-var form.                                                                  |
| `PORTFOLIO_LLM_PROVIDER`     | Override provider for **custom/unknown models only** (emits a warning when used).                 |
| `PORTFOLIO_CRITIC_LLM_MODEL` | Critic model. Provider is inferred. Recommended: use a different model to avoid self-review bias. |

**Execution:** all per-ticker LLM calls run concurrently via `ThreadPoolExecutor`
(`max_workers = n_tickers`), reducing wall time from `n × ~5s` to `~5–8s`.

---

## Orchestration

The workflow is a **directed acyclic graph** assembled with LangGraph
`StateGraph`. All agents read from and write to a single `PortfolioState`
object — no agent calls another agent directly.

**Graph topology:**

```
PortfolioAgent → RiskAgent → MarketAgent
                                  │
                   volatility > 30%│         within threshold
                                  ▼                  │
                            NewsAgent ───────────────┤
                                                     ▼
                                            DecisionAgent
                                                  │
                                            CriticAgent
                                         ╔══════╧══════╗
                                  approved=True   approved=False
                                         │              │
                                  FormatterAgent   (retry loop, max 1)
```

**Conditional routing:**

- `volatility_router()` runs after `MarketAgent`; routes to `NewsAgent` if any
  ticker volatility > 30 %, otherwise straight to `DecisionAgent`.
- `critic_router()` runs after `CriticAgent`; routes back to `DecisionAgent`
  (with feedback injected) if `approved = False`, up to one retry.
- `--no-news` flag bypasses the volatility conditional entirely.

**Retry loop:** `critic_retry_count` in `PortfolioState` tracks retries;
`workflow.py` caps it at 1 to prevent infinite cycles.

---

## Guardrails

Every output the LLM produces is post-processed by a stack of deterministic
rules before it reaches the user. These guardrails run **after** the LLM
responds and override or reject its output entirely — the LLM never has the
final word on safety-critical fields.

### 1. Action floor — DOUBLE_DOWN requires score ≥ 2

`_apply_action_floor()` in `decision_agent.py`:

- If the LLM outputs `DOUBLE_DOWN` but the quant score is `< 2`, the action is
  downgraded to `HOLD`, `allocation_change` is forced to `0%`, and a note is
  prepended to the reason.
- Prevents the LLM from aggressively adding to positions on a weak single-signal
  conviction (`score = +1`).

### 2. Deterministic allocation sizing (score + weight formula)

`_apply_allocation_change()` in `decision_agent.py`:

- `allocation_change` is **fully overwritten** after parsing — the LLM's
  suggested number is always discarded.
- `EXIT → -100%`, `HOLD → 0%` (hard overrides).
- `REDUCE`: base from score (`-1→-15%`, `-2→-30%`, `≤-3→-45%`) plus
  a concentration penalty (`>10%→-5`, `>20%→-10`, `>30%→-15`), capped at `-70%`.
- `DOUBLE_DOWN`: `score +2 → +10%`, `score ≥+3 → +20%`.
- Guarantees reproducible, identical allocation numbers across every run for
  the same input data.

### 3. Confidence calibration

`_apply_score_confidence()` in `decision_agent.py`:

- LLM-reported confidence is always overridden.
- `abs(score) ≥ 2 → high`; `abs(score) ≤ 1 → moderate`.
- Prevents the LLM from falsely claiming `high` confidence on weak signals.

### 4. Structural validation

`validate_decision()` in `tools/validation.py`:

- Rejects any response where `action`, `confidence`, `reason`, or
  `allocation_change` are missing or malformed.
- On failure the LLM is shown its own bad output and retried once.

### 5. Two-stage critic

`CriticAgent` in `subagents/critic_agent.py`:

- **Stage 1 (hardcoded):** rejects `low` confidence decisions; flags risky
  `DOUBLE_DOWN` on >20% unrealised loss; warns on >50% exit/reduce rate.
- **Stage 2 (LLM):** single portfolio-level call checking news overreaction,
  quant score alignment, risk coherence, and cross-ticker inconsistency.
  Optionally uses a **different model** than the decision agent to avoid
  self-review bias (`PORTFOLIO_CRITIC_LLM_PROVIDER` / `PORTFOLIO_CRITIC_LLM_MODEL`).
- Stage 2 failure is silent — Stage 1 rules always apply regardless.

### 6. LLM JSON corruption guard

- Each provider's `get_llm()` call uses `temperature=0` (Ollama) or low
  temperature to reduce hallucination in structured output.
- The JSON parser strips markdown fences before attempting `json.loads()`.
- A single retry with the bad output appended to the conversation gives the
  model a chance to self-correct before the call is abandoned.

---

## Architecture

### Agent Graph

```
PortfolioAgent
      │
      ▼
  RiskAgent
      │
      ▼
 MarketAgent
      │
      ├─── volatility > 30% ───► NewsAgent ──┐
      │                                       │
      └─── all within threshold ─────────────┤
                                              ▼
                                       DecisionAgent
                                              │
                                              ▼
                                        CriticAgent
                                              │
                                              ▼
                                       FormatterAgent
```

### Conditional Routing

After `MarketAgent` runs, `volatility_router()` inspects every ticker in
`state.stock_insights`. If **any** ticker's `volatility` exceeds
`VOLATILITY_THRESHOLD` (0.30 / 30 %), the graph routes to `NewsAgent` first.
Otherwise it jumps straight to `DecisionAgent`.

```python
# workflow.py
def volatility_router(state: PortfolioState) -> str:
    for insight in state.stock_insights.values():
        if insight.get("volatility", 0) > VOLATILITY_THRESHOLD:
            return "high_volatility"   # → NewsAgent
    return "normal"                    # → DecisionAgent
```

The `--no-news` flag bypasses this conditional entirely and wires a static edge
`market → decision`, useful for fast dry-runs.

---

## Folder Structure

```
src/agents/portfolio/
│
├── README.md                   ← this file
│
├── workflow.py                 ← graph assembly + CLI entry point
│
├── state/
│   └── __init__.py             ← PortfolioState dataclass
│
├── subagents/
│   ├── __init__.py
│   ├── portfolio_agent.py      ← loads positions into state
│   ├── risk_agent.py           ← calculates portfolio risk metrics
│   ├── market_agent.py         ← fetches per-ticker market data
│   ├── news_agent.py           ← fetches news for high-vol tickers
│   ├── decision_agent.py       ← generates trade recommendations
│   ├── critic_agent.py         ← validates decision quality
│   └── formatter_agent.py      ← renders the final report
│
└── tools/
    ├── __init__.py
    ├── mock_data.py            ← mock investor profile + positions
    ├── portfolio_tools.py      ← user profile + position loader
    ├── market_tools.py         ← live price, volatility, trailing/forward P/E via yfinance
    ├── news_tools.py           ← live headlines + VADER sentiment + news score
    ├── risk_tools.py           ← portfolio risk calculator
    ├── scoring.py              ← deterministic quantitative score per ticker
    ├── rebalance_tools.py      ← portfolio-level rebalance / diversification logic
    └── validation.py           ← structural validation for DecisionAgent output
```

The graph is assembled in `workflow.py` using LangGraph's `StateGraph`.

---

## Shared State

**File:** `state/__init__.py`

`PortfolioState` is a Python `dataclass` that acts as the single source of
truth flowing through every node. All fields default to empty collections so
any agent can be run in isolation without requiring prior nodes to have set
anything.

```python
@dataclass
class PortfolioState:
    user_profile:      Dict[str, Any]            # investor name, risk tolerance, horizon
    portfolio:         List[Dict[str, Any]]       # raw positions: ticker, shares, avg_cost, sector
    sector_allocation: Dict[str, float]           # sector → portfolio weight (%)
    risk_metrics:      Dict[str, Any]             # concentration, volatility, PnL totals
    stock_insights:    Dict[str, Dict[str, Any]]  # per-ticker enriched market data
    news:              Dict[str, List[...]]        # per-ticker headlines (high-vol tickers only)
    decisions:         Dict[str, Dict[str, Any]]  # ticker → action, confidence, reason, gain_pct, allocation_change
    critic_feedback:   Dict[str, Any]             # approved flag, warnings, per-ticker issues
    portfolio_action:  Dict[str, Any]             # deterministic rebalance / diversification recommendation
    final_output:      str                        # rendered report text
```

### Field Ownership

| Field               | Written by       | Read by                                                             |
| ------------------- | ---------------- | ------------------------------------------------------------------- |
| `user_profile`      | `PortfolioAgent` | `DecisionAgent`, `FormatterAgent`                                   |
| `portfolio`         | `PortfolioAgent` | `RiskAgent`, `MarketAgent`, `DecisionAgent`                         |
| `sector_allocation` | `RiskAgent`      | `DecisionAgent`, `FormatterAgent`                                   |
| `risk_metrics`      | `RiskAgent`      | `DecisionAgent`, `FormatterAgent`                                   |
| `stock_insights`    | `MarketAgent`    | `NewsAgent`, `DecisionAgent`, `FormatterAgent`, `volatility_router` |
| `news`              | `NewsAgent`      | `DecisionAgent`, `FormatterAgent`                                   |
| `decisions`         | `DecisionAgent`  | `CriticAgent`, `FormatterAgent`                                     |
| `critic_feedback`   | `CriticAgent`    | `FormatterAgent`                                                    |
| `portfolio_action`  | `DecisionAgent`  | `FormatterAgent`                                                    |
| `final_output`      | `FormatterAgent` | `workflow.py` (print)                                               |

---

## Agents (Subagents)

Every agent exposes a single method: `run(state: PortfolioState) -> PortfolioState`.
LangGraph calls this method and passes the returned state to the next node.

---

### PortfolioAgent

**File:** `subagents/portfolio_agent.py`

Loads investor profile and position data as the first node in the graph.

| Reads       | Writes                                  |
| ----------- | --------------------------------------- |
| _(nothing)_ | `state.user_profile`, `state.portfolio` |

**Data source:** `tools/portfolio_tools.get_portfolio()` — currently mock data.
Replace the function body with a brokerage API call, CSV reader, or database
query for production use.

---

### RiskAgent

**File:** `subagents/risk_agent.py`

Calculates portfolio-level risk metrics using current positions. At this point
in the pipeline `stock_insights` is still empty, so `risk_tools.calculate_risk`
falls back to using `avg_cost` as the current price for any ticker without
market data.

| Reads                                     | Writes                                          |
| ----------------------------------------- | ----------------------------------------------- |
| `state.portfolio`, `state.stock_insights` | `state.risk_metrics`, `state.sector_allocation` |

**Metrics produced:**

| Metric                  | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `total_portfolio_value` | Sum of `shares × current_price` across all positions      |
| `unrealized_pnl`        | Total `(current_price − avg_cost) × shares`               |
| `unrealized_pnl_pct`    | P&L as a percentage of cost basis                         |
| `weighted_volatility`   | Portfolio-value-weighted average of per-ticker volatility |
| `sector_allocation`     | Sector → percentage of total portfolio value              |
| `top_sector`            | Sector with the highest allocation                        |
| `concentration_risk`    | `"high"` (>40 %), `"moderate"` (>25 %), or `"low"`        |

---

### MarketAgent

**File:** `subagents/market_agent.py`

Fetches current market data for every position and enriches it with
cost-basis context from the portfolio.

| Reads             | Writes                 |
| ----------------- | ---------------------- |
| `state.portfolio` | `state.stock_insights` |

**Volatility threshold:**

```python
VOLATILITY_THRESHOLD = 0.30   # imported by NewsAgent and workflow.py
```

Each entry in `state.stock_insights[ticker]` contains:

| Key              | Source         | Description                      |
| ---------------- | -------------- | -------------------------------- |
| `price`          | `market_tools` | Current market price             |
| `change_pct`     | `market_tools` | Daily change %                   |
| `volatility`     | `market_tools` | Annualised volatility            |
| `pe_ratio`       | `market_tools` | Trailing price-to-earnings ratio |
| `forward_pe`     | `market_tools` | Forward price-to-earnings ratio  |
| `52w_high`       | `market_tools` | 52-week high                     |
| `52w_low`        | `market_tools` | 52-week low                      |
| `avg_cost`       | `portfolio`    | Your purchase price per share    |
| `shares`         | `portfolio`    | Number of shares held            |
| `sector`         | `portfolio`    | Sector classification            |
| `unrealized_pnl` | computed       | `(price − avg_cost) × shares`    |

---

### NewsAgent

**File:** `subagents/news_agent.py`

**Optional node** — only reached when the `volatility_router` returns
`"high_volatility"`. Fetches recent headlines and sentiment labels for every
ticker whose volatility exceeds `VOLATILITY_THRESHOLD`.

| Reads                  | Writes       |
| ---------------------- | ------------ |
| `state.stock_insights` | `state.news` |

Each entry in `state.news[ticker]` is a list of:

```python
{"headline": str, "sentiment": "positive" | "negative" | "neutral"}
```

**Data source:** `tools/news_tools.get_news(ticker)` — two-stage fetch strategy:

1. **Primary** — `yfinance` (`yf.Ticker(ticker).news`): no API key required.
2. **Fallback** — RSS feeds (Google News, then Yahoo Finance RSS) via `feedparser`,
   used automatically when yfinance returns zero articles.

Both sources score every headline with VADER sentiment analysis.
Returns `[]` only when both sources fail or are empty.

**Empty-result handling:** tickers that return zero articles are not stored in
`state.news`. The DecisionAgent prompt omits the `Recent news:` block entirely
for those tickers — no noise from empty sections.

---

### DecisionAgent

**File:** `subagents/decision_agent.py`

Generates a trade recommendation for every portfolio position using a real LLM.

| Reads                                | Writes            |
| ------------------------------------ | ----------------- |
| `state.stock_insights`, `state.news` | `state.decisions` |

Each entry in `state.decisions[ticker]`:

```python
{
    "action":            "EXIT" | "REDUCE" | "HOLD" | "DOUBLE_DOWN",
    "confidence":        "high" | "moderate" | "low",
    "reason":            str,     # human-readable explanation
    "gain_pct":          float,   # (current_price / avg_cost − 1) × 100, computed locally
    "allocation_change": str,     # e.g. "-15%", "+10%", "0%", "-100%"
}
```

#### Actions

| Action        | Meaning                                                         |
| ------------- | --------------------------------------------------------------- |
| `EXIT`        | Close the full position (`allocation_change` is always `-100%`) |
| `REDUCE`      | Trim part of the position (e.g. `-15%`)                         |
| `HOLD`        | No change (`allocation_change` is always `0%`)                  |
| `DOUBLE_DOWN` | Add to the position (e.g. `+10%`)                               |

#### Allocation change

`allocation_change` is **fully deterministic** — computed from the quant score
and the position's current portfolio weight. The LLM's suggested value is
discarded after parsing.

| Action        | Rule                                                                  |
| ------------- | --------------------------------------------------------------------- |
| `EXIT`        | Always `-100%`                                                        |
| `HOLD`        | Always `0%`                                                           |
| `REDUCE`      | Base from score + concentration penalty (see below), capped at `-70%` |
| `DOUBLE_DOWN` | From score only: score `+2` → `+10%`; score `≥+3` → `+20%`            |

**REDUCE base by score:**

| Score  | Base   |
| ------ | ------ |
| `-1`   | `-15%` |
| `-2`   | `-30%` |
| `≤ -3` | `-45%` |

**Concentration penalty (added on top of base):**

| Position weight | Extra trim |
| --------------- | ---------- |
| > 30 %          | `−15 %`    |
| > 20 %          | `−10 %`    |
| > 10 %          | `−5 %`     |

**Example:** MSFT with score `+1`, action `REDUCE` (due to 39.6 % weight),
base = `−15%`, penalty = `−15%` → always `−30%`. Identical across every run.

#### DOUBLE_DOWN threshold

`DOUBLE_DOWN` requires a quantitative score ≥ 2. A score of +1 is a weak buy
signal and is automatically downgraded to `HOLD` with a note prepended to the
reason. This prevents the LLM from aggressively adding to positions on thin
conviction.

#### Confidence calibration

LLM-reported confidence is overridden deterministically after each response:

| `abs(score)` | Confidence |
| ------------ | ---------- |
| 0 or 1       | `moderate` |
| ≥ 2          | `high`     |

#### Time horizon awareness

The investor's `investment_horizon` (e.g. `"5 years"`, `"18 months"`) is
parsed into a float and injected as an `INVESTMENT HORIZON` section in the
system prompt. For long-term horizons (≥ 3 years):

- The daily price change signal is suppressed in the quantitative score
- The prompt instructs the LLM that daily moves are noise and `HOLD` is
  strongly preferred unless the thesis is broken or the loss exceeds −35 %

#### Parallel execution

All per-ticker LLM calls are dispatched concurrently via a `ThreadPoolExecutor`
(`max_workers = n_tickers`). Telemetry and logging run sequentially after all
futures complete. This reduces total execution time from ~`n × 5s` to ~`5–8s`
regardless of portfolio size.

#### Quantitative scoring

Before calling the LLM, `tools/scoring.py::score_stock()` computes a
deterministic integer score from market data and news sentiment. The score
and the tier label are injected into the prompt as the primary decision anchor:

| Score   | Tier          | LLM guidance                                 |
| ------- | ------------- | -------------------------------------------- |
| ≥ 3     | `strong_buy`  | Lean toward `DOUBLE_DOWN`                    |
| +2      | `buy`         | `HOLD` or `DOUBLE_DOWN`                      |
| +1      | `buy`         | `HOLD` only (insufficient for `DOUBLE_DOWN`) |
| 0       | `neutral`     | `HOLD`                                       |
| −1 / −2 | `sell`        | `REDUCE` before considering `EXIT`           |
| ≤ −3    | `strong_sell` | `EXIT`; `REDUCE` if long-term thesis intact  |

#### LLM provider and model selection

Specify only `--model`. The provider is **inferred automatically** from the model name.
No `--provider` flag is needed or accepted.

| Method            | Example                                         |
| ----------------- | ----------------------------------------------- |
| CLI `--model`     | `--model gpt-4o`                                |
| Environment var   | `$env:PORTFOLIO_LLM_MODEL="gpt-4o"`             |
| Built-in defaults | `ollama` / `gpt-oss:120b` (when nothing is set) |

```powershell
# Use defaults (ollama / gpt-oss:120b)
python -m src.agents.portfolio.workflow

# Switch model — provider is inferred
python -m src.agents.portfolio.workflow --model gpt-4o
python -m src.agents.portfolio.workflow --model gemini-1.5-pro
python -m src.agents.portfolio.workflow --model llama3
```

Default models per provider:

| Provider | Default model    |
| -------- | ---------------- |
| `ollama` | `gpt-oss:120b`   |
| `openai` | `gpt-4o`         |
| `google` | `gemini-1.5-pro` |

Unknown model names fail fast at startup with a clear error listing all known
models across every provider. For custom/fine-tuned models not in the known list,
set `PORTFOLIO_LLM_PROVIDER` explicitly — that env var is only consulted as a
last resort and emits a warning.

---

### CriticAgent

**File:** `subagents/critic_agent.py`

Two-stage validation of decisions produced by `DecisionAgent`.
Does **not** change any decisions — only annotates `critic_feedback`.

| Reads                                   | Writes                  |
| --------------------------------------- | ----------------------- |
| `state.decisions`, `state.user_profile` | `state.critic_feedback` |

#### Stage 1 — Hardcoded rules (always runs)

| Check                   | Threshold                                      | Effect                                     |
| ----------------------- | ---------------------------------------------- | ------------------------------------------ |
| High exit/reduce rate   | > 50 % of positions are `EXIT` or `REDUCE`     | Adds portfolio-level warning               |
| Low-confidence decision | `confidence == "low"`                          | Sets `approved = False`, flags ticker      |
| Risky double-down       | `action == DOUBLE_DOWN` and `gain_pct < −20 %` | Adds portfolio-level warning, flags ticker |

#### Stage 2 — LLM qualitative critique (only when Stage 1 passes)

A single portfolio-level LLM call — **one call for all tickers** — that checks:

1. **Overreaction to news** — is a REDUCE/EXIT driven purely by short-term headlines with no fundamental backing?
2. **Quant score alignment** — does the action contradict the score direction without a stated reason?
3. **Risk/reward coherence** — does the overall mix suit the investor's risk level and horizon?
4. **Internal inconsistency** — conflicting decisions on correlated tickers without explanation.

The LLM for the critic can be a **different model** from the decision agent to
avoid self-review bias. Only the model name is needed — provider is inferred:

```powershell
$env:PORTFOLIO_CRITIC_LLM_MODEL = "gpt-4o"   # openai inferred automatically
```

When unset, falls back to `PORTFOLIO_LLM_MODEL` / `PORTFOLIO_LLM_PROVIDER`.
LLM failures silently fall back to `approved = True` — hardcoded rules still
apply regardless.

`state.critic_feedback` structure:

```python
{
    "approved": bool,
    "warnings": [str, ...],           # portfolio-level warnings
    "per_ticker": {
        ticker: {
            "status": "ok" | "flagged",
            "issues": [str, ...],     # stage-1 or "[LLM critic] ..." prefixed
        }
    }
}
```

---

### FormatterAgent

**File:** `subagents/formatter_agent.py`

Final node. Renders the full analysis report as a multi-section string and
writes it to `state.final_output`. The `workflow.py` main function prints this
after `engine.run()` returns.

| Reads            | Writes               |
| ---------------- | -------------------- |
| All state fields | `state.final_output` |

**Report sections:**

1. **Header** — investor name, risk level, investment horizon
2. **Portfolio Summary** — total value, P&L, weighted volatility, concentration
3. **Sector Allocation** — bar chart sorted by weight descending
4. **Stock Decisions** — per-ticker action icon, confidence, `allocation_change`, P&L, price, reason, critic issues
5. **Portfolio Action** — deterministic rebalance / diversification recommendation (sector > 60 % threshold)
6. **News** _(only when `state.news` is non-empty)_ — headlines with sentiment arrows
7. **Critic Warnings** — portfolio-level flags from `CriticAgent`
8. **Footer** — disclaimer

**Action icons:**

| Action      | Icon |
| ----------- | ---- |
| EXIT        | 🔴   |
| REDUCE      | 🟠   |
| HOLD        | 🟡   |
| DOUBLE_DOWN | 🟢   |

---

## Tools

Tool functions are stateless utilities called by agents. `market_tools` and
`news_tools` fetch live data; `portfolio_tools` reads from `mock_data.py`.
To connect a real brokerage, replace only the `get_portfolio()` body in
`portfolio_tools.py` — no agent code needs to change.

---

### scoring

**File:** `tools/scoring.py`  
**Function:** `score_stock(insight, gain_pct, horizon_years, news_score) → Dict`

Computes a deterministic integer score (typically −5 to +5) from market data
without calling the LLM. Each signal contributes ±1 point:

| Signal                         | Threshold                                                          |
| ------------------------------ | ------------------------------------------------------------------ |
| Daily change (short-term only) | ±3 % triggers ±1                                                   |
| Unrealized gain/loss           | −20 %/+20 % triggers ±1                                            |
| 52-week range position         | Near high or low triggers ±1                                       |
| Forward vs trailing P/E ratio  | Improving/deteriorating ±1                                         |
| News sentiment                 | Majority-positive → +1; majority-negative → −1 (via `_sig_news()`) |

`news_score` defaults to `0` (backward-compatible) so tickers without news
are unaffected. For horizons ≥ 3 years the daily-change signal is suppressed
entirely. The result includes a `long_term: bool` flag so the prompt can note
when daily change was excluded.

---

### rebalance_tools

**File:** `tools/rebalance_tools.py`  
**Function:** `compute_portfolio_action(sector_allocation, decisions, risk_metrics, portfolio) → Dict`

Produces a deterministic portfolio-level action recommendation (independent of
the LLM). Two rules are evaluated:

| Rule                  | Threshold  | Output                                                  |
| --------------------- | ---------- | ------------------------------------------------------- |
| Top-sector overweight | > 60 %     | `rebalance=True`, `reduce_sector`, `priority_exits`     |
| Too few sectors       | < 3 unique | `add_diversification=True`, `missing_sectors` (up to 4) |

The result is stored in `state.portfolio_action` and rendered in the
`── PORTFOLIO ACTION ──` section of the report.

---

### validation

**File:** `tools/validation.py`  
**Function:** `validate_decision(decision) → (bool, str)`

Structural validation called after every LLM response. Checks:

- `action` is one of `EXIT | REDUCE | HOLD | DOUBLE_DOWN`
- `confidence` is one of `high | moderate | low`
- `reason` is a non-empty string
- `allocation_change` matches `^[+-]?\d+(\.\d+)?%$`

---

### portfolio_tools

**File:** `tools/portfolio_tools.py`  
**Function:** `get_portfolio() → Dict`

Returns user profile and list of positions from `tools/mock_data.py`.
Mock investor: **Alex Johnson**, moderate risk, 5-year horizon, 6 positions
(META, MSFT, NVDA, TSLA, SOFI, AMZN).

---

### market_tools

**File:** `tools/market_tools.py`  
**Function:** `get_stock_data(ticker: str) → Dict`

Fetches live data via `yfinance`: current price, daily change %, annualised
30-day volatility (`std × √252`), trailing P/E, forward P/E, and 52-week high/low.

The `volatility_router` in `workflow.py` compares each ticker's live volatility
against `VOLATILITY_THRESHOLD = 0.30`. Any ticker above the threshold causes
the graph to route through `NewsAgent`.

---

### news_tools

**File:** `tools/news_tools.py`

#### `get_news(ticker: str) → List[Dict]`

Two-stage fetch strategy:

| Stage        | Source                                               | Condition                       |
| ------------ | ---------------------------------------------------- | ------------------------------- |
| 1 (primary)  | `yf.Ticker(ticker).news` via `yfinance`              | Always tried first              |
| 2 (fallback) | Google News RSS → Yahoo Finance RSS via `feedparser` | Only when stage 1 returns empty |

Every headline from either source is scored with VADER sentiment analysis
(`compound ≥ 0.05` → positive, `≤ −0.05` → negative, else neutral).
Returns `[]` only when both sources fail or are empty.

#### `compute_news_score(articles: List[Dict]) → int`

Collapses a list of VADER-scored articles into a single `−1 / 0 / +1` signal:

1. Map each article: `positive → +1`, `neutral → 0`, `negative → −1`
2. Sum all mapped values
3. Sign-normalize: `sum > 0 → +1`, `sum == 0 → 0`, `sum < 0 → −1`
4. Returns `0` for an empty list

**Example:** sentiments `['negative', 'negative', 'positive', 'neutral', 'negative']`  
→ mapped values `[−1, −1, +1, 0, −1]` → sum `−2` → **score `−1`**

The result is passed to `score_stock(news_score=...)` in `DecisionAgent`, where
`_sig_news()` translates it into the quantitative signal stack.

---

### risk_tools

**File:** `tools/risk_tools.py`  
**Function:** `calculate_risk(positions, stock_insights) → Dict`

Pure function — no side effects, no state dependency. Computes:

- Total portfolio value and unrealized P&L from position data
- Portfolio-value-weighted volatility
- Sector breakdown as percentages
- Concentration risk label based on the largest sector weight

Falls back to `avg_cost` (from the position) and `0.20` volatility for tickers
not present in `stock_insights`.

---

## LangGraph Integration

The workflow is assembled in `workflow.py` using LangGraph's `StateGraph`,
the same library used by the ecommerce agent in this repository.

```python
graph = StateGraph(PortfolioState)
graph.add_node("portfolio_agent", PortfolioAgent().run)
# ... remaining nodes
graph.add_conditional_edges("market", volatility_router, {...})
return graph.compile()
```

Node names `"portfolio_agent"` and `"news_agent"` are suffixed with `_agent`
because LangGraph 0.3+ rejects node names that collide with `PortfolioState`
field names (`portfolio` and `news`).

`compiled.invoke(PortfolioState())` runs the full graph and returns the
final state.

---

## Running the Workflow

Make sure you are in the repository root and the virtual environment is active.

### Standard run (conditional news routing)

```powershell
python -m src.agents.portfolio.workflow
```

With live data, any ticker whose annualised 30-day volatility exceeds 0.30
triggers the `NewsAgent` branch before `DecisionAgent`.

### Skip NewsAgent

```powershell
python -m src.agents.portfolio.workflow --no-news
```

Wires a static `market → decision` edge. Useful for fast runs or when live
news data is not required.

### Console output during execution

```
======================================================================
  STARTING PORTFOLIO ANALYSIS WORKFLOW
======================================================================

  Loaded 6 positions for Alex Johnson (risk: moderate)
  Concentration: HIGH | Top sector: Technology (69.3%) | Portfolio value: $...
  Fetched data for 6 tickers. High-volatility (>30%): ['TSLA', ...]
  TSLA: 5 articles fetched
  META:  DOUBLE_DOWN   [MODERATE]  gain: -10.3%
  ...
  Critic: APPROVED | Warnings: 0 | Flagged tickers: 0
  Report generated.
```

---

## Sample Output

```
======================================================================
  PORTFOLIO ANALYSIS REPORT
======================================================================
  Investor   : Alex Johnson
  Risk Level : Moderate
  Horizon    : 5 years

── PORTFOLIO SUMMARY ──────────────────────────────────────────────
  Total Value      :      $57,070.00
  Unrealized P&L   :       +$5,070.00  (+9.75%)
  Weighted Vol.    :          25.78%
  Concentration    :          HIGH

── SECTOR ALLOCATION ───────────────────────────────────────────────
  Technology            69.3%  █████████████
  Finance               10.2%  ██
  Consumer Cyclical      9.1%  █
  Communication          8.8%  █

── STOCK DECISIONS ─────────────────────────────────────────────────
  🟡  MSFT    HOLD          [HIGH]    alloc:    0%  PnL: +48.2%  @ $415.00
       → Strong fundamentals and forward PE improving. Thesis intact.
  🟢  NVDA    DOUBLE_DOWN   [HIGH]    alloc:  +10%  PnL: +12.1%  @ $875.00
       → Score +3: strong buy. 52w range near mid-point; earnings growth intact.
  🟠  META    REDUCE        [HIGH]    alloc:  -15%  PnL:  -8.3%  @ $412.50
       → Score -1: weak sell. Trim overweight position while thesis remains intact.
  🔴  TSLA    EXIT          [HIGH]    alloc: -100%  PnL: -20.5%  @ $175.00
       → Down 20.5% with high volatility (55%). Risk-reward is unfavourable. Cut losses.

── PORTFOLIO ACTION ────────────────────────────────────────────────
  ⚠  REBALANCE RECOMMENDED
  Reduce sector  : Technology (currently 69.3% → target ≤ 60%)
  Priority exits : TSLA  (already flagged for EXIT)
  Diversify into : Healthcare, Consumer Staples
  Summary        : Technology overweight at 69.3%; trim to ≤60% target.

── NEWS  (High-Volatility Tickers) ─────────────────────────────────
  TSLA:
    ↓  Tesla cuts prices for the third time this year
    ↓  Cybertruck recall issued for accelerator defect
    ↑  Tesla energy division posts record quarterly revenue

  Overall Review : ✅  APPROVED
======================================================================
  ⚠  DISCLAIMER: AI-generated decision support only.
     Consult a licensed financial advisor before making any trades.
======================================================================
```

---

## Extending the System

### Switch the LLM provider or model

Use the `--model` CLI flag — provider is inferred, no code changes needed:

```powershell
python -m src.agents.portfolio.workflow --model gpt-4-turbo
python -m src.agents.portfolio.workflow --model gemini-pro
python -m src.agents.portfolio.workflow --model llama3
```

Or set the environment variable to make the selection persist across runs:

```powershell
$env:PORTFOLIO_LLM_MODEL = "gpt-4-turbo"
python -m src.agents.portfolio.workflow
```

Provider is inferred automatically from the model name. Running
`python -m src.agents.portfolio.workflow --help` shows the current active
model in the flag description.

### Add a new ticker

Add a row to `get_portfolio()` in `tools/portfolio_tools.py`, then add
corresponding entries in `_MOCK_MARKET_DATA` (`market_tools.py`) and
`_MOCK_NEWS` (`news_tools.py`).

### Add a new agent node

1. Create `subagents/my_agent.py` with a `run(state) -> PortfolioState` method.
2. Import and register it in `workflow.py`:
   ```python
   from src.agents.portfolio.subagents.my_agent import MyAgent
   engine.add_node("my_agent", MyAgent())
   engine.add_edge("critic", "my_agent")
   engine.add_edge("my_agent", "formatter")
   # remove old edge: engine.add_edge("critic", "formatter")
   ```

### Replace mock data with real APIs

| Tool file            | Suggested replacement                                 |
| -------------------- | ----------------------------------------------------- |
| `portfolio_tools.py` | Brokerage API (Alpaca, Interactive Brokers, Schwab)   |
| `market_tools.py`    | Polygon.io, Alpha Vantage, Yahoo Finance (`yfinance`) |
| `news_tools.py`      | NewsAPI, Polygon News, Google News RSS                |

### Connect to the registry

To make the portfolio workflow discoverable alongside the ecommerce agent,
add a `registry.py` following the pattern in `src/agents/ecommerce/registry.py`.
