# Portfolio Multi-Agent Analysis System

A multi-agent workflow that analyses an investor's equity portfolio and produces
a structured trade-decision report. Agents are wired into a directed graph using
**LangGraph**, with conditional routing based on real-time volatility signals.

---

## Table of Contents

- [Overview](#overview)
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

| Goal                | Implementation                                                         |
| ------------------- | ---------------------------------------------------------------------- |
| Single shared state | `PortfolioState` dataclass passed through every LangGraph node         |
| Decoupled agents    | Each agent only reads/writes specific state fields                     |
| Conditional logic   | Volatility-driven routing — skip or include `NewsAgent`                |
| Real LLM decisions  | `DecisionAgent` calls `get_llm()` via `PORTFOLIO_LLM_PROVIDER` env var |
| Live market data    | `market_tools` and `news_tools` fetch live data via `yfinance` + VADER |

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
    ├── news_tools.py           ← live headlines + VADER sentiment
    └── risk_tools.py           ← portfolio risk calculator
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
    user_profile:     Dict[str, Any]            # investor name, risk tolerance, horizon
    portfolio:        List[Dict[str, Any]]       # raw positions: ticker, shares, avg_cost, sector
    sector_allocation: Dict[str, float]          # sector → portfolio weight (%)
    risk_metrics:     Dict[str, Any]             # concentration, volatility, PnL totals
    stock_insights:   Dict[str, Dict[str, Any]]  # per-ticker enriched market data
    news:             Dict[str, List[...]]        # per-ticker headlines (high-vol tickers only)
    decisions:        Dict[str, Dict[str, Any]]  # ticker → action, reason, confidence, gain_pct
    critic_feedback:  Dict[str, Any]             # approved flag, warnings, per-ticker issues
    final_output:     str                        # rendered report text
```

### Field Ownership

| Field               | Written by       | Read by                                                             |
| ------------------- | ---------------- | ------------------------------------------------------------------- |
| `user_profile`      | `PortfolioAgent` | `FormatterAgent`                                                    |
| `portfolio`         | `PortfolioAgent` | `RiskAgent`, `MarketAgent`                                          |
| `sector_allocation` | `RiskAgent`      | `FormatterAgent`                                                    |
| `risk_metrics`      | `RiskAgent`      | `FormatterAgent`                                                    |
| `stock_insights`    | `MarketAgent`    | `NewsAgent`, `DecisionAgent`, `FormatterAgent`, `volatility_router` |
| `news`              | `NewsAgent`      | `DecisionAgent`, `FormatterAgent`                                   |
| `decisions`         | `DecisionAgent`  | `CriticAgent`, `FormatterAgent`                                     |
| `critic_feedback`   | `CriticAgent`    | `FormatterAgent`                                                    |
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

**Data source:** `tools/news_tools.get_news(ticker)` — fetches live headlines
from `yfinance` and scores each headline with VADER sentiment analysis.
Returns `[]` on any fetch error.

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
    "action":     "EXIT" | "HOLD" | "DOUBLE_DOWN",
    "confidence": "high" | "moderate" | "low",
    "reason":     str,       # human-readable explanation
    "gain_pct":   float,     # (current_price / avg_cost − 1) × 100
}
```

#### LLM provider selection

Set the `PORTFOLIO_LLM_PROVIDER` environment variable before running:

```powershell
$env:PORTFOLIO_LLM_PROVIDER = "openai"   # or "google", defaults to "ollama"
python -m src.agents.portfolio.workflow
```

The agent uses `get_llm()` from `src/llm/providers.py` (the same factory used
by the ecommerce agent) behind an `@lru_cache(maxsize=1)` singleton.
A structured JSON prompt asks the model for `action`, `confidence`, and
`reason`; `gain_pct` is computed locally so arithmetic is always exact.

If the LLM call fails or returns unparseable output, the exception propagates —
there is no silent fallback. Configure the provider before running (see above).

---

### CriticAgent

**File:** `subagents/critic_agent.py`

Validates the quality and consistency of decisions produced by `DecisionAgent`.
Does **not** change any decisions — only annotates `critic_feedback`.

| Reads             | Writes                  |
| ----------------- | ----------------------- |
| `state.decisions` | `state.critic_feedback` |

**Checks performed:**

| Check                    | Threshold                                      | Effect                                     |
| ------------------------ | ---------------------------------------------- | ------------------------------------------ |
| High portfolio exit rate | > 50 % of positions are EXIT                   | Adds portfolio-level warning               |
| Low-confidence decision  | `confidence == "low"`                          | Sets `approved = False`, flags ticker      |
| Risky double-down        | `action == DOUBLE_DOWN` and `gain_pct < −20 %` | Adds portfolio-level warning, flags ticker |

`state.critic_feedback` structure:

```python
{
    "approved": bool,
    "warnings": [str, ...],           # portfolio-level warnings
    "per_ticker": {
        ticker: {
            "status": "ok" | "flagged",
            "issues": [str, ...],
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
4. **Stock Decisions** — per-ticker action icon, confidence, P&L, price, reason, critic issues
5. **News** _(only when `state.news` is non-empty)_ — headlines with sentiment arrows
6. **Critic Warnings** — portfolio-level flags from `CriticAgent`
7. **Footer** — disclaimer

**Action icons:**

| Action      | Icon |
| ----------- | ---- |
| EXIT        | 🔴   |
| HOLD        | 🟡   |
| DOUBLE_DOWN | 🟢   |

---

## Tools

Tool functions are stateless utilities called by agents. `market_tools` and
`news_tools` fetch live data; `portfolio_tools` reads from `mock_data.py`.
To connect a real brokerage, replace only the `get_portfolio()` body in
`portfolio_tools.py` — no agent code needs to change.

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
**Function:** `get_news(ticker: str) → List[Dict]`

Fetches recent headlines from `yf.Ticker(ticker).news` and scores each one
with VADER sentiment analysis (`compound ≥ 0.05` → positive,
`≤ −0.05` → negative, else neutral). Returns `[]` on any fetch error.

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
  Technology            49.3%  █████████
  Finance               10.2%  ██
  Healthcare             6.6%  █
  Automotive             3.1%

── STOCK DECISIONS ─────────────────────────────────────────────────
  🟡  AAPL    HOLD         [HIGH]   PnL: +22.3%  @ $189.50
       → Healthy gain of 22.3% with clean news feed. Thesis intact.
  🟡  MSFT    HOLD         [HIGH]   PnL: +48.2%  @ $415.00
       → Healthy gain of 48.2% with clean news feed. Thesis intact.
  ...
  🔴  TSLA    EXIT         [HIGH]   PnL: -20.5%  @ $175.00
       → Down 20.5% with high volatility (55%). Risk-reward is unfavourable. Cut losses.

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

### Switch the LLM provider

Set `PORTFOLIO_LLM_PROVIDER` to `openai` or `google` before running — no code
changes needed. The default is `ollama`.

```powershell
$env:PORTFOLIO_LLM_PROVIDER = "openai"
python -m src.agents.portfolio.workflow
```

To customise the model within a provider, edit `src/llm/providers.py`
(e.g. change `"gpt-4o"` to `"gpt-4-turbo"` inside `OpenAIProvider`).

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
