# AlphaScannerAgent — Quality-Stock Opportunity Scanner

> **Multi-agent platform** › `src/agents/opportunity/` — Live BUY Signal Scanner

> ⚠️ **EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE**
> This agent produces quantitative signals and LLM-derived commentary for learning purposes only.
> Nothing it outputs constitutes financial advice or a recommendation to buy or sell any security.
> **Do not use these signals to make real investment decisions.**

A production-grade, event-driven three-stage agent that scans live market data during NYSE/NSE trading hours and surfaces **quality-first BUY opportunities**. It ranks names by deterministic business-quality signals first, then applies news sentiment and a final LLM BUY/IGNORE decision.

> **Scope constraint:** This agent only ever outputs `BUY SIGNAL` or `IGNORE`. It has no concept of SELL, REDUCE, or EXIT — those decisions belong exclusively to `PortfolioAgent`. Portfolio data is used only for warning overlays, not for the core quality ranking.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [File Structure](#file-structure)
- [Pipeline Stages](#pipeline-stages)
  - [1. Market Data Fetch](#1-market-data-fetch)
  - [2. PreFilterEngine](#2-prefilterengine)
  - [3. SignalEngine](#3-signalengine)
  - [4. Candidate Filter](#4-candidate-filter)
  - [5. NewsNode — Sentiment Analysis](#5-newsnode--sentiment-analysis)
  - [6. OpportunityDecisionAgent (LLM)](#6-opportunitydecisionagent-llm)
  - [7. Sort and Emit](#7-sort-and-emit)
- [State Object](#state-object)
- [Output Format](#output-format)
- [Observability & Telemetry](#observability--telemetry)
- [Running the Agent](#running-the-agent)
  - [CLI — Single Scan](#cli--single-scan)
  - [CLI — Continuous Batch Scan](#cli--continuous-batch-scan)
  - [Programmatic API](#programmatic-api)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Supported LLM Providers](#supported-llm-providers)
  - [Tuning Constants](#tuning-constants)
- [Real-Time & Event-Driven Behavior](#real-time--event-driven-behavior)
- [Design Decisions](#design-decisions)

---

## Architecture Overview

```
Watchlist (explicit --tickers  OR  get_liquid_universe(--top-n, market))
        │  US: up to 500 S&P 500 constituents
        │  IN / IN_MID / IN_SMALL: NIFTY 50 / MIDCAP 100 / SMALLCAP 100
        │
        ▼  ──────────────────────────────────────────────────────────────────
        │                      [scanner node]
        │   1. Parallel fetch              (_fetch_extended — 10 workers)
        │   2. PreFilter                   (OR-logic triage, zero LLM cost)
        │   3. SignalEngine                (8 weighted signals, deterministic)
        │   4. Candidate filter            (score >= 2 + cooldown clear)
        │
        ▼  ──────────────────────────────────────────────────────────────────
        │                      [news node]
        │   5. Headline fetch              (up to 5 headlines via yfinance)
        │      LLM sentiment per candidate (positive / neutral / negative)
        │      + one-line catalyst
        │
        ▼  ──────────────────────────────────────────────────────────────────
        │                      [decision node]
        │   6. LLM BUY/IGNORE per candidate
        │      (quantitative score + analyst consensus + vol pressure + news)
        │   7. Sort by confidence → effective score
        │
        ▼  ──────────────────────────────────────────────────────────────────
   buy_opportunities  [{ticker, action ("BUY SIGNAL"), confidence, sector,
                        opportunity_score, news_sentiment, news_catalyst, ...}]
```

---

## File Structure

```
src/agents/opportunity/
├── __init__.py
├── state.py               OpportunityState dataclass
├── workflow.py            LangGraph graph + batch loop + CLI entry point
├── engines/
│   ├── __init__.py
│   ├── prefilter_engine.py    PreFilterEngine  — lightweight triage
│   ├── signal_engine.py       SignalEngine     — deterministic 8-signal scorer
│   └── decision_agent.py      OpportunityDecisionAgent  — LLM BUY/IGNORE
├── nodes/
│   ├── __init__.py
│   ├── alpha_scanner_agent.py AlphaScannerAgent — stages 1-5 (fetch → guards)
│   ├── news_node.py           NewsNode          — stage 6 (headlines + sentiment)
│   └── decision_node.py       DecisionNode      — stages 7-9 (LLM + sort)
└── markets/
    ├── __init__.py
    └── market_strategy.py     MarketStrategy ABC → _NSEMarketStrategy
                               → USMarketStrategy (US)
                               → INMarketStrategy (IN)
                               → INMidCapStrategy (IN_MID)
                               → INSmallCapStrategy (IN_SMALL)
```

---

## Pipeline Stages

### 1. Market Data Fetch

**File:** `nodes/alpha_scanner_agent.py` → `_fetch_extended(ticker)`

All tickers in the watchlist are fetched **in parallel** using `ThreadPoolExecutor` (up to `_MAX_FETCH_WORKERS` = 10 concurrent threads). Each fetch makes a single `yf.Ticker(ticker).info` call and returns:

| Field            | Source                                                      |
| ---------------- | ----------------------------------------------------------- |
| `price`          | `currentPrice` / `regularMarketPrice` / `previousClose`     |
| `change_pct`     | Computed from `regularMarketPreviousClose`                  |
| `volatility`     | Annualised std dev from 30-day daily returns                |
| `pe_ratio`       | `trailingPE`                                                |
| `forward_pe`     | `forwardPE`                                                 |
| `52w_high/low`   | `fiftyTwoWeekHigh` / `fiftyTwoWeekLow`                      |
| `sector`         | `info["sector"]` (used for portfolio guardrails)            |
| `volume`         | `regularMarketVolume`                                       |
| `avg_volume`     | `averageVolume` (30-day average)                            |
| `analyst_rating` | `recommendationKey` (e.g. `"buy"`, `"hold"`, `"sell"`)      |
| `analyst_count`  | `numberOfAnalystOpinions`                                   |
| `analyst_target` | `targetMeanPrice` — mean analyst price target (float)       |
| `vol_pressure`   | Derived: `"buying"` / `"selling"` / `"neutral"` (see below) |

**Volume pressure proxy**

`vol_pressure` is derived from price direction × volume spike — no paid Level-2 data needed:

```
volume >= avg_volume × 1.5  AND  change_pct > 0  →  "buying"
volume >= avg_volume × 1.5  AND  change_pct < 0  →  "selling"
otherwise                                          →  "neutral"
```

Tickers that fail to fetch (network error, invalid symbol) are recorded in `state.scan_errors` and silently skipped.

---

### 2. PreFilterEngine

**File:** `engines/prefilter_engine.py`

A lightweight **OR-logic** gate that runs before the scoring engine. A ticker passes if **any one** condition is true. The goal is to discard clearly uninteresting tickers at zero LLM cost when scanning large watchlists (100+ tickers).

| Condition                       | Threshold       | Rationale                                 |
| ------------------------------- | --------------- | ----------------------------------------- |
| `abs(change_pct) > 3.0%`        | 3.0             | Meaningful intraday price movement        |
| Price in lower 20% of 52w range | 20th percentile | Potential value-entry zone                |
| `volume > avg_volume × 1.5`     | 1.5×            | Unusual institutional or retail interest  |
| `volatility > 35%`              | 0.35 annualised | Elevated risk warrants closer examination |

A ticker with no price data is unconditionally rejected. All thresholds are defined in `_PREFILTER_THRESHOLDS` — no magic numbers in logic.

---

### 3. SignalEngine

**File:** `engines/signal_engine.py`

Deterministic quantitative scoring applied to every ticker that passed the prefilter. Eight signals fire independently using a configurable `_WEIGHTS` dict; their points are summed into a final score.

| Signal             | Condition                                             | Weight | Rationale                                         |
| ------------------ | ----------------------------------------------------- | ------ | ------------------------------------------------- |
| `pe_improvement`   | `forward_pe < pe_ratio`                               | **+1** | Earnings growth expected — valuation improving    |
| `52w_lower_band`   | Price in lower 30% of 52w range                       | **+1** | Potential value-entry zone                        |
| `analyst_bullish`  | `buy`/`strong_buy`, ≥ 3 analysts, ≥ 20% target upside | **+1** | Wall Street consensus supports new entry          |
| `buying_pressure`  | price up + volume ≥ 1.5× avg                          | **+1** | Institutional accumulation proxy                  |
| `capitulation`     | price ≤ −8% + volume ≥ 3× avg                         | **+2** | Panic exhaustion — mean-reversion entry zone      |
| `near_52w_high`    | Price within 5% of 52w high                           | **−1** | Chasing strength — unfavourable entry risk/reward |
| `analyst_bearish`  | `underperform`/`sell`, ≥ 3 analysts                   | **−1** | Professional consensus against new entry          |
| `selling_pressure` | price ≤ −8% without panic volume                      | **−1** | Distribution signal without exhaustion            |
| `volatility`       | `volatility > 35%` (suppressed during capitulation)   | **−1** | Elevated entry risk penalty                       |

> **Capitulation suppresses volatility penalty:** During a capitulation event (drop ≥8% on ≥3× volume), high volatility is expected and already captured by the +2 signal — penalising it a second time would unfairly double-count the same risk.

**Tier mapping (raised for selectivity):**

| Score | Tier           |
| ----- | -------------- |
| ≥ +3  | `"strong_buy"` |
| ≥ +2  | `"buy"`        |
| ≤ +1  | `"neutral"`    |
| < 0   | `"avoid"`      |

**Opportunity type — deterministic, not LLM-derived:**

| Condition                        | Type         |
| -------------------------------- | ------------ |
| `52w_lower_band` signal fired    | `"dip_buy"`  |
| Only `pe_improvement` fired      | `"value"`    |
| Neither fundamental signal fired | `"momentum"` |

**Output per ticker:**

```python
{
    "score":   int,        # net weighted signal total
    "signals": List[str],  # human-readable descriptions including weights
    "tier":    str,        # "strong_buy" | "buy" | "neutral" | "avoid"
    "type":    str,        # "dip_buy" | "value" | "momentum"
}
```

---

### 3a. Opportunity Score (composite ranking)

**File:** `nodes/alpha_scanner_agent.py` → `_compute_opportunity_score()`

After scoring, each qualified ticker receives a **composite ranking metric** that blends four orthogonal dimensions into a single float in `[0.0, 1.0]`:

```
opportunity_score =
    0.4 × normalized_signal_score   (raw_score / 5.0)
  + 0.3 × analyst_upside            (50% upside = 1.0, clamped)
  + 0.2 × volume_spike              (10× avg volume = 1.0, clamped)
  + 0.1 × freshness                 (1.0 = new idea, 0.0 = in recent_signal_context)
```

This composite is used for **ranking before the cooldown gate** — highest `opportunity_score` first. Two tickers at the same raw signal score (e.g., both `+3`) are now differentiated by analyst upside and volume spike. The `opportunity_score` is stored in the signal dict, in `buy_entry`, and printed in the output.

---

### 4. Candidate Filter

**File:** `nodes/alpha_scanner_agent.py`

Two conditions must both hold for a ticker to advance to the news and decision stages:

1. **Score ≥ `_CANDIDATE_MIN_SCORE`** (default: 2) — tier must be `"buy"` or `"strong_buy"`
2. **Cooldown clear** — the ticker must not have emitted a BUY signal within the configured cooldown window

Tickers are sorted by `opportunity_score` descending **before** the cooldown gate, so the highest-conviction ideas always compete first regardless of dict iteration order.

**Configurable cooldown unit** (`_COOLDOWN_UNIT`):

| Unit      | Effective window                                    |
| --------- | --------------------------------------------------- |
| `minutes` | `_COOLDOWN_MINUTES` minutes (default: 30, intraday) |
| `hours`   | `_COOLDOWN_MINUTES` hours (hourly scan cadence)     |
| `days`    | `_COOLDOWN_MINUTES` calendar days (daily cadence)   |

Cooldown state is carried forward between batch-scan cycles via `state.recent_signals`.

**Freshness bypass (`_is_fresh_despite_cooldown`):** A ticker inside its cooldown window re-enters the candidate list if a new material catalyst has emerged:

- Capitulation event fires (drop ≥8% on ≥3× avg volume) — panic exhaustion is always a fresh setup
- Price has dropped ≥5% since the price recorded at the last BUY signal (new lower entry zone)

Bypassed tickers are tagged `override_reason = "fresh_catalyst"` in the signal dict.

---

### 5. NewsNode — Sentiment Analysis

**File:** `nodes/news_node.py`

A dedicated LangGraph node that runs **after** candidates are filtered and **before** the LLM decision. Only candidates (score ≥ 1, cooldown clear) receive news analysis — fetching headlines for every ticker in a 200-stock universe would be wasteful.

**Processing per ticker:**

1. Fetch up to `_MAX_HEADLINES` (5) recent news titles via `yf.Ticker(ticker).news`
2. Send headlines to the LLM with a strict analyst-prompt
3. Parse structured JSON response

**Output per ticker written to `state.news_sentiment`:**

```python
{
    "sentiment":      "positive" | "neutral" | "negative",
    "catalyst":       "One-sentence primary driver.",
    "headline_count": int   # number of headlines found
}
```

**Concurrency:** Up to `_MAX_NEWS_WORKERS` (3) parallel threads, capped further by `OllamaProvider.max_concurrency = 1`.

**Fallback:** Any fetch or LLM failure returns `{"sentiment": "neutral", "catalyst": "No recent news found.", "headline_count": 0}` — the scan never halts on a single ticker failure.

---

### 6. OpportunityDecisionAgent (LLM)

**File:** `engines/decision_agent.py`, `nodes/decision_node.py`

A per-ticker LLM call that produces a structured `BUY` or `IGNORE` verdict. The LLM receives four enriched context blocks:

| Context block      | Source                                                  |
| ------------------ | ------------------------------------------------------- |
| Quantitative score | SignalEngine — score, tier, fired signals               |
| Market data        | `_fetch_extended` — price, PE, vol, 52w                 |
| Analyst consensus  | `analyst_rating`, `analyst_count`, `analyst_target`     |
| Volume pressure    | `vol_pressure` — `BUYING / SELLING / NEUTRAL`           |
| News sentiment     | `NewsNode` — `POSITIVE / NEUTRAL / NEGATIVE` + catalyst |

**Parallelisation and caching** (in `DecisionNode`):

All candidates are dispatched concurrently using `ThreadPoolExecutor`, capped by `min(candidates, provider.max_concurrency, _MAX_LLM_WORKERS)`. For Ollama this resolves to 1 (sequential); for cloud providers it runs up to 5 in parallel. An **instance-level short-TTL cache** avoids redundant LLM round-trips:

- **Cache key:** `"{ticker}:{score}:{type}"`
- **TTL:** `_DECISION_CACHE_TTL_MINUTES` (default: 15 minutes)

**Output JSON (4 LLM-derived fields — `type` is injected from SignalEngine):**

```json
{
    "action":        "BUY" | "IGNORE",
    "confidence":    "high" | "moderate" | "low",
    "entry_quality": "strong" | "moderate" | "weak",
    "reason":        "one concise sentence referencing score tier and key signals"
}
```

| Field           | Values                           | Description                        |
| --------------- | -------------------------------- | ---------------------------------- |
| `action`        | `BUY` \| `IGNORE`                | The only two permitted outputs     |
| `confidence`    | `high` \| `moderate` \| `low`    | Conviction level                   |
| `entry_quality` | `strong` \| `moderate` \| `weak` | Entry risk/reward profile          |
| `reason`        | string                           | Factual one-sentence justification |

> `type` (`dip_buy` | `value` | `momentum`) is determined by SignalEngine — never by the LLM. This ensures reproducible, deterministic opportunity classification.

**Error handling:** Any JSON parse failure or LLM exception returns a safe `IGNORE / low / weak` fallback — the scan never crashes on a single ticker.

---

### 7. Sort, Cap, and Emit

**File:** `nodes/decision_node.py`

BUY decisions are sorted by two keys:

1. **Confidence rank** — `high` (0) → `moderate` (1) → `low` (2)
2. **Opportunity score descending** within the same confidence band

After sorting, the list is **capped at `_MAX_CONCURRENT_BUYS` (default: 3)** — only the highest-conviction setups are emitted per cycle.

---

## State Object

**File:** `state.py` — `OpportunityState` dataclass

All pipeline stages read from and write to a single shared state object.

| Field                   | Type              | Written by | Description                                                        |
| ----------------------- | ----------------- | ---------- | ------------------------------------------------------------------ |
| `watchlist`             | `List[str]`       | caller     | Input tickers for this scan cycle                                  |
| `market_data`           | `Dict[str, Dict]` | `scanner`  | Extended market data per ticker                                    |
| `prefiltered`           | `List[str]`       | `scanner`  | Tickers that passed PreFilterEngine                                |
| `signals`               | `Dict[str, Dict]` | `scanner`  | SignalEngine output per ticker (includes `opportunity_score`)      |
| `candidates`            | `List[str]`       | `scanner`  | Score ≥ 2 + cooldown clear (or freshness bypass)                   |
| `skipped_cooldown`      | `List[str]`       | `scanner`  | Tickers blocked by active cooldown this cycle                      |
| `recent_signal_context` | `Dict[str, Any]`  | `decision` | `{ticker: {price, score}}` at last BUY — used for freshness bypass |
| `news_sentiment`        | `Dict[str, Dict]` | `news`     | `{ticker: {sentiment, catalyst, headline_count}}`                  |
| `decisions`             | `Dict[str, Dict]` | `decision` | LLM output per candidate                                           |
| `buy_opportunities`     | `List[Dict]`      | `decision` | Final sorted BUY list (capped at `_MAX_CONCURRENT_BUYS`)           |
| `scan_errors`           | `Dict[str, str]`  | `scanner`  | Fetch failures: ticker → error message                             |
| `recent_signals`        | `Dict[str, str]`  | `decision` | Cooldown tracker: ticker → last BUY ISO UTC timestamp              |

### Notes on portfolio context

The opportunity agent is portfolio-agnostic. It does not accept or use any portfolio data. Portfolio constraint checking (position caps, sector caps, cash limits) is the responsibility of the caller or a higher-level supervisor agent.

---

## Output Format

The final `buy_opportunities` list is sorted by confidence then opportunity score descending. Each entry includes quantitative signals and news context:

```json
[
  {
    "ticker": "MSFT",
    "action": "BUY SIGNAL",
    "confidence": "high",
    "entry_quality": "moderate",
    "reason": "Score +3 with forward P/E below trailing, price near 52-week low, and analyst upside.",
    "type": "dip_buy",
    "score": 3,
    "opportunity_score": 0.6812,
    "signals": [
      "Forward P/E (19.0) < Trailing P/E (22.5) → earnings growth expected",
      "Price in lower 30% of 52w range (7%) → potential value entry",
      "Analyst consensus: buy (14 analysts, target $480, +22% upside)"
    ],
    "sector": "Technology",
    "news_sentiment": "positive",
    "news_catalyst": "Microsoft Azure revenue beat analyst estimates by 8%."
  }
]
```

### CLI output format

Each BUY signal is printed in three clearly separated sections:

```
============================================================
ALPHA SCANNER — SIGNALS DETECTED (1 found)
============================================================

============================================================
ALPHA SIGNAL — OPPORTUNITY DETECTED
============================================================

  Ticker        : MSFT
  Signal        : BUY SIGNAL
  Confidence    : high
  Entry Quality : strong
  Score         : +3
  Type          : dip_buy
  Sector        : Technology

  News          : POSITIVE  —  Azure revenue beat estimates by 8%.

  Quantitative Signals:
      - Forward P/E (19.0) < Trailing P/E (22.5) → earnings growth expected
      - Price in lower 30% of 52w range (7%) → potential value entry
      - Analyst consensus: buy (14 analysts, target $480, +22% upside)

  Reason:
  Score +3 (strong_buy tier): multiple confirming signals with analyst upside.

------------------------------------------------------------
EXECUTION STATUS
------------------------------------------------------------

  Actionable    : YES

  Suggested Actions:
      - Execute BUY for MSFT
      - Set stop-loss below 52-week low

============================================================
```

---

## Observability & Telemetry

**Provider:** Traceloop SDK via `src.observability.get_telemetry_logger()`

Seven telemetry events are emitted per scan cycle:

| Event                 | When emitted                      | Key payload fields                                                                                                                      |
| --------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `scan_start`          | Before any fetch                  | `watchlist_size`, `timestamp`                                                                                                           |
| `prefilter_complete`  | After PreFilterEngine             | `total`, `passed`, `filtered_out`, `latency_ms`                                                                                         |
| `signals_generated`   | After SignalEngine                | `scored`, `strong_buy_tier`, `buy_tier`, `neutral_tier`, `avoid_tier`, `latency_ms`                                                     |
| `candidates_filtered` | After candidate + cooldown filter | `candidate_count`, `filtered_count`, `skipped_cooldown`                                                                                 |
| `llm_decision`        | Per ticker, after LLM call        | `ticker`, `score`, `action`, `confidence`, `latency_ms`                                                                                 |
| `buy_signal_emitted`  | Per BUY result                    | `ticker`, `score`, `confidence`, `entry_quality`, `type`, `sector`                                                                      |
| `scan_summary`        | End of every scan cycle           | `watchlist_size`, `fetched`, `prefiltered`, `scored`, `candidates`, `approved`, `buy_opportunities`, `step_latencies_ms`, `buy_tickers` |

#### Per-step latency tracking

Every pipeline stage is timed with `time.monotonic()`. The `scan_summary` event contains `step_latencies_ms`:

```json
{
  "step_latencies_ms": {
    "fetch_ms": 423.0,
    "prefilter_ms": 1.2,
    "signal_ms": 0.8,
    "news_ms": 1250.0,
    "llm_total_ms": 3521.0,
    "total_ms": 5196.0
  }
}
```

Telemetry is non-blocking — a missing or disabled `TRACELOOP_API_KEY` downgrades to a no-op logger.

---

## Running the Agent

### Prerequisites

```bash
pip install yfinance langchain-core langgraph
# Plus one of the LLM provider packages:
pip install langchain-ollama    # for Ollama (default)
pip install langchain-openai    # for OpenAI
pip install langchain-google-genai  # for Google Gemini
```

---

### CLI — Single Scan

Run once and print results. Useful for testing outside market hours.

```bash
# Explicit watchlist
python -m src.agents.opportunity.workflow --tickers AAPL MSFT NVDA GOOGL META --once

# Dynamic universe — top 50 S&P 500 constituents (US, default)
python -m src.agents.opportunity.workflow --top-n 50 --once

# Verbose — per-ticker pipeline digest table after the scan
python -m src.agents.opportunity.workflow --top-n 50 --once --verbose

# Indian large cap — NIFTY 50
python -m src.agents.opportunity.workflow --top-n 50 --market IN --once

# Indian mid cap — NIFTY MIDCAP 100
python -m src.agents.opportunity.workflow --top-n 100 --market IN_MID --once

# Indian small cap — NIFTY SMALLCAP 100
python -m src.agents.opportunity.workflow --top-n 100 --market IN_SMALL --once

# Override LLM model (provider auto-inferred)
python -m src.agents.opportunity.workflow --top-n 100 --market IN_MID --model gpt-4o --once
```

Either `--tickers` or `--top-n` is required. `--market` defaults to `US`.

---

### CLI — Continuous Batch Scan

Runs automatically while the market is open. Scans every 15 minutes by default.

```bash
# US market, explicit watchlist, default interval (15 min)
python -m src.agents.opportunity.workflow --tickers AAPL MSFT NVDA GOOGL META AMZN TSLA

# US dynamic universe, every 5 minutes
python -m src.agents.opportunity.workflow --top-n 100 --interval 5

# Indian large cap (NIFTY 50), every 10 minutes
python -m src.agents.opportunity.workflow --top-n 50 --market IN --interval 10

# Indian mid cap (NIFTY MIDCAP 100), every 15 minutes
python -m src.agents.opportunity.workflow --top-n 100 --market IN_MID --interval 15

# Indian small cap (NIFTY SMALLCAP 100), every 15 minutes
python -m src.agents.opportunity.workflow --top-n 100 --market IN_SMALL --interval 15
```

The process exits automatically when the market closes. Cooldown state persists across cycles.

---

### Programmatic API

#### Single-shot

```python
from src.agents.opportunity.workflow import trigger_scan

opportunities = trigger_scan(
    tickers=["AAPL", "MSFT", "NVDA", "META", "GOOGL"],
)

for opp in opportunities:
    print(f"{opp['ticker']} — {opp['confidence']} | news: {opp.get('news_sentiment')}")
    print(f"  catalyst: {opp.get('news_catalyst')}")
```

#### Dynamic universe helper

```python
from src.agents.opportunity.markets.market_strategy import get_liquid_universe

us_tickers  = get_liquid_universe(n=100, market="US")       # top 100 S&P 500
in_tickers  = get_liquid_universe(n=50,  market="IN")       # NIFTY 50
mid_tickers = get_liquid_universe(n=100, market="IN_MID")   # NIFTY MIDCAP 100
sml_tickers = get_liquid_universe(n=100, market="IN_SMALL") # NIFTY SMALLCAP 100
```

#### Continuous batch loop

```python
from src.agents.opportunity.workflow import run_batch_scan

run_batch_scan(tickers=["AAPL", "MSFT", "NVDA", "META"], interval_minutes=15)
# Blocks until market closes
```

#### Market hours check

```python
from src.agents.opportunity.workflow import is_market_open

if is_market_open("US"):
    run_scan()
```

---

## Configuration

### Environment Variables

| Variable                  | Required    | Description                                                                    |
| ------------------------- | ----------- | ------------------------------------------------------------------------------ |
| `ALPHA_SCANNER_LLM_MODEL` | Recommended | Model for this agent. Provider auto-inferred. Overrides `PORTFOLIO_LLM_MODEL`. |
| `PORTFOLIO_LLM_MODEL`     | Optional    | Fallback model when `ALPHA_SCANNER_LLM_MODEL` is not set.                      |
| `PORTFOLIO_LLM_PROVIDER`  | Optional    | Fallback provider (`ollama` \| `openai` \| `google`). Default: `ollama`.       |
| `TRACELOOP_API_KEY`       | Optional    | Enables Traceloop telemetry. Scan runs without it.                             |
| `OLLAMA_API_KEY`          | Conditional | Required when using the Ollama provider.                                       |
| `OPENAI_API_KEY`          | Conditional | Required when using the OpenAI provider.                                       |
| `GOOGLE_API_KEY`          | Conditional | Required when using the Google provider.                                       |

---

### Supported Markets

| Code       | Index              | Exchange | Universe size | Hours (local)           |
| ---------- | ------------------ | -------- | ------------- | ----------------------- |
| `US`       | S&P 500            | NYSE     | up to 500     | Mon–Fri 09:30–16:00 ET  |
| `IN`       | NIFTY 50           | NSE      | 50            | Mon–Fri 09:15–15:30 IST |
| `IN_MID`   | NIFTY MIDCAP 100   | NSE      | 100           | Mon–Fri 09:15–15:30 IST |
| `IN_SMALL` | NIFTY SMALLCAP 100 | NSE      | 100           | Mon–Fri 09:15–15:30 IST |

All three Indian codes share the same NSE trading-hours check (`_NSEMarketStrategy` base). To add a new index, subclass `_NSEMarketStrategy` and add one entry to `_REGISTRY` in `markets/market_strategy.py`.

---

### Supported LLM Providers

| Provider key | Models                                                                         |
| ------------ | ------------------------------------------------------------------------------ |
| `ollama`     | `gpt-oss:120b`, `llama3`, `llama3:70b`, `mistral`, `mixtral`, `gemma2`, `phi3` |
| `openai`     | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`               |
| `google`     | `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro`, `gemini-2.0-flash`         |

Custom or fine-tuned models are accepted — an unknown name emits a warning but does not abort. Set `PORTFOLIO_LLM_PROVIDER` explicitly for custom models.

---

### Tuning Constants

**`nodes/alpha_scanner_agent.py`**

| Constant                   | Default     | Description                                                         |
| -------------------------- | ----------- | ------------------------------------------------------------------- |
| `_COOLDOWN_MINUTES`        | `30`        | Cooldown window length (unit determined by `_COOLDOWN_UNIT`)        |
| `_COOLDOWN_UNIT`           | `"minutes"` | Cooldown unit: `"minutes"` \| `"hours"` \| `"days"`                 |
| `_CANDIDATE_MIN_SCORE`     | `2`         | Minimum signal score required to advance to news + LLM decision     |
| `_MAX_FETCH_WORKERS`       | `10`        | Maximum parallel yfinance fetch threads                             |
| `_MAX_CONCURRENT_BUYS`     | `3`         | Max BUY signals emitted per scan cycle                              |
| `_MAX_SIGNAL_SCORE`        | `5.0`       | Theoretical max raw signal score (opportunity score normalization)  |
| `_EVENT_OVERRIDE_MIN_MOVE` | `8.0`       | Abs price move % that triggers event override (bypass score gate)   |
| `_EVENT_OVERRIDE_52W_BAND` | `0.30`      | Price must be in lower 30% of 52w range for event override to apply |

**`nodes/decision_node.py`**

| Constant                      | Default | Description                                            |
| ----------------------------- | ------- | ------------------------------------------------------ |
| `_MAX_LLM_WORKERS`            | `5`     | Maximum concurrent LLM decision threads per scan cycle |
| `_DECISION_CACHE_TTL_MINUTES` | `15`    | LLM response cache lifetime (key: `ticker:score:type`) |

**`nodes/news_node.py`**

| Constant            | Default | Description                              |
| ------------------- | ------- | ---------------------------------------- |
| `_MAX_HEADLINES`    | `5`     | Max headlines per ticker sent to the LLM |
| `_MAX_NEWS_WORKERS` | `3`     | Max parallel news LLM threads            |

**`engines/signal_engine.py` → `_WEIGHTS`**

| Key                | Default | Description                                          |
| ------------------ | ------- | ---------------------------------------------------- |
| `pe_improvement`   | `+1`    | Forward PE improving relative to trailing            |
| `52w_lower_band`   | `+1`    | Price in lower 30% of 52-week range                  |
| `analyst_bullish`  | `+1`    | Buy/strong_buy consensus (≥ 3 analysts, > 5% upside) |
| `buying_pressure`  | `+1`    | Price up on volume spike (≥ 1.5× avg)                |
| `near_52w_high`    | `−1`    | Near 52-week high — unfavourable entry               |
| `analyst_bearish`  | `−1`    | Underperform/sell consensus (≥ 3 analysts)           |
| `selling_pressure` | `−1`    | Price down on volume spike (≥ 1.5× avg)              |
| `volatility`       | `−2`    | Volatility penalty (cancels two bullish signals)     |

**`engines/signal_engine.py` → `_THRESHOLDS`**

| Key                     | Default | Description                                       |
| ----------------------- | ------- | ------------------------------------------------- |
| `volatility_penalty`    | `0.35`  | Annualised volatility above this → weight applied |
| `near_52w_high_pct`     | `0.05`  | Within this fraction of 52w high → −1 point       |
| `lower_30_band_pct`     | `0.30`  | Price in lower 30% of range → +1 point            |
| `min_analyst_count`     | `3`     | Minimum analysts required for consensus signals   |
| `analyst_target_upside` | `0.05`  | Min target upside (> 5%) for bullish signal       |
| `vol_spike_ratio`       | `1.5`   | Volume ratio threshold for pressure signals       |

**`engines/prefilter_engine.py` → `_PREFILTER_THRESHOLDS`**

| Key                  | Default | Description                                        |
| -------------------- | ------- | -------------------------------------------------- |
| `change_pct_abs`     | `3.0`   | Minimum abs daily price move to pass filter        |
| `lower_band_pct`     | `0.20`  | Price must be in lower 20% of 52w range            |
| `volume_spike_ratio` | `1.5`   | Volume must exceed 30-day average by this multiple |
| `volatility_spike`   | `0.35`  | Annualised volatility threshold                    |

---

## Real-Time & Event-Driven Behavior

### Batch Mode (default)

```
while market_open():
    tickers = get_watchlist()
    run_alpha_scan(tickers)
    sleep(interval_minutes * 60)
```

`recent_signals` bridges scan cycles so the cooldown is enforced **across** iterations.

### Event-Driven Mode

`trigger_scan()` is a stateless single-call function designed to be invoked by any external trigger:

```python
@app.post("/hooks/price-spike")
async def on_price_spike(event: PriceSpikeEvent):
    opportunities = trigger_scan(tickers=[event.ticker])
    if opportunities:
        notify_trader(opportunities)
```

Other trigger patterns: volatility alerts, earnings calendar, index rebalance events.

---

## Design Decisions

| Decision                                                 | Rationale                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3-node LangGraph graph (scanner → news → decision)**   | Each node has one responsibility. NewsNode can be disabled or swapped independently. DecisionNode isolation makes it easy to replace the LLM without touching fetch or signal logic.                                                                                                                                                                  |
| **NewsNode only for candidates**                         | Fetching headlines + running LLM for every ticker in a 200-stock universe adds ~200 LLM calls per scan. Running news only on candidates (score ≥ 1) keeps cost proportional to signal quality.                                                                                                                                                        |
| **Analyst signals in SignalEngine, not LLM prompt only** | Analyst consensus is a quantitative fact (rating + count + target). Encoding it as a scored signal (+1/−1) means it affects the tier and candidate filter deterministically, not just LLM verdict probability.                                                                                                                                        |
| **Vol pressure proxy from free data**                    | Level-2 order flow requires a paid data feed. Price direction × volume spike is a free, reproducible proxy for institutional buy/sell pressure available in every yfinance info call.                                                                                                                                                                 |
| **`engines/`, `nodes/`, `markets/` subfolders**          | Groups files by responsibility: engines are pure computation (no I/O, no graph nodes); nodes are LangGraph-runnable pipeline stages; markets isolate exchange-specific schedules and universes.                                                                                                                                                       |
| **PreFilterEngine before SignalEngine**                  | At 500+ tickers, running 8 signals and LLM on every ticker is expensive. The prefilter drops uninteresting tickers with zero scoring cost.                                                                                                                                                                                                            |
| **Volatility weight = −2**                               | A weight of −1 let `pe_improvement (+1) + 52w_lower_band (+1) + volatility (−1) = +1` pass. At −2 the net is 0 (neutral) and the ticker does not advance.                                                                                                                                                                                             |
| **Type from SignalEngine, not LLM**                      | `dip_buy / value / momentum` is inferred from which fundamental signals fired. Deterministic inference is reproducible across providers and immune to prompt drift.                                                                                                                                                                                   |
| **Warnings, not filters for cap breaches**               | Removed — the opportunity agent has no portfolio context. All signals are emitted unconditionally; portfolio constraint checking is the caller's responsibility.                                                                                                                                                                                      |
| **Auto-fetch via PortfolioAgent pipeline**               | Removed — the opportunity agent is now fully portfolio-agnostic. It does not import or call any portfolio subagents.                                                                                                                                                                                                                                  |
| **LLM parallelisation + short-TTL cache**                | 5 concurrent decision threads reduce wall-clock time from N×3.5s to ~3.5s. The 15-min cache prevents redundant API calls for the same ticker in the same signal state across repeated intraday scans.                                                                                                                                                 |
| **`zoneinfo` over `pytz`**                               | `zoneinfo` is Python 3.9+ stdlib — zero extra dependency, identical capability for NYSE/NSE hours checking.                                                                                                                                                                                                                                           |
| **`OllamaProvider.max_concurrency = 1`**                 | Ollama returns HTTP 429 when multiple requests arrive simultaneously. All LLM call sites — `DecisionNode`, `NewsNode`, `OpportunityDecisionAgent` — cap workers via `min(candidates, provider.max_concurrency, _MAX_LLM_WORKERS)`. A secondary 3-attempt exponential-backoff retry (2 s → 4 s) handles any 429 that slips through on cloud providers. |

---

## Integration with PortfolioAgent

`AlphaScannerAgent` and `PortfolioAgent` are complementary, fully independent agents:

| Concern                                         | Agent                                    |
| ----------------------------------------------- | ---------------------------------------- |
| Scan for new BUY entries                        | **AlphaScannerAgent**                    |
| Manage existing holdings (HOLD / REDUCE / EXIT) | **PortfolioAgent**                       |
| Risk calculation for existing portfolio         | **RiskAgent** (via PortfolioAgent graph) |

They share no code, no imports, and no runtime coupling. You can run either agent standalone. If you want to combine insights from both (e.g., skip BUY signals for positions already above a size cap), that orchestration belongs in a higher-level supervisor agent — not inside either individual agent.
