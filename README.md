# Agents Project

> ⚠️ **EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE**
>
> This project is built strictly for learning, research, and demonstration purposes.
> Nothing in this repository constitutes financial advice, investment advice, trading
> recommendations, or any other type of professional advice. The buy/sell signals,
> scores, and opportunity outputs produced by these agents are **not** recommendations
> to buy or sell any financial instrument.
>
> **Do not use this software to make real investment or trading decisions.**
> Past signal quality is not indicative of future results. Always consult a qualified
> financial adviser before making any investment decision.
>
> The authors and contributors of this project accept **no liability** for any financial
> loss or damage arising from the use, misuse, or inability to use this software.

---

This repository hosts multiple AI agents across different business domains. It provides a shared Python package layout for agent implementations, LLM providers, observability integrations, and agent-specific tools.

Three agents are currently implemented:

- **E-commerce support agent** — LangGraph-based customer support workflow with tool-calling (refunds, messaging)
- **Portfolio analysis agent** — LangGraph multi-agent pipeline that analyses an investor's equity portfolio using live market data and an LLM to produce hold/reduce/exit/double-down recommendations with actionable allocation changes
- **Opportunity scanner agent** — 3-node LangGraph pipeline that scans live equity markets (US S&P 500, NIFTY 50/MIDCAP 100/SMALLCAP 100) for high-quality BUY signals using a composite opportunity score (signal strength, analyst upside, volume spike, idea freshness), news sentiment, and LLM verdict

The goal is to use the same shared foundation for additional agents such as fulfillment, finance, operations, HR, or other domain-specific assistants.

## Project Overview

The repository is organized around two layers:

- shared infrastructure used by all agents
- domain-specific agent packages under `src/agents`

Shared infrastructure currently includes:

- LLM provider strategy implementations
- observability abstractions and integrations
- package-level conventions for agent modules and tool registration

Each agent package can define its own:

- orchestration logic
- prompt rules
- tools
- state types
- domain-specific workflows

## Repository Layout

- `src/llm/providers.py`: LLM provider strategy and factory (`get_llm()`)
- `src/llm/__init__.py`: shared LLM exports
- `src/observability/__init__.py`: shared `get_telemetry_logger()` singleton
- `src/observability/base.py`: observability abstraction
- `src/observability/traceloop_logger.py`: TraceLoop implementation
- `src/agents/ecommerce/support/agent.py`: e-commerce support agent (LangGraph + tool-calling)
- `src/agents/ecommerce/support/tools.py`: refund and messaging tools
- `src/agents/ecommerce/support/types.py`: state types
- `src/agents/portfolio/workflow.py`: portfolio analysis entry point (LangGraph `StateGraph`)
- `src/agents/portfolio/subagents/`: 7 pipeline nodes (portfolio, risk, market, news, decision, critic, formatter)
- `src/agents/portfolio/tools/`: live data tools (yfinance, VADER, news score), scoring, rebalance logic, and mock positions
- `src/agents/portfolio/state/`: `PortfolioState` dataclass
- `src/agents/opportunity/workflow.py`: opportunity scanner entry point (LangGraph 3-node graph + CLI)
- `src/agents/opportunity/engines/`: PreFilterEngine, SignalEngine (8 signals), OpportunityDecisionAgent
- `src/agents/opportunity/nodes/`: AlphaScannerAgent, NewsNode, DecisionNode
- `src/agents/opportunity/markets/`: MarketStrategy (US, IN, IN_MID, IN_SMALL)
- `src/agents/opportunity/state.py`: `OpportunityState` dataclass
- `src/requirements.txt`: Python dependencies

## Quick Start

```powershell
# ── E-commerce support ────────────────────────────────────────────────────
python -m src.agents.ecommerce.support.agent

# ── Portfolio analysis ────────────────────────────────────────────────────
python -m src.agents.portfolio.workflow                        # default (Ollama)
python -m src.agents.portfolio.workflow --no-news              # skip NewsAgent
python -m src.agents.portfolio.workflow --model gpt-4o         # OpenAI
python -m src.agents.portfolio.workflow --model gemini-1.5-pro # Google

# ── Opportunity scanner — single scan ─────────────────────────────────────
python -m src.agents.opportunity.workflow --top-n 50 --once            # US S&P 500 top 50
python -m src.agents.opportunity.workflow --tickers AAPL MSFT NVDA --once  # explicit tickers
python -m src.agents.opportunity.workflow --top-n 50 --market IN --once    # NIFTY 50
python -m src.agents.opportunity.workflow --top-n 100 --market IN_MID --once   # NIFTY MIDCAP 100
python -m src.agents.opportunity.workflow --top-n 100 --market IN_SMALL --once # NIFTY SMALLCAP 100

# ── Opportunity scanner — continuous (runs while market is open) ───────────
python -m src.agents.opportunity.workflow --top-n 100                  # every 15 min
python -m src.agents.opportunity.workflow --top-n 100 --interval 5     # every 5 min
python -m src.agents.opportunity.workflow --top-n 50 --market IN --interval 10
```

All agents accept `--model <name>` to switch the LLM — the provider is inferred automatically from the model name.

---

## Running the Agents

### E-commerce support agent

```powershell
python -m src.agents.ecommerce.support.agent
```

### Portfolio analysis agent

```powershell
# With live news routing (default)
python -m src.agents.portfolio.workflow

# Skip NewsAgent
python -m src.agents.portfolio.workflow --no-news

# Select model — provider is inferred automatically
python -m src.agents.portfolio.workflow --model gpt-4-turbo
python -m src.agents.portfolio.workflow --model gemini-pro
python -m src.agents.portfolio.workflow --model llama3
```

See [src/agents/portfolio/README.md](src/agents/portfolio/README.md) for full documentation.

### Opportunity scanner agent

| Flag             | Type    | Description                                                             |
| ---------------- | ------- | ----------------------------------------------------------------------- |
| `--tickers`      | `str …` | Explicit list of ticker symbols to scan                                 |
| `--top-n`        | `int`   | Scan top-N most liquid tickers from the built-in universe               |
| `--market`       | `str`   | Market universe: `US` (default), `IN`, `IN_MID`, `IN_SMALL`             |
| `--interval`     | `int`   | Minutes between scans in continuous mode (default: 15)                  |
| `--once`         | flag    | Run a single scan and exit instead of looping                           |
| `--model`        | `str`   | Override the LLM model (provider auto-inferred)                         |
| `--verbose`      | flag    | Print per-ticker pipeline digest table after the scan                   |
| `--enforce-cash` | flag    | Block all signals when portfolio cash ≤ 0 (default: off — always scans) |

Either `--tickers` or `--top-n` is required. `--once` is recommended outside market hours.

```powershell
# Single scan — US large cap
python -m src.agents.opportunity.workflow --top-n 50 --once

# Single scan — Indian mid cap, OpenAI model
python -m src.agents.opportunity.workflow --top-n 100 --market IN_MID --model gpt-4o --once

# Continuous scan — US, every 10 minutes
python -m src.agents.opportunity.workflow --top-n 200 --interval 10
```

See [src/agents/opportunity/README.md](src/agents/opportunity/README.md) for full documentation.

## Technologies Used

- Python
- LangChain
- LangGraph
- LangChain Ollama integration
- LangChain OpenAI integration
- LangChain Google Generative AI integration
- TraceLoop SDK
- feedparser (RSS news fallback)

## Current Agent Example

The current implemented example is the e-commerce customer-support agent.

It exposes these tools to the model:

- `send_customer_message(order_id, text)`
  Sends the final customer-facing response.
- `issue_refund(order_id, amount)`
  Queues a refund for the order.
- `cancel_order(order_id)`
  Cancels an order that has not shipped.
- `update_address_for_order(order_id, shipping_address)`
  Updates the shipping address for a pending order.

## Setup

Create and activate a virtual environment, then install dependencies.

### PowerShell

```powershell
cd D:\projects\agents
.\.venv\Scripts\Activate.ps1
pip install -r src\requirements.txt
```

## Environment Variables

Set only the variables required for the provider you plan to use.

- `OLLAMA_API_KEY`: required for the configured default Ollama provider in `src/llm/providers.py`
- `OPENAI_API_KEY`: required if you switch the model provider to OpenAI
- `GOOGLE_API_KEY`: required if you switch the model provider to Google Gemini
- `TRACELOOP_API_KEY`: optional, enables TraceLoop telemetry when set

Example in PowerShell:

```powershell
$env:OLLAMA_API_KEY = "your-ollama-api-key"
$env:TRACELOOP_API_KEY = "your-traceloop-api-key"
```

## How To Run

Run an agent from the repository root as a module:

```powershell
cd D:\projects\agents
python -m src.agents.ecommerce.support.agent
```

Do not run agent files directly. Agents are part of the `src` package and should be executed with `python -m`.

You can also resolve ecommerce agents by name through the registry module:

```python
from src.agents.ecommerce import get_agent_graph, list_agents

print(list_agents())
graph = get_agent_graph("support")
```

## Runtime Notes

- LLM initialization is lazy, so importing the module does not immediately require provider credentials.
- TraceLoop telemetry is optional. If `TRACELOOP_API_KEY` is not set, telemetry stays disabled and the agent still runs.
- The default provider is currently `ollama` in `src/agents/ecommerce/support/agent.py`.

## Extending The Project

### Add a new agent domain

Create a new package under `src/agents`, keep its tools and types local to that package, and expose a module entrypoint that can be run with `python -m`.

### Add a new ecommerce agent

Create a package under `src/agents/ecommerce`, then register it in `src/agents/ecommerce/registry.py`.

Example:

- create `src/agents/ecommerce/returns/agent.py`
- create `src/agents/ecommerce/returns/tools.py`
- create `src/agents/ecommerce/returns/types.py`
- add `"returns": "src.agents.ecommerce.returns.agent"` to `AGENT_REGISTRY`

### Add a new LLM provider

Implement a new provider class in `src/llm/providers.py` and register it in `PROVIDERS`.

### Add a new observability provider

Implement the `TelemetryLogger` interface from `src/observability/base.py` and instantiate that strategy in the agent.

### Add a new business tool

Add a new tool function in the target agent package, for example `src/agents/ecommerce/support/tools.py`, and include it in `TOOLS`.
