# Agents Project

This repository is intended to host multiple AI agents across different business domains. It provides a shared Python package layout for agent implementations, LLM providers, telemetry integrations, and agent-specific tools.

The current implemented example is an e-commerce customer-support agent, but the repository is not limited to e-commerce. The goal is to use the same foundation for additional agents such as fulfillment, finance, operations, HR, internal copilots, or other domain-specific assistants.

## Project Overview

The repository is organized around two layers:

- shared infrastructure used by all agents
- domain-specific agent packages under `src/agents`

Shared infrastructure currently includes:

- LLM provider strategy implementations
- telemetry abstractions and integrations
- package-level conventions for agent modules and tool registration

Each agent package can define its own:

- orchestration logic
- prompt rules
- tools
- state types
- domain-specific workflows

## Repository Layout

- `src/llm/providers.py`: LLM provider strategy implementations and provider factory
- `src/llm/__init__.py`: shared LLM exports
- `src/telemetry/base.py`: telemetry abstraction
- `src/telemetry/traceloop_logger.py`: TraceLoop telemetry implementation
- `src/agents/`: home for all domain-specific agents
- `src/agents/ecommerce/registry.py`: registry for ecommerce agent modules
- `src/agents/ecommerce/support/agent.py`: ecommerce support agent implementation
- `src/agents/ecommerce/support/tools.py`: tools for the ecommerce support agent
- `src/agents/ecommerce/support/types.py`: state types for the ecommerce support agent
- `src/agents/ecommerce_customer_support/`: backward-compatible shim package for the old path
- `src/requirements.txt`: Python dependencies

## Recommended Structure For Multiple Agents

The current structure is generally correct for a multi-agent repository.

Recommended convention:

```text
src/
  llm/
    providers.py
  agents/
    ecommerce/
      registry.py
      support/
        agent.py
        tools.py
        types.py
      returns/
        agent.py
        tools.py
        types.py
      fraud/
        agent.py
        tools.py
        types.py
    finance/
      agent.py
      tools.py
      types.py
    hr/
      agent.py
      tools.py
      types.py
  telemetry/
    base.py
    traceloop_logger.py
```

This layout works well if each agent remains self-contained and shared concerns stay outside agent folders.

## Structure Assessment

The current project structure is a good starting point, with a few important notes.

What is already correct:

- shared LLM code is outside agent packages
- shared telemetry code is outside agent packages
- agent-specific tools and types live with the agent that owns them
- `src/agents` is the right top-level location for multiple domain agents

What should be kept consistent as the repo grows:

- use snake_case folder names for all agent packages
- keep each agent in its own package under `src/agents`
- keep shared integrations out of agent-specific directories
- run agents as Python modules from the repository root

What will likely need improvement later:

- if many agents share common prompting, execution helpers, or graph utilities, add a shared package such as `src/agents/common` instead of duplicating logic

What has already been standardized in this repository:

- shared LLM code now lives under `src/llm`
- ecommerce agents now live under a domain package at `src/agents/ecommerce`
- the current agent entrypoint uses `agent.py`
- tools and types live directly inside each agent package instead of a nested `config` package

One cleanup item outside the Python package structure:

- the top-level `ecommerce-agent/` folder is currently empty and does not appear to be part of the package layout; it should either be removed or repurposed to avoid confusion

## Technologies Used

- Python
- LangChain
- LangGraph
- LangChain Ollama integration
- LangChain OpenAI integration
- LangChain Google Generative AI integration
- TraceLoop SDK

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

### Add a new telemetry provider

Implement the `TelemetryLogger` interface from `src/telemetry/base.py` and instantiate that strategy in the agent.

### Add a new business tool

Add a new tool function in the target agent package, for example `src/agents/ecommerce/support/tools.py`, and include it in `TOOLS`.
