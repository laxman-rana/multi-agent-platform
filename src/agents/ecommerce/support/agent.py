import json
import argparse
import logging
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph

from src.llm import get_llm
from src.memory.config import load_config
from src.memory.factory import create_memory_provider
from src.observability import get_telemetry_logger

from .tools import TOOLS
from .types import AgentState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_agent_llm():
    """Build the configured LLM lazily so module import stays safe.

    Provider and model are driven by env vars:
      PORTFOLIO_LLM_PROVIDER  — ollama | openai | google  (default: ollama)
      PORTFOLIO_LLM_MODEL     — model name override
    """
    import os as _os
    provider = _os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama").lower()
    return get_llm(
        model_name=provider,
        tools=TOOLS,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


@lru_cache(maxsize=1)
def get_memory():
    """Return a mem0 cloud provider for the ecommerce support agent."""
    return create_memory_provider(config=load_config())


def clear_cached_dependencies() -> None:
    """Clear cached singleton-style dependencies for a fresh runtime state."""

    get_agent_llm.cache_clear()
    get_memory.cache_clear()
    get_telemetry_logger.cache_clear()


def invoke_model(state: AgentState) -> AgentState:
    """Process customer messages, inspect order details, and complete the workflow via tools."""

    history = state.get("messages", [])
    order = state.get("order", {})
    llm = get_agent_llm()
    memory = get_memory()
    telemetry = get_telemetry_logger()

    telemetry.log_event(
        "agent_invoked",
        {
            "order_id": order.get("order_id", "unknown"),
            "message_count": len(history),
            "order_status": order.get("status", "unknown"),
        },
    )

    if not order:
        error_msg = "No order information provided. Please provide order details to assist the customer."
        telemetry.log_event("agent_error", {"error": "missing_order_data"})
        return {"messages": history + [SystemMessage(content=error_msg)]}

    order_id = order.get("order_id", "unknown")

    # Double-refund guard: search for prior refund memories for this order.
    # Uses search() (the canonical mem0 retrieval path) rather than get_all()
    # with v2 AND-filters, which can miss records written via the v1 add endpoint.
    search_results = memory.search(
        query=f"refund order {order_id}",
        user_id=order_id,
        agent_id="ecommerce_support",
        limit=10,
    )
    if search_results:
        logger.info(
            "[EcommerceAgent] Retrieved %d memory record(s) for order %s: %s",
            len(search_results),
            order_id,
            [m.get("memory", "") for m in search_results],
        )
    else:
        logger.info("[EcommerceAgent] No prior memory found for order %s", order_id)
    refund_already_issued = any(
        "refund" in m.get("memory", "").lower()
        for m in search_results
    )
    if refund_already_issued:
        telemetry.log_event(
            "double_refund_attempt",
            {"order_id": order_id, "prior_memory_count": len(search_results)},
        )
        logger.info(
            "[EcommerceAgent] Prior refund found for order %s — passing to LLM as constraint",
            order_id,
        )

    refund_issued_this_turn: bool = False
    refund_guard = (
        "\n## IMPORTANT: Refund Already Issued\n"
        f"A refund for order {order_id} has already been processed.\n"
        "- Do NOT call issue_refund under any circumstances.\n"
        "- Politely inform the customer and offer to escalate to the support team if needed.\n"
    ) if refund_already_issued else ""

    prompt = f"""
            You are a professional customer support assistant for an e-commerce platform.

            ## Role
            - Help customers resolve issues related to their orders.
            - Be polite, concise, and empathetic.
            - Always prioritize customer satisfaction while following business rules.

            ## Available Tools
            - get_order_details(order_id)           — ALWAYS call this first to retrieve order data.
            - issue_refund(order_id, amount)        — use the `total` field from get_order_details as amount.
            - send_customer_message(order_id, text) — ALWAYS call this last to reply to the customer.
            - cancel_order(order_id)
            - update_address_for_order(order_id, shipping_address)

            ## Workflow
            1. Call get_order_details to fetch the full order.
            2. Decide the right action based on the order data and the customer's issue.
            3. Execute the action (e.g. issue_refund) if appropriate.
            4. Call send_customer_message with the final reply — this ends the conversation.
            5. Do NOT call any more tools or produce any more output after send_customer_message.

            ## Rules
            - Do NOT ask the customer for information already available in the order data.
            - Do NOT issue refunds if `refund_eligible` is false.
            - Do NOT hallucinate order details — use only what get_order_details returns.
            - The refund amount is always the `total` field from the order — never ask the customer.
            {refund_guard}
            ## Order ID
            {order_id}
            """

    messages = [SystemMessage(content=prompt)] + list(history)
    tool_calls_count = 0
    user_message = next((m.content for m in history if isinstance(m, HumanMessage)), "")

    while True:
        ai_msg: AIMessage = llm.invoke(messages)
        messages.append(ai_msg)
        telemetry.log_llm_interaction(
            prompt=user_message or "No user message",
            response=str(ai_msg.content),
        )

        if not getattr(ai_msg, "tool_calls", None):
            break

        for tool_call in ai_msg.tool_calls:
            logger.info("[EcommerceAgent] Tool called: %s | args: %s", tool_call["name"], tool_call["args"])
            tool_calls_count += 1

            if tool_call["name"] == "issue_refund":
                if refund_already_issued:
                    # LLM ignored the constraint — block at execution level as last resort
                    logger.warning(
                        "[EcommerceAgent] LLM attempted issue_refund despite prior refund guard — blocked"
                    )
                    messages.append(
                        ToolMessage(content="error: refund already issued for this order", tool_call_id=tool_call["id"])
                    )
                    continue
                refund_issued_this_turn = True

            tool_func = next(tool for tool in TOOLS if tool.name == tool_call["name"])
            result = tool_func.invoke(tool_call["args"])

            telemetry.log_tool_usage(
                tool_name=tool_call["name"],
                input_data=tool_call["args"],
                output_data={"result": str(result)},
            )

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

            if tool_call["name"] == "send_customer_message":
                break

        # Stop the agentic loop once the customer has been replied to.
        if any(tc["name"] == "send_customer_message" for tc in ai_msg.tool_calls):
            break

    # ------------------------------------------------------------------ #
    # Persist refund event so future requests can detect it               #
    # ------------------------------------------------------------------ #
    if refund_issued_this_turn:
        memory.add(
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"Refund issued for order {order_id}."},
            ],
            user_id=order_id,
            agent_id="ecommerce_support",
            metadata={"event": "refund_issued", "order_id": order_id},
        )
        logger.info("[EcommerceAgent] Refund memory stored for order %s", order_id)

    telemetry.log_event(
        "agent_completed",
        {
            "order_id": order_id,
            "tools_used": tool_calls_count,
            "total_messages": len(messages),
        },
    )

    return {"messages": messages}


def construct_graph():
    graph = StateGraph(AgentState)
    graph.add_node("assistant", invoke_model)
    graph.set_entry_point("assistant")
    return graph.compile()


def main(clear_cache: bool = False, order_id: str | None = None, message: str | None = None, order_fields: dict | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if clear_cache:
        clear_cached_dependencies()

    graph = construct_graph()

    # ------------------------------------------------------------------ #
    # Custom invocation from CLI                                           #
    # ------------------------------------------------------------------ #
    if order_id and message:
        order: dict = {"order_id": order_id}
        if order_fields:
            order.update(order_fields)
        logger.info("=" * 60)
        logger.info("Custom invocation — order: %s", order_id)
        logger.info("=" * 60)
        result = graph.invoke({"order": order, "messages": [HumanMessage(content=message)]})
        for msg in result["messages"]:
            if msg.type == "ai" and not msg.content and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc["name"] == "send_customer_message":
                        logger.info("ai → customer: %s", tc["args"].get("text", ""))
                    else:
                        logger.info("ai: [tool call: %s | args: %s]", tc["name"], tc["args"])
            else:
                logger.info("%s: %s", msg.type, msg.content)
        return

    scenarios = [
        {
            "name": "Delivered damaged item",
            "order": {
                "order_id": "A12345",
                "status": "Delivered",
                "total": 19.99,
                "item": "Glass Water Bottle",
                "delivery_date": "2026-03-22",
                "refund_eligible": True,
            },
            "messages": [
                HumanMessage(
                    content="My glass water bottle arrived shattered. Can I get a refund?"
                )
            ],
        },
        {
            "name": "Package marked delivered but missing",
            "order": {
                "order_id": "B77890",
                "status": "Delivered",
                "total": 64.50,
                "item": "Wireless Keyboard",
                "delivery_date": "2026-03-27",
                "refund_eligible": True,
            },
            "messages": [
                HumanMessage(
                    content=(
                        "Tracking says delivered yesterday, but I never received the package. "
                        "Please help."
                    )
                )
            ],
        },
        {
            "name": "Late return request",
            "order": {
                "order_id": "C55671",
                "status": "Delivered",
                "total": 129.99,
                "item": "Noise Cancelling Headphones",
                "delivery_date": "2025-12-20",
                "refund_eligible": False,
            },
            "messages": [
                HumanMessage(
                    content=(
                        "I want to return these headphones now, but I bought them over 3 months ago. "
                        "Can you still refund me?"
                    )
                )
            ],
        },
    ]

    for idx, scenario in enumerate(scenarios, start=1):
        logger.info("=" * 60)
        logger.info("Scenario %d: %s", idx, scenario["name"])
        logger.info("=" * 60)

        result = graph.invoke(
            {
                "order": scenario["order"],
                "messages": scenario["messages"],
            }
        )

        for message in result["messages"]:
            logger.info("%s: %s", message.type, message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ecommerce support agent.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run built-in demo scenarios:\n"
            "  python -m src.agents.ecommerce.support.agent\n\n"
            "  # Send a custom message:\n"
            "  python -m src.agents.ecommerce.support.agent \\\n"
            "      --order-id A12345 \\\n"
            "      --message \"My bottle arrived broken, I want a refund.\" \\\n"
            "      --order '{\"status\":\"Delivered\",\"total\":19.99,\"item\":\"Glass Water Bottle\",\"refund_eligible\":true}'\n"
        ),
    )
    parser.add_argument(
        "-e",
        "--evict-cache",
        action="store_true",
        help="Clear cached LLM and logger instances before running.",
    )
    parser.add_argument(
        "--order-id",
        metavar="ORDER_ID",
        help="Order ID to pass to the agent (required with --message).",
    )
    parser.add_argument(
        "--message",
        metavar="TEXT",
        help="Customer message to send to the agent.",
    )
    parser.add_argument(
        "--order",
        metavar="JSON",
        help="Extra order fields as a JSON object (merged with --order-id).",
    )
    args = parser.parse_args()

    extra_fields: dict | None = None
    if args.order:
        try:
            extra_fields = json.loads(args.order)
        except json.JSONDecodeError as exc:
            parser.error(f"--order is not valid JSON: {exc}")

    if bool(args.order_id) != bool(args.message):
        parser.error("--order-id and --message must be used together.")

    main(
        clear_cache=args.evict_cache,
        order_id=args.order_id,
        message=args.message,
        order_fields=extra_fields,
    )
