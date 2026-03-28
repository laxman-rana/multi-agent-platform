import json
from functools import lru_cache

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph

from src.llm import get_llm
from src.telemetry.traceloop_logger import TraceLoopLogger

from .tools import TOOLS
from .types import AgentState


@lru_cache(maxsize=1)
def get_agent_llm():
    """Build the configured LLM lazily so module import stays safe."""

    return get_llm(
        model_name="ollama",
        tools=TOOLS,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


@lru_cache(maxsize=1)
def get_telemetry_logger() -> TraceLoopLogger:
    """Build the telemetry logger lazily so env issues do not break import."""

    return TraceLoopLogger()


def invoke_model(state: AgentState) -> AgentState:
    """Process customer messages, inspect order details, and complete the workflow via tools."""

    history = state.get("messages", [])
    order = state.get("order", {})
    llm = get_agent_llm()
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

    order_json = json.dumps(order, ensure_ascii=False)
    prompt = f"""
            You are a professional customer support assistant for an e-commerce platform.

            ## Role
            - Help customers resolve issues related to their orders.
            - Be polite, concise, and empathetic.
            - Always prioritize customer satisfaction while following business rules.

            ## Context
            ORDER:
            {order_json}

            ## Available Tools
            You have access to business tools such as:
            - issue_refund(order_id, amount)
            - send_customer_message(order_id, text)

            ## Instructions

            ### 1. Decision Making
            - If the issue is valid (e.g., damaged item), take appropriate action (e.g., refund).
            - Do NOT ask unnecessary questions if enough information is already available.
            - If information is missing, ask the customer for clarification instead of acting.

            ### 2. Tool Usage Rules
            - ALWAYS call required business tools before responding to the user.
            - After performing backend actions (e.g., refund), you MUST call `send_customer_message`.
            - `send_customer_message` should contain the final user-facing response.

            ### 3. Response Rules
            - Do NOT return a final answer directly to the user.
            - The final user communication MUST be sent via `send_customer_message`.
            - Keep messages clear, friendly, and professional.

            ### 4. Tone Guidelines
            - Apologize when appropriate.
            - Be reassuring and transparent.
            - Avoid technical jargon.

            ### 5. Safety Rules
            - Do NOT issue refunds if the order is not eligible.
            - Do NOT hallucinate order details.
            - Use only the provided ORDER data.

            ## Goal
            Resolve the user's issue completely using tools and provide a clear confirmation message.
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
            print(f"AI called tool: {tool_call['name']}")
            tool_calls_count += 1

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

    telemetry.log_event(
        "agent_completed",
        {
            "order_id": order.get("order_id", "unknown"),
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


def main():
    graph = construct_graph()
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
        print("\n" + "=" * 80)
        print(f"Scenario {idx}: {scenario['name']}")
        print("=" * 80)

        result = graph.invoke(
            {
                "order": scenario["order"],
                "messages": scenario["messages"],
            }
        )

        for message in result["messages"]:
            print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
