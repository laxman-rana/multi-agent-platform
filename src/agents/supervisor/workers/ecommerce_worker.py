"""
ecommerce_worker.py
-------------------
Supervisor worker adapter over the Ecommerce support agent.

Wraps src.agents.ecommerce.registry.get_agent_graph() — no changes to
the underlying agent.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.ecommerce.registry import get_agent_graph

from . import BaseWorker, WorkerName, worker

logger = logging.getLogger(__name__)


class EcommerceInput(BaseModel):
    customer_message: str = Field(
        ...,
        description="The customer's support message, complaint, or question about their order.",
    )
    order_id: str = Field(
        ...,
        description="The order ID referenced by the customer.",
    )
    order_status: str = Field(
        "pending",
        description=(
            "Current status of the order if known, e.g. 'pending', 'shipped', "
            "'delivered', 'damaged', 'cancelled'."
        ),
    )
    order_amount: float = Field(
        0.0,
        description="Order total amount in USD if known.",
    )


@worker
class EcommerceWorker(BaseWorker):
    """Handles customer support queries for e-commerce orders."""

    name = WorkerName.ECOMMERCE
    description = (
        "Handle customer support requests for e-commerce orders. Use this when the user "
        "mentions an order issue such as a missing package, damaged item, refund request, "
        "order cancellation, shipping address change, or any other post-purchase support matter. "
        "Requires an order ID and a description of the customer's issue."
    )
    input_schema = EcommerceInput

    def invoke(
        self,
        customer_message: str,
        order_id: str,
        order_status: str = "pending",
        order_amount: float = 0.0,
    ) -> str:
        logger.info(
            "[EcommerceWorker] Handling support — order_id=%s status=%s",
            order_id,
            order_status,
        )
        order = {
            "order_id": order_id,
            "status": order_status,
            "total": order_amount,
        }
        graph = get_agent_graph("support")
        initial_state = {
            "order": order,
            "messages": [HumanMessage(content=customer_message)],
        }
        result = graph.invoke(initial_state)
        messages = result.get("messages", [])
        # Return the last non-tool AIMessage content as the result
        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", None)
            ):
                return str(msg.content)
        return "Support request processed."



