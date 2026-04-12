import logging

from langchain.tools import tool
from traceloop.sdk.decorators import workflow

from .mock_data import ORDERS

logger = logging.getLogger(__name__)


@tool
def get_order_details(order_id: str) -> dict:
    """Retrieve full order details for a given order ID."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order '{order_id}' not found."}
    logger.info("[Tool] get_order_details | order_id=%s | result=%s", order_id, order)
    return order


@tool
@workflow(name="send_customer_message")
def send_customer_message(order_id: str, text: str) -> str:
    """Send a plain response to the customer."""

    logger.info("[Tool] send_customer_message | order_id=%s | text=%s", order_id, text)
    return "sent"


@tool
@workflow(name="issue_refund")
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for the given order."""

    logger.info("[Tool] issue_refund | order_id=%s | amount=%s", order_id, amount)
    return "refund_queued"


@tool
@workflow(name="cancel_order")
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""

    logger.info("[Tool] cancel_order | order_id=%s", order_id)
    return "cancelled"


@tool
@workflow(name="update_address_for_order")
def update_address_for_order(order_id: str, shipping_address: dict) -> str:
    """Change the shipping address for a pending order."""

    logger.info("[Tool] update_address_for_order | order_id=%s | address=%s", order_id, shipping_address)
    return "address_updated"


TOOLS = [get_order_details, send_customer_message, issue_refund, cancel_order, update_address_for_order]
