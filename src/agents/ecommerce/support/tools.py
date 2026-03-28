from langchain.tools import tool
from traceloop.sdk.decorators import workflow


@tool
@workflow(name="send_customer_message")
def send_customer_message(order_id: str, text: str) -> str:
    """Send a plain response to the customer."""

    print(f"[TOOL] send_customer_message(order_id={order_id}) -> {text}")
    return "sent"


@tool
@workflow(name="issue_refund")
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for the given order."""

    print(f"[TOOL] issue_refund(order_id={order_id}, amount={amount})")
    return "refund_queued"


@tool
@workflow(name="cancel_order")
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""

    print(f"[TOOL] cancel_order(order_id={order_id})")
    return "cancelled"


@tool
@workflow(name="update_address_for_order")
def update_address_for_order(order_id: str, shipping_address: dict) -> str:
    """Change the shipping address for a pending order."""

    print(f"[TOOL] update_address_for_order(order_id={order_id}, address={shipping_address})")
    return "address_updated"


TOOLS = [send_customer_message, issue_refund, cancel_order, update_address_for_order]
