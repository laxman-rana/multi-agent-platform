"""Mock order database for local development and testing."""

ORDERS: dict[str, dict] = {
    "A12345": {
        "order_id": "A12345",
        "status": "Delivered",
        "total": 19.99,
        "item": "Glass Water Bottle",
        "delivery_date": "2026-03-22",
        "refund_eligible": True,
    },
    "B77890": {
        "order_id": "B77890",
        "status": "Delivered",
        "total": 64.50,
        "item": "Wireless Keyboard",
        "delivery_date": "2026-03-27",
        "refund_eligible": True,
    },
    "C00456": {
        "order_id": "C00456",
        "status": "Shipped",
        "total": 34.00,
        "item": "Yoga Mat",
        "delivery_date": None,
        "refund_eligible": False,
    },
}
