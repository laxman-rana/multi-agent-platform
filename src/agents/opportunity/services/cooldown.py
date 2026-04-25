from datetime import datetime, timezone
from typing import Any, Dict


class CooldownPolicy:
    def __init__(self, cooldown_minutes: int = 30, cooldown_unit: str = "minutes") -> None:
        self._cooldown_minutes = cooldown_minutes
        self._cooldown_unit = cooldown_unit

    def is_cooled_down(self, ticker: str, recent_signals: Dict[str, str]) -> bool:
        last_signal = recent_signals.get(ticker)
        if not last_signal:
            return True
        try:
            last_dt = datetime.fromisoformat(last_signal)
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
            if self._cooldown_unit == "hours":
                return (elapsed / 3600) >= self._cooldown_minutes
            if self._cooldown_unit == "days":
                return (elapsed / 86400) >= self._cooldown_minutes
            return (elapsed / 60) >= self._cooldown_minutes
        except (ValueError, TypeError):
            return True

    def is_fresh_despite_cooldown(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        recent_signal_context: Dict[str, Any],
    ) -> bool:
        change_pct = market_data.get("change_pct", 0.0)
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("avg_volume", 0)

        if change_pct <= -8.0 and avg_volume > 0 and volume / avg_volume >= 3.0:
            return True

        ctx = recent_signal_context.get(ticker)
        if not ctx:
            return False

        prev_price = ctx.get("price", 0.0)
        curr_price = market_data.get("price", 0.0)
        if prev_price > 0 and curr_price > 0:
            drop_since_pct = (curr_price - prev_price) / prev_price * 100
            return drop_since_pct <= -5.0
        return False
