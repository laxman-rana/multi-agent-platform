import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# All tuneable thresholds in one place — no magic numbers in the logic below.
_PREFILTER_THRESHOLDS: Dict[str, float] = {
    "change_pct_abs": 3.0,
    "lower_band_pct": 0.20,
    "volume_spike_ratio": 1.5,
    "volatility_spike": 0.35,
    "min_market_cap": 2e9,
    "min_profit_margin": 0.08,
    "min_revenue_growth": 0.05,
}


class PreFilterEngine:
    """
    Lightweight OR-logic pre-filter applied before the SignalEngine and LLM.

    The quality-first agent remains intentionally permissive: a ticker can pass
    either because the business already shows a basic quality footprint or
    because market behaviour suggests the name deserves a closer look.

    A ticker is unconditionally rejected only when its price data is absent,
    as there is nothing to score.
    """

    def pre_filter(self, market_data: Dict[str, Any]) -> bool:
        """
        Return True if the ticker should proceed to the SignalEngine.

        Parameters
        ----------
        market_data : dict produced by `_fetch_extended()` in alpha_scanner_agent.
        """
        price = market_data.get("price")
        if not price:
            return False

        change_pct = market_data.get("change_pct", 0.0)
        volatility = market_data.get("volatility", 0.0)
        high = market_data.get("52w_high", 0.0)
        low = market_data.get("52w_low", 0.0)
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("avg_volume", 0)
        market_cap = market_data.get("market_cap") or 0
        profit_margin = market_data.get("profit_margins")
        revenue_growth = market_data.get("revenue_growth")

        sym = market_data.get("ticker", "?")

        if (
            market_cap >= _PREFILTER_THRESHOLDS["min_market_cap"]
            and (
                (profit_margin is not None and profit_margin >= _PREFILTER_THRESHOLDS["min_profit_margin"])
                or (revenue_growth is not None and revenue_growth >= _PREFILTER_THRESHOLDS["min_revenue_growth"])
            )
        ):
            logger.debug(
                "PreFilter PASS [%s]: quality footprint market_cap=%.1fB profit_margin=%s revenue_growth=%s",
                sym,
                market_cap / 1e9,
                f"{profit_margin:.0%}" if profit_margin is not None else "N/A",
                f"{revenue_growth:.0%}" if revenue_growth is not None else "N/A",
            )
            return True

        if abs(change_pct) > _PREFILTER_THRESHOLDS["change_pct_abs"]:
            logger.debug("PreFilter PASS [%s]: price move %+.1f%%", sym, change_pct)
            return True

        if high and low and high > low:
            range_size = high - low
            lower_band = low + _PREFILTER_THRESHOLDS["lower_band_pct"] * range_size
            if price <= lower_band:
                logger.debug("PreFilter PASS [%s]: price in lower 20%% of 52w range", sym)
                return True

        if avg_volume > 0 and volume >= avg_volume * _PREFILTER_THRESHOLDS["volume_spike_ratio"]:
            logger.debug("PreFilter PASS [%s]: volume spike %d vs avg %d", sym, volume, avg_volume)
            return True

        if volatility > _PREFILTER_THRESHOLDS["volatility_spike"]:
            logger.debug("PreFilter PASS [%s]: volatility %.0f%% > 35%%", sym, volatility * 100)
            return True

        vol_ratio = (volume / avg_volume) if avg_volume > 0 else 0.0
        if high and low and high > low:
            lower_band = low + _PREFILTER_THRESHOLDS["lower_band_pct"] * (high - low)
            band_info = f"  52w-band=${lower_band:.2f} (price=${price:.2f})"
        else:
            band_info = "  52w range unavailable"
        logger.debug(
            "PreFilter FAIL [%s]: move=%+.1f%% (need ±3%%)  vol-ratio=%.2fx (need 1.5x)"
            "  ann-vol=%.0f%% (need >35%%)  market_cap=%s  profit_margin=%s  revenue_growth=%s%s",
            sym,
            change_pct,
            vol_ratio,
            volatility * 100,
            f"${market_cap / 1e9:.1f}B" if market_cap else "N/A",
            f"{profit_margin:.0%}" if profit_margin is not None else "N/A",
            f"{revenue_growth:.0%}" if revenue_growth is not None else "N/A",
            band_info,
        )
        return False
