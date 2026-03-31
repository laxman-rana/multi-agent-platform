import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# All tuneable thresholds in one place — no magic numbers in the logic below.
_PREFILTER_THRESHOLDS: Dict[str, float] = {
    "change_pct_abs":    3.0,   # abs daily price move threshold
    "lower_band_pct":    0.20,  # price must be in the bottom 20% of 52w range
    "volume_spike_ratio": 1.5,  # current volume must exceed avg by this multiple
    "volatility_spike":  0.35,  # annualised volatility threshold
}


class PreFilterEngine:
    """
    Lightweight OR-logic pre-filter applied before the SignalEngine and LLM.

    A ticker passes if ANY single condition is True.  The goal is to discard
    obviously uninteresting tickers early — reducing signal-engine and LLM
    calls at scale — while being very permissive to avoid false negatives.

    A ticker is unconditionally rejected only when its price data is absent,
    as there is nothing to score.
    """

    def pre_filter(self, market_data: Dict[str, Any]) -> bool:
        """
        Return True if the ticker should proceed to the SignalEngine.

        Parameters
        ----------
        market_data : dict produced by _fetch_extended() in alpha_scanner_agent.
        """
        price = market_data.get("price")
        if not price:
            return False

        change_pct = market_data.get("change_pct", 0.0)
        volatility  = market_data.get("volatility", 0.0)
        high        = market_data.get("52w_high", 0.0)
        low         = market_data.get("52w_low", 0.0)
        volume      = market_data.get("volume", 0)
        avg_volume  = market_data.get("avg_volume", 0)

        sym = market_data.get("ticker", "?")

        # 1. Meaningful price move (up or down)
        if abs(change_pct) > _PREFILTER_THRESHOLDS["change_pct_abs"]:
            logger.debug("PreFilter PASS [%s]: price move %+.1f%%", sym, change_pct)
            return True

        # 2. Price near 52-week low — potential value-entry zone
        if high and low and high > low:
            range_size = high - low
            lower_band = low + _PREFILTER_THRESHOLDS["lower_band_pct"] * range_size
            if price <= lower_band:
                logger.debug("PreFilter PASS [%s]: price in lower 20%% of 52w range", sym)
                return True

        # 3. Above-average volume — unusual interest or institutional activity
        if avg_volume > 0 and volume >= avg_volume * _PREFILTER_THRESHOLDS["volume_spike_ratio"]:
            logger.debug("PreFilter PASS [%s]: volume spike %d vs avg %d", sym, volume, avg_volume)
            return True

        # 4. Elevated volatility — warrants closer examination
        if volatility > _PREFILTER_THRESHOLDS["volatility_spike"]:
            logger.debug("PreFilter PASS [%s]: volatility %.0f%% > 35%%", sym, volatility * 100)
            return True

        # None of the OR conditions fired — log the miss at DEBUG so --verbose shows it.
        vol_ratio = (volume / avg_volume) if avg_volume > 0 else 0.0
        if high and low and high > low:
            lower_band = low + _PREFILTER_THRESHOLDS["lower_band_pct"] * (high - low)
            band_info  = f"  52w-band=${lower_band:.2f} (price=${price:.2f})"
        else:
            band_info  = "  52w range unavailable"
        logger.debug(
            "PreFilter FAIL [%s]: move=%+.1f%% (need ±3%%)  vol-ratio=%.2fx (need 1.5x)"
            "  ann-vol=%.0f%% (need >35%%)%s",
            sym, change_pct, vol_ratio, volatility * 100, band_info,
        )
        return False
