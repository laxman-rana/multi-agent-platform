import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# All tuneable thresholds in one place.
_THRESHOLDS: Dict[str, float] = {
    "volatility_penalty":          0.35,  # annualised vol above this → risk penalty
    "near_52w_high_pct":           0.05,  # within this fraction of 52w high → limited upside
    "lower_30_band_pct":           0.30,  # price must be in bottom 30% of 52w range for value signal
    "min_analyst_count":           3,     # require at least this many analysts for consensus signal
    "analyst_strong_upside":       0.20,  # analyst mean target must be ≥20% above price for bullish signal
    "vol_spike_ratio":             1.5,   # volume ratio threshold for buying pressure
    "capitulation_vol_ratio":      3.0,   # volume ratio for capitulation (panic-sell reversal)
    "capitulation_move_pct":       8.0,   # min abs(change_pct) to trigger capitulation/selling branch
}

# Signal weights — positive = bullish, negative = bearish.
# Volatility is -1 but suppressed during capitulation (high vol is expected
# during a panic sell and is already captured by the capitulation signal).
_WEIGHTS: Dict[str, int] = {
    "pe_improvement":   +1,
    "52w_lower_band":   +1,
    "volatility":       -1,   # elevated vol penalises non-event entries only
    "near_52w_high":    -1,
    "analyst_bullish":  +1,   # buy/strong_buy consensus with ≥20% upside target
    "analyst_bearish":  -1,   # underperform/sell consensus
    "buying_pressure":  +1,   # price up + volume spike → institutional accumulation proxy
    "selling_pressure": -1,   # big drop without panic volume → distribution signal
    "capitulation":     +2,   # big drop on extreme volume → panic exhaustion / reversal zone
}

# Tier thresholds: evaluated top-to-bottom; first match wins.
# score ≥ 3  → strong_buy  (multiple confirming signals — high conviction)
# score ≥ 2  → buy         (solid setup — worth LLM review)
# score ≤ 1  → neutral     (weak or ambiguous — filtered before LLM)
# score <  0 → avoid       (net bearish signals)
# Score range: -3 (all bearish) … +5 (capitulation + all bullish signals).
_TIERS: List[Tuple[int, str]] = [
    (3, "strong_buy"),   # ≥3 confirming signals — genuine conviction required
    (2, "buy"),          # 2 signals — solid setup, worth LLM review
    (0, "neutral"),
]


def _tier(score: int) -> str:
    for threshold, label in _TIERS:
        if score >= threshold:
            return label
    return "avoid"


def _infer_opportunity_type(pe_improved: bool, in_lower_band: bool) -> str:
    """
    Determine the opportunity type deterministically from which bullish signals fired.

    Priority order:
      in_lower_band                → "dip_buy"   price is in a dip zone (with or without PE backing)
      pe_improved (no lower band)  → "value"     pure earnings/fundamental play at current price
      neither fired                → "momentum"  ticker was surfaced by prefilter price/volume signal
    """
    if in_lower_band:
        return "dip_buy"
    if pe_improved:
        return "value"
    return "momentum"


# ---------------------------------------------------------------------------
# Individual signal functions — each returns (points: int, description: str).
# A return of (0, "") means the condition did not fire.
# ---------------------------------------------------------------------------

def _sig_pe_improvement(pe: Any, fwd_pe: Any) -> Tuple[int, str]:
    """Forward P/E < Trailing P/E → earnings expected to grow → bullish for new entry."""
    if pe and fwd_pe and fwd_pe < pe:
        return _WEIGHTS["pe_improvement"], f"Forward P/E ({fwd_pe:.1f}) < Trailing P/E ({pe:.1f}) → earnings growth expected"
    return 0, ""


def _sig_52w_lower_band(price: float, high: float, low: float) -> Tuple[int, str]:
    """Price in the lower 30% of its 52-week range → potential value entry point."""
    if not high or not low or high == low:
        return 0, ""
    range_size = high - low
    lower_band = low + _THRESHOLDS["lower_30_band_pct"] * range_size
    if price <= lower_band:
        pct_in_range = (price - low) / range_size * 100
        return _WEIGHTS["52w_lower_band"], (
            f"Price in lower 30% of 52w range ({pct_in_range:.0f}%) "
            f"→ potential value entry"
        )
    return 0, ""


def _sig_volatility_penalty(vol: float, is_capitulation: bool = False) -> Tuple[int, str]:
    """
    Elevated annualised volatility → higher entry risk, penalise the score.

    Suppressed during capitulation events: high volatility is expected when a
    stock drops ≥8% and is already reflected in the capitulation signal itself.
    Penalising it a second time unfairly double-counts the same risk.
    """
    if vol > _THRESHOLDS["volatility_penalty"]:
        if is_capitulation:
            return 0, ""
        return _WEIGHTS["volatility"], f"Volatility {vol:.0%} > 35% → elevated entry risk"
    return 0, ""


def _sig_near_52w_high(price: float, high: float) -> Tuple[int, str]:
    """Price near the 52-week high → limited near-term upside, avoid chasing."""
    if not high or high == 0:
        return 0, ""
    pct_from_high = (high - price) / high
    if pct_from_high < _THRESHOLDS["near_52w_high_pct"]:
        return _WEIGHTS["near_52w_high"], (
            f"Price within 5% of 52w high (${high:.2f}) "
            f"→ limited upside, buying near top"
        )
    return 0, ""


def _sig_analyst_consensus(
    rating: str, count: Any, target: Any, price: float
) -> Tuple[int, str]:
    """
    Analyst consensus signal using yfinance recommendationKey.

    Bullish (+1): buy/strong_buy consensus with ≥3 analysts AND mean target
    ≥20% above current price.  Requiring 20% upside filters out stale "buy"
    ratings where the target has already been reached, and avoids double-
    weighting the analyst source with a weak signal.
    Bearish (-1): underperform/sell consensus with ≥3 analysts.
    Neutral (0): hold, insufficient coverage, target upside <20%, or no data.
    """
    if not rating or rating == "none":
        return 0, ""
    try:
        count = int(count or 0)
    except (TypeError, ValueError):
        count = 0
    min_count = int(_THRESHOLDS["min_analyst_count"])
    if count < min_count:
        return 0, ""

    if rating in ("strong_buy", "buy"):
        if target and price:
            upside = (float(target) - float(price)) / float(price)
            if upside >= _THRESHOLDS["analyst_strong_upside"]:
                return _WEIGHTS["analyst_bullish"], (
                    f"Analyst consensus: {rating.replace('_', ' ')} ({count} analysts, "
                    f"target ${float(target):.2f}, +{upside:.0%} upside)"
                )
        return 0, ""

    if rating in ("underperform", "sell"):
        return _WEIGHTS["analyst_bearish"], (
            f"Analyst consensus: {rating} ({count} analysts) → negative outlook"
        )

    return 0, ""


def _sig_volume_event(
    change_pct: float, volume: int, avg_volume: int
) -> Tuple[int, str]:
    """
    Unified volume-event signal — avoids double-counting by making
    capitulation and selling-pressure mutually exclusive.

    Buying pressure  (+1): price up AND volume ≥ 1.5x avg
    Capitulation     (+2): price ≤ −8% AND volume ≥ 3x avg
                           (panic exhaustion → mean-reversion entry zone)
    Selling pressure (−1): price ≤ −8% AND volume < 3x avg
                           (distribution without exhaustion)
    Neutral (0): moderate drops (<8%), low volume, or flat price.
    """
    if not avg_volume or avg_volume <= 0:
        return 0, ""
    ratio = volume / avg_volume

    if change_pct > 0 and ratio >= _THRESHOLDS["vol_spike_ratio"]:
        return _WEIGHTS["buying_pressure"], (
            f"Buying pressure: price +{change_pct:.1f}% on {ratio:.1f}x avg volume "
            f"→ elevated demand signal"
        )

    if change_pct <= -_THRESHOLDS["capitulation_move_pct"]:
        if ratio >= _THRESHOLDS["capitulation_vol_ratio"]:
            return _WEIGHTS["capitulation"], (
                f"Capitulation: price {change_pct:.1f}% on {ratio:.1f}x avg volume "
                f"→ panic selling / exhaustion reversal zone"
            )
        return _WEIGHTS["selling_pressure"], (
            f"Selling pressure: price {change_pct:.1f}% (no panic volume) "
            f"→ distribution signal"
        )

    return 0, ""


class SignalEngine:
    """
    Deterministic quantitative scoring for BUY opportunity candidates.

    Key signals and weights:
      +2  Capitulation: drop ≥8% on ≥3x avg volume  (panic exhaustion reversal)
      +1  Buying pressure: price up on ≥1.5x avg volume
      +1  Forward P/E < Trailing P/E  (earnings growth expected)
      +1  Price in lower 30% of 52w range  (value entry zone)
      +1  Analyst buy consensus with ≥20% mean target upside
      -1  Selling pressure: drop ≥8% without panic volume
      -1  Volatility > 35%  (suppressed if capitulation fired)
      -1  Price within 5% of 52w high  (limited upside)
      -1  Analyst sell/underperform consensus

    Tier:
      score >= 3  → "strong_buy"
      score >= 2  → "buy"
      score <= 1  → "neutral"
      score <  0  → "avoid"
    """

    def score(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single ticker.

        Parameters
        ----------
        market_data : extended market data dict from _fetch_extended().

        Returns
        -------
        dict with:
            score   : int             — net signal total
            signals : List[str]       — human-readable descriptions of fired signals
            tier    : str             — "buy" | "neutral" | "avoid"
            type    : str             — "dip_buy" | "value" | "momentum"  (deterministic)
        """
        # Compute the two bullish signals first so their fired state can be
        # inspected for deterministic opportunity-type classification.
        pe_sig   = _sig_pe_improvement(
            market_data.get("pe_ratio"),
            market_data.get("forward_pe"),
        )
        band_sig = _sig_52w_lower_band(
            market_data.get("price", 0.0),
            market_data.get("52w_high", 0.0),
            market_data.get("52w_low", 0.0),
        )

        # Volume event is computed first — its result determines whether the
        # volatility penalty should be suppressed (capitulation branch).
        vol_event_sig = _sig_volume_event(
            market_data.get("change_pct", 0.0),
            market_data.get("volume", 0),
            market_data.get("avg_volume", 0),
        )
        is_capitulation = vol_event_sig[0] == _WEIGHTS["capitulation"]

        raw_signals: List[Tuple[int, str]] = [
            pe_sig,
            band_sig,
            _sig_volatility_penalty(market_data.get("volatility", 0.0), is_capitulation),
            _sig_near_52w_high(
                market_data.get("price", 0.0),
                market_data.get("52w_high", 0.0),
            ),
            _sig_analyst_consensus(
                market_data.get("analyst_rating", "none"),
                market_data.get("analyst_count", 0),
                market_data.get("analyst_target"),
                market_data.get("price", 0.0),
            ),
            vol_event_sig,
        ]

        active      = [(pts, label) for pts, label in raw_signals if pts != 0 and label]
        total_score = sum(pts for pts, _ in active)
        signals     = [label for _, label in active]

        return {
            "score":   total_score,
            "signals": signals,
            "tier":    _tier(total_score),
            "type":    _infer_opportunity_type(
                pe_improved=pe_sig[0] > 0,
                in_lower_band=band_sig[0] > 0,
            ),
        }
