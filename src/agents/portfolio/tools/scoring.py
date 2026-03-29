"""
scoring.py
----------
Deterministic quantitative scoring layer for portfolio positions.

Each signal adds or subtracts a fixed integer point, producing a total score
that is passed to the LLM alongside the raw data.  This grounds the model's
reasoning in objective metrics and mirrors the signal-stacking approach used
in real quant systems.

Usage::

    from src.agents.portfolio.tools.scoring import score_stock

    result = score_stock(insight, gain_pct=12.5)
    # result["score"]      → int, e.g. +2
    # result["tier"]       → str, e.g. "buy"
    # result["signals"]    → list of (label, points) tuples — for prompt rendering
    # result["breakdown"]  → human-readable list of strings
"""

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Tier thresholds
# ---------------------------------------------------------------------------
_TIERS: List[Tuple[int, str]] = [
    (3,  "strong_buy"),
    (1,  "buy"),
    (0,  "neutral"),
    (-2, "sell"),
]   # anything below -2 → "strong_sell"


def _tier(score: int) -> str:
    for threshold, label in _TIERS:
        if score >= threshold:
            return label
    return "strong_sell"


# ---------------------------------------------------------------------------
# Individual signal functions
# ---------------------------------------------------------------------------

def _sig_pe_improvement(pe: Any, fwd_pe: Any) -> Tuple[int, str]:
    """Forward PE < trailing PE → earnings expected to grow → bullish."""
    if pe and fwd_pe and fwd_pe < pe:
        return +1, f"Forward P/E ({fwd_pe:.1f}) < Trailing P/E ({pe:.1f}) → earnings growth expected"
    return 0, ""


def _sig_pe_valuation(pe: Any) -> Tuple[int, str]:
    """Raw valuation check on trailing PE."""
    if not pe:
        return 0, ""
    if pe > 40:
        return -1, f"Trailing P/E {pe:.1f} > 40 → expensive valuation"
    if pe < 15:
        return +1, f"Trailing P/E {pe:.1f} < 15 → attractive valuation"
    return 0, ""


def _sig_volatility(vol: float) -> Tuple[int, str]:
    """Annualised volatility risk penalty."""
    if vol > 0.50:
        return -2, f"Volatility {vol:.0%} > 50% → very high risk"
    if vol > 0.30:
        return -1, f"Volatility {vol:.0%} > 30% → elevated risk"
    if vol < 0.20:
        return +1, f"Volatility {vol:.0%} < 20% → low risk"
    return 0, ""


def _sig_unrealised_pnl(gain_pct: float) -> Tuple[int, str]:
    """Position-level gain/loss momentum."""
    if gain_pct < -20:
        return -1, f"Down {gain_pct:.1f}% → extended drawdown, downtrend risk"
    if gain_pct > 30:
        return +1, f"Up {gain_pct:.1f}% → strong uptrend momentum"
    return 0, ""


def _sig_daily_change(change_pct: float) -> Tuple[int, str]:
    """
    Short-term price momentum from today's session.

    Threshold raised to 3% (from 2%) to avoid penalising every stock on a
    broad market down-day — a 2-3% intraday move is normal volatility and
    does not distinguish individual weakness from macro noise.
    """
    if change_pct > 3.0:
        return +1, f"Daily change {change_pct:+.1f}% → strong positive session"
    if change_pct < -3.0:
        return -1, f"Daily change {change_pct:+.1f}% → strong negative session"
    return 0, ""


def _sig_52w_position(price: float, high: float, low: float) -> Tuple[int, str]:
    """
    Where the price sits in its 52-week range.
    Near the high → momentum signal.
    Near the low  → downtrend risk (tightened to 5% from 10% to avoid
                    triggering on normal mid-range dips).
    """
    if not high or not low or high == low:
        return 0, ""
    pct_from_high = (high - price) / high
    pct_from_low  = (price - low)  / low
    if pct_from_high < 0.05:
        return +1, f"Price within 5% of 52w high (${high:.2f}) \u2192 near-high momentum"
    if pct_from_low < 0.05:
        return -1, f"Price within 5% of 52w low (${low:.2f}) \u2192 near support / downtrend"
    return 0, ""


def _sig_mean_reversion(
    price: float,
    high: float,
    low: float,
    pe: Any,
    fwd_pe: Any,
    gain_pct: float,
) -> Tuple[int, str]:
    """
    Contrarian / mean-reversion signal.

    A stock pulled down to the lower portion of its 52-week range while
    still showing positive earnings-growth fundamentals (forward PE improving)
    is a candidate for a bounce, not just an exit.  This counterbalances the
    pure momentum / downtrend penalty from _sig_52w_position.

    Fires when ALL three conditions hold:
      1. Price is in the lower 30% of its 52-week range (oversold territory)
      2. Forward PE < Trailing PE  (earnings expected to grow)
      3. Unrealised loss is not catastrophic (> -25%)  — avoids catching
         falling knives on fundamentally broken positions
    """
    if not high or not low or high == low:
        return 0, ""
    range_size   = high - low
    lower_30_pct = low + 0.30 * range_size
    if price > lower_30_pct:
        return 0, ""
    if not (pe and fwd_pe and fwd_pe < pe):
        return 0, ""
    if gain_pct < -25:
        return 0, ""  # too deep a loss — skip contrarian signal
    pct_in_range = (price - low) / range_size * 100
    return (
        +1,
        f"Price in lower 30% of 52w range ({pct_in_range:.0f}%) + earnings growth "
        f"expected \u2192 mean-reversion / bounce candidate",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_stock(
    insight: Dict[str, Any],
    gain_pct: float,
    horizon_years: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute a deterministic quantitative score for a single position.

    Parameters
    ----------
    insight        : dict produced by MarketAgent (price, volatility, pe_ratio, etc.)
    gain_pct       : unrealised P&L percentage, pre-computed by DecisionAgent
    horizon_years  : investor's time horizon in years (default 1.0).
                     When ≥ 3 the daily-change signal is suppressed — a single
                     session's move is noise at a multi-year timescale.

    Returns
    -------
    dict with keys:
        score      int            — net signal total
        tier       str            — "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell"
        signals    list[tuple]    — [(label, points), ...] for every non-zero signal
        breakdown  list[str]      — human-readable lines for LLM prompt injection
        long_term  bool           — True when horizon_years >= 3 (daily signal suppressed)
    """
    long_term = horizon_years >= 3.0

    raw_signals: List[Tuple[int, str]] = [
        _sig_pe_improvement(insight.get("pe_ratio"), insight.get("forward_pe")),
        _sig_pe_valuation(insight.get("pe_ratio")),
        _sig_volatility(insight.get("volatility", 0.0)),
        _sig_unrealised_pnl(gain_pct),
        # Daily session change is noise for long-term investors — suppress at ≥ 3 years.
        (0, "") if long_term else _sig_daily_change(insight.get("change_pct", 0.0)),
        _sig_52w_position(
            insight.get("price", 0.0),
            insight.get("52w_high", 0.0),
            insight.get("52w_low", 0.0),
        ),
        _sig_mean_reversion(
            insight.get("price", 0.0),
            insight.get("52w_high", 0.0),
            insight.get("52w_low", 0.0),
            insight.get("pe_ratio"),
            insight.get("forward_pe"),
            gain_pct,
        ),
    ]

    active = [(pts, label) for pts, label in raw_signals if pts != 0 and label]
    score  = sum(pts for pts, _ in active)
    tier   = _tier(score)

    breakdown = [
        f"  {'[+'+str(pts)+']' if pts > 0 else '['+str(pts)+']':>4}  {label}"
        for pts, label in active
    ]

    return {
        "score":     score,
        "tier":      tier,
        "signals":   [(label, pts) for pts, label in active],
        "breakdown": breakdown,
        "long_term": long_term,
    }
