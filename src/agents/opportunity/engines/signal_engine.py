"""
signal_engine.py
----------------
Deterministic quality scoring for buy-opportunity candidates.

CHANGES vs previous version
----------------------------
Fix 1 — Growth tiering (was: binary on/off at 5%)
  Revenue growth and earnings growth now have two tiers:
    - baseline  (>=5%  rev / >=5%  earnings) : +1 pt  (business is growing)
    - high       (>=20% rev / >=25% earnings) : +1 pt  additional (compounder quality)
  This correctly separates a 6%-grower from a 70%-compounder.

Fix 2 — FCF yield replaces naked FCF positivity (was: fcf > 0 = +1)
  FCF is now normalised by market cap.
    - FCF yield >= 2%  : +1 pt  (self-funding business)
    - FCF yield >= 5%  : +2 pts (cash machine — banker-grade conviction)
  A $100B company with $100M FCF no longer scores the same as AAPL.

Fix 3 — Debt/equity threshold tightened (was: D/E <= 1.0)
  - D/E <= 0.75 : +1 pt  (balance-sheet discipline)
  - D/E > 1.5   : -1 pt  (leverage penalty, new)
  Rationale: D/E of 1.0 is permissive for most non-financial businesses.
  Investment-grade quality generally runs below 0.75x.

Fix 4 — Score tier boundaries recalibrated for quality-first system
  Old boundaries assumed soft signals (lower_band, buying_pressure, analyst)
  were needed to reach STRONG_BUY.  A pure-fundamentals elite business should
  reach STRONG_BUY on fundamentals alone.

  New max achievable score (all signals fire, no penalties):
    profitability      +2
    operating_margin   +1
    roe                +1
    balance_sheet      +1
    fcf_yield_base     +1
    fcf_yield_high     +1  (new)
    revenue_growth     +1
    revenue_high       +1  (new)
    earnings_growth    +1
    earnings_high      +1  (new)
    valuation_support  +1
    analyst_bullish    +1
    lower_band         +1
    buying_pressure    +1
    ─────────────────────
    theoretical max   +15

  Tier boundaries (recalibrated):
    elite        >= 9  (was 7) — requires most Tier 1 fundamentals to fire
    high_quality >= 6  (was 5) — solid fundamentals, may lack one or two
    watchlist    >= 4  (was 3) — some quality present, needs monitoring
    avoid        < 4

Fix 5 — _CANDIDATE_MIN_SCORE raised from 4 to 5
  Score 4 was reachable with: profitability(+2) + any one growth signal(+1)
  + lower_band(+1) — i.e. a cheap growing stock with no other quality markers.
  Score 5 now requires at least two independent quality signals beyond basic
  profitability, making value-trap filtering meaningfully stricter.
  (Applied in alpha_scanner_agent.py, documented here for traceability.)
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── All tuneable thresholds in one place ──────────────────────────────────────
_THRESHOLDS: Dict[str, float] = {
    # Profitability
    "min_profit_margin":        0.10,
    "min_operating_margin":     0.12,
    "min_roe":                  0.15,

    # Balance sheet — Fix 3: tightened from 1.0, added leverage penalty
    "max_debt_to_equity":       0.75,   # was 1.0
    "high_debt_to_equity":      1.5,    # new: penalty above this

    # FCF yield — Fix 2: replaces naked fcf > 0
    "min_fcf_yield":            0.02,   # 2% FCF/market_cap = self-funding
    "high_fcf_yield":           0.05,   # 5% FCF/market_cap = cash machine

    # Growth — Fix 1: tiered thresholds
    "min_revenue_growth":       0.05,
    "high_revenue_growth":      0.20,   # new: compounder tier
    "min_earnings_growth":      0.05,
    "high_earnings_growth":     0.25,   # new: compounder tier

    # Valuation
    "max_forward_pe":           35.0,
    "extreme_forward_pe":       45.0,

    # 52-week range
    "near_52w_high_pct":        0.03,
    "lower_35_band_pct":        0.35,

    # Volatility
    "volatility_penalty":       0.45,

    # Analyst
    "min_analyst_count":        3,
    "analyst_strong_upside":    0.15,

    # Volume
    "vol_spike_ratio":          1.5,
    "capitulation_vol_ratio":   3.0,
    "capitulation_move_pct":    8.0,
}

# ── Point weights ──────────────────────────────────────────────────────────────
_WEIGHTS: Dict[str, int] = {
    # Tier 1 — hard quality signals
    "profitability":        +2,
    "operating_margin":     +1,
    "roe":                  +1,
    "balance_sheet":        +1,
    "fcf_yield_base":       +1,   # Fix 2: was "free_cash_flow" = fcf > 0
    "fcf_yield_high":       +1,   # Fix 2: new — high FCF yield bonus
    "revenue_growth":       +1,
    "revenue_high":         +1,   # Fix 1: new — high revenue growth bonus
    "earnings_growth":      +1,
    "earnings_high":        +1,   # Fix 1: new — high earnings growth bonus
    "valuation_support":    +1,

    # Tier 2 — soft/market signals (contextual, do not drive decision)
    "analyst_bullish":      +1,
    "lower_band":           +1,
    "buying_pressure":      +1,

    # Penalties
    "extreme_valuation":    -2,
    "high_leverage":        -1,   # Fix 3: new leverage penalty
    "near_52w_high":        -1,
    "analyst_bearish":      -1,
    "selling_pressure":     -1,
    "volatility":           -1,
}

# ── Quality tier boundaries — Fix 4: recalibrated ─────────────────────────────
# Old: elite>=7, high_quality>=5, watchlist>=3
# New: elite>=9, high_quality>=6, watchlist>=4
# Rationale: with max ~15 pts available, a fundamentals-only elite business
# (no market/analyst signals) can score ~10-11. Raising elite to 9 ensures
# only businesses with broad Tier 1 signal coverage reach STRONG_BUY.
_QUALITY_TIERS: List[Tuple[int, str]] = [
    (9, "elite"),          # was 7
    (6, "high_quality"),   # was 5
    (4, "watchlist"),      # was 3
]


def _quality_tier(score: int) -> str:
    for threshold, label in _QUALITY_TIERS:
        if score >= threshold:
            return label
    return "avoid"


def _tier(score: int) -> str:
    """Backward-compatible coarse tier for callers expecting the old shape."""
    if score >= 6:
        return "strong_buy"
    if score >= 4:
        return "buy"
    if score >= 2:
        return "neutral"
    return "avoid"


def _fmt_pct(value: float) -> str:
    return f"{value:.0%}"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Tier 1 signal functions ───────────────────────────────────────────────────

def _sig_profitability(margin: Any) -> Tuple[int, str]:
    margin_f = _safe_float(margin)
    if margin_f is not None and margin_f >= _THRESHOLDS["min_profit_margin"]:
        return _WEIGHTS["profitability"], (
            f"Healthy profit margin {_fmt_pct(margin_f)} supports durable profitability"
        )
    return 0, ""


def _sig_operating_margin(margin: Any) -> Tuple[int, str]:
    margin_f = _safe_float(margin)
    if margin_f is not None and margin_f >= _THRESHOLDS["min_operating_margin"]:
        return _WEIGHTS["operating_margin"], (
            f"Operating margin {_fmt_pct(margin_f)} indicates pricing power"
        )
    return 0, ""


def _sig_roe(roe: Any) -> Tuple[int, str]:
    roe_f = _safe_float(roe)
    if roe_f is not None and roe_f >= _THRESHOLDS["min_roe"]:
        return _WEIGHTS["roe"], (
            f"Return on equity {_fmt_pct(roe_f)} shows efficient capital allocation"
        )
    return 0, ""


def _sig_balance_sheet(debt_to_equity: Any) -> Tuple[int, str]:
    """
    Fix 3: tightened D/E ceiling from 1.0 to 0.75; added -1 penalty above 1.5.
    Returns a list of (score, label) tuples — at most one reward and one penalty.
    """
    debt_f = _safe_float(debt_to_equity)
    signals: List[Tuple[int, str]] = []

    if debt_f is not None:
        if debt_f <= _THRESHOLDS["max_debt_to_equity"]:
            signals.append((
                _WEIGHTS["balance_sheet"],
                f"Debt-to-equity {debt_f:.2f} reflects investment-grade balance-sheet discipline",
            ))
        if debt_f > _THRESHOLDS["high_debt_to_equity"]:
            signals.append((
                _WEIGHTS["high_leverage"],
                f"Debt-to-equity {debt_f:.2f} indicates elevated leverage risk",
            ))

    return signals  # type: ignore[return-value]
    # NOTE: caller handles this as a list — see score() below


def _sig_fcf_yield(free_cash_flow: Any, market_cap: Any) -> List[Tuple[int, str]]:
    """
    Fix 2: FCF quality measured as yield (FCF / market_cap), not just sign.
    Returns 0, 1, or 2 signals.
    """
    fcf_f = _safe_float(free_cash_flow)
    cap_f = _safe_float(market_cap)
    signals: List[Tuple[int, str]] = []

    if fcf_f is None or cap_f is None or cap_f <= 0:
        return signals

    if fcf_f <= 0:
        return signals

    fcf_yield = fcf_f / cap_f

    if fcf_yield >= _THRESHOLDS["high_fcf_yield"]:
        return [
            (_WEIGHTS["fcf_yield_base"] + _WEIGHTS["fcf_yield_high"],
             f"FCF yield {_fmt_pct(fcf_yield)} — cash-generative business quality (+2pts)"),
        ]
    if fcf_yield >= _THRESHOLDS["min_fcf_yield"]:
        return [
            (_WEIGHTS["fcf_yield_base"],
             f"FCF yield {_fmt_pct(fcf_yield)} confirms self-funded growth capability"),
        ]

    return signals


def _sig_revenue_growth(revenue_growth: Any) -> List[Tuple[int, str]]:
    """
    Tiered revenue growth. High tier merges both points into one label (no duplicate).
    """
    growth = _safe_float(revenue_growth)
    if growth is None:
        return []
    if growth >= _THRESHOLDS["high_revenue_growth"]:
        return [
            (_WEIGHTS["revenue_growth"] + _WEIGHTS["revenue_high"],
             f"Revenue growth {_fmt_pct(growth)} — compounder-grade expansion (+2pts)"),
        ]
    if growth >= _THRESHOLDS["min_revenue_growth"]:
        return [
            (_WEIGHTS["revenue_growth"],
             f"Revenue growth {_fmt_pct(growth)} supports business durability"),
        ]
    return []


def _sig_earnings_growth(earnings_growth: Any) -> List[Tuple[int, str]]:
    """
    Tiered earnings growth. High tier merges both points into one label (no duplicate).
    """
    growth = _safe_float(earnings_growth)
    if growth is None:
        return []
    if growth >= _THRESHOLDS["high_earnings_growth"]:
        return [
            (_WEIGHTS["earnings_growth"] + _WEIGHTS["earnings_high"],
             f"Earnings growth {_fmt_pct(growth)} — operating leverage at scale (+2pts)"),
        ]
    if growth >= _THRESHOLDS["min_earnings_growth"]:
        return [
            (_WEIGHTS["earnings_growth"],
             f"Earnings growth {_fmt_pct(growth)} supports compounding potential"),
        ]
    return []


def _sig_valuation_support(forward_pe: Any, trailing_pe: Any) -> Tuple[int, str]:
    forward  = _safe_float(forward_pe)
    trailing = _safe_float(trailing_pe)

    if forward is None:
        return 0, ""

    if forward > _THRESHOLDS["extreme_forward_pe"]:
        return (
            _WEIGHTS["extreme_valuation"],
            f"Forward P/E {forward:.1f} is too rich for a quality-first entry",
        )

    if forward <= _THRESHOLDS["max_forward_pe"] and (trailing is None or forward <= trailing):
        trailing_txt = f" vs trailing P/E {trailing:.1f}" if trailing is not None else ""
        return (
            _WEIGHTS["valuation_support"],
            f"Forward P/E {forward:.1f}{trailing_txt} keeps valuation disciplined",
        )

    return 0, ""


def _sig_analyst_consensus(
    rating: str, count: Any, target: Any, price: float
) -> Tuple[int, str]:
    if not rating or rating == "none":
        return 0, ""

    try:
        analyst_count = int(count or 0)
    except (TypeError, ValueError):
        analyst_count = 0

    if analyst_count < int(_THRESHOLDS["min_analyst_count"]):
        return 0, ""

    if rating in ("strong_buy", "buy"):
        target_f = _safe_float(target)
        if target_f and price:
            upside = (target_f - price) / price
            if upside >= _THRESHOLDS["analyst_strong_upside"]:
                return (
                    _WEIGHTS["analyst_bullish"],
                    f"Analyst support constructive ({analyst_count} analysts, +{upside:.0%} target upside)",
                )
        return 0, ""

    if rating in ("underperform", "sell"):
        return (
            _WEIGHTS["analyst_bearish"],
            f"Analyst consensus is cautious ({rating}, {analyst_count} analysts)",
        )

    return 0, ""


def _sig_52w_context(
    price: float, high: float, low: float
) -> List[Tuple[int, str]]:
    if not high or not low or high <= low or not price:
        return []

    range_size   = high - low
    lower_band   = low + _THRESHOLDS["lower_35_band_pct"] * range_size
    pct_from_high = (high - price) / high
    signals: List[Tuple[int, str]] = []

    if price <= lower_band:
        pct_in_range = (price - low) / range_size
        signals.append((
            _WEIGHTS["lower_band"],
            f"Price in lower {_fmt_pct(_THRESHOLDS['lower_35_band_pct'])} of 52w range "
            f"({_fmt_pct(pct_in_range)}) — potential value entry",
        ))

    if pct_from_high < _THRESHOLDS["near_52w_high_pct"]:
        signals.append((
            _WEIGHTS["near_52w_high"],
            f"Price within {_fmt_pct(_THRESHOLDS['near_52w_high_pct'])} of 52w high — "
            f"reduced margin of safety",
        ))

    return signals


def _sig_volume_event(
    change_pct: float, volume: int, avg_volume: int
) -> Tuple[int, str]:
    if not avg_volume or avg_volume <= 0:
        return 0, ""

    ratio = volume / avg_volume

    if change_pct > 0 and ratio >= _THRESHOLDS["vol_spike_ratio"]:
        return (
            _WEIGHTS["buying_pressure"],
            f"Buying interest: price +{change_pct:.1f}% on {ratio:.1f}x avg volume",
        )

    if (
        change_pct <= -_THRESHOLDS["capitulation_move_pct"]
        and ratio < _THRESHOLDS["capitulation_vol_ratio"]
    ):
        return (
            _WEIGHTS["selling_pressure"],
            f"Selling pressure: price {change_pct:.1f}% without exhaustion volume",
        )

    return 0, ""


def _sig_volatility_penalty(volatility: float) -> Tuple[int, str]:
    if volatility > _THRESHOLDS["volatility_penalty"]:
        return (
            _WEIGHTS["volatility"],
            f"Annualised volatility {_fmt_pct(volatility)} adds meaningful execution risk",
        )
    return 0, ""


def _infer_opportunity_type(active_signal_names: List[str], lower_band: bool) -> str:
    has_high_growth = "revenue_high" in active_signal_names or "earnings_high" in active_signal_names
    has_high_fcf    = "fcf_yield_high" in active_signal_names

    if lower_band and has_high_growth:
        return "quality_value_compounder"
    if lower_band:
        return "quality_value"
    if has_high_growth and has_high_fcf:
        return "elite_compounder"
    if has_high_growth:
        return "compounder"
    return "quality_watchlist"


class SignalEngine:
    """
    Deterministic quality scoring for a single ticker.

    Scoring philosophy (investment-banker grade):
      - Tier 1 fundamentals are the primary quality arbiter.
      - FCF yield, not FCF positivity, is the cash-quality signal.
      - Growth is tiered: a compounder earns more points than a grower.
      - Balance sheet discipline is measured against investment-grade standards.
      - Market behaviour (52w range, volume) is contextual, not decisive.
    """

    def score(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # ── Tier 1: hard quality signals ──────────────────────────────────
        profitability_sig     = _sig_profitability(market_data.get("profit_margins"))
        operating_margin_sig  = _sig_operating_margin(market_data.get("operating_margins"))
        roe_sig               = _sig_roe(market_data.get("return_on_equity"))

        # Fix 3: balance sheet returns a list (reward + optional penalty)
        balance_sheet_sigs    = _sig_balance_sheet(market_data.get("debt_to_equity"))

        # Fix 2: FCF yield returns a list (base + optional high-yield bonus)
        fcf_yield_sigs        = _sig_fcf_yield(
            market_data.get("free_cash_flow"),
            market_data.get("market_cap"),
        )

        # Fix 1: growth returns a list (base + optional compounder bonus)
        revenue_growth_sigs   = _sig_revenue_growth(market_data.get("revenue_growth"))
        earnings_growth_sigs  = _sig_earnings_growth(market_data.get("earnings_growth"))

        valuation_sig         = _sig_valuation_support(
            market_data.get("forward_pe"),
            market_data.get("pe_ratio"),
        )

        # ── Tier 2: soft/contextual signals ───────────────────────────────
        analyst_sig   = _sig_analyst_consensus(
            market_data.get("analyst_rating", "none"),
            market_data.get("analyst_count", 0),
            market_data.get("analyst_target"),
            market_data.get("price", 0.0),
        )
        range_sigs    = _sig_52w_context(
            market_data.get("price", 0.0),
            market_data.get("52w_high", 0.0),
            market_data.get("52w_low", 0.0),
        )
        volume_sig    = _sig_volume_event(
            market_data.get("change_pct", 0.0),
            market_data.get("volume", 0),
            market_data.get("avg_volume", 0),
        )
        volatility_sig = _sig_volatility_penalty(market_data.get("volatility", 0.0))

        # ── Flatten all signals into a uniform (name, pts, label) list ────
        # Scalar signals: (name, (pts, label))
        scalar_signals = [
            ("profitability",    profitability_sig),
            ("operating_margin", operating_margin_sig),
            ("roe",              roe_sig),
            ("valuation_support", valuation_sig),
            ("analyst_consensus", analyst_sig),
            ("volume_event",     volume_sig),
            ("volatility",       volatility_sig),
        ]

        named_signals: List[Tuple[str, int, str]] = []

        for sig_name, (pts, label) in scalar_signals:
            if pts != 0 and label:
                named_signals.append((sig_name, pts, label))

        # List-returning signals (balance sheet, FCF yield, growth)
        def _name_for(pts: int, base_name: str, high_name: str, idx: int) -> str:
            """Return canonical signal name, using high_name for the second entry."""
            return high_name if idx > 0 else base_name

        for idx, (pts, label) in enumerate(balance_sheet_sigs):
            if pts != 0 and label:
                name = "balance_sheet" if pts > 0 else "high_leverage"
                named_signals.append((name, pts, label))

        for pts, label in fcf_yield_sigs:
            if pts != 0 and label:
                name = "fcf_yield_high" if pts > 1 else "fcf_yield_base"
                named_signals.append((name, pts, label))

        for pts, label in revenue_growth_sigs:
            if pts != 0 and label:
                name = "revenue_high" if pts > 1 else "revenue_growth"
                named_signals.append((name, pts, label))

        for pts, label in earnings_growth_sigs:
            if pts != 0 and label:
                name = "earnings_high" if pts > 1 else "earnings_growth"
                named_signals.append((name, pts, label))

        # 52w range returns its own name derivation
        for pts, label in range_sigs:
            if pts != 0 and label:
                name = "lower_band" if pts > 0 else "near_52w_high"
                named_signals.append((name, pts, label))

        # ── Aggregate ──────────────────────────────────────────────────────
        total_score  = sum(pts for _, pts, _ in named_signals)
        signals      = [label for _, _, label in named_signals]
        active_names = [name for name, _, _ in named_signals]
        lower_band_active = "lower_band" in active_names

        return {
            "score":           total_score,
            "quality_score":   total_score,
            "signals":         signals,
            "quality_signals": list(signals),
            "tier":            _tier(total_score),
            "quality_tier":    _quality_tier(total_score),
            "type":            _infer_opportunity_type(active_names, lower_band_active),
        }
