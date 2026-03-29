"""
rebalance_tools.py
------------------
Deterministic portfolio-level action engine.

Analyses completed per-stock decisions alongside sector allocation and
concentration metrics to produce a single `portfolio_action` recommendation
that answers the question the per-ticker loop cannot:

    "What should we do with the PORTFOLIO as a whole?"

No LLM involved — all logic is rule-based and reproducible.
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum acceptable single-sector weight for a moderate-risk investor.
_SECTOR_TARGET_MAX_PCT = 60.0

# If the top sector exceeds this level it's flagged as critically overweight.
_SECTOR_CRITICAL_PCT = 80.0

# Minimum number of sectors for adequate diversification.
_MIN_SECTORS = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_portfolio_action(
    sector_allocation: Dict[str, float],
    decisions: Dict[str, Dict[str, Any]],
    risk_metrics: Dict[str, Any],
    portfolio: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Derive a portfolio-level action from current state.

    Parameters
    ----------
    sector_allocation : sector → current weight (%)
    decisions         : ticker → {action, confidence, reason, gain_pct}
    risk_metrics      : output of calculate_risk()
    portfolio         : raw positions list (for sector membership lookup)

    Returns
    -------
    dict with keys:
        rebalance          bool        — True when action is recommended
        reduce_sector      str | None  — overweight sector to trim
        current_exposure   float|None  — current sector weight (%)
        target_exposure    str | None  — e.g. "60%"
        priority_exits     list[str]   — tickers in reduce_sector already flagged EXIT
        add_diversification bool       — True when < _MIN_SECTORS sectors present
        missing_sectors    list[str]   — broad sectors that are absent
        summary            str         — one human-readable sentence
    """
    action: Dict[str, Any] = {
        "rebalance":           False,
        "reduce_sector":       None,
        "current_exposure":    None,
        "target_exposure":     None,
        "priority_exits":      [],
        "add_diversification": False,
        "missing_sectors":     [],
        "summary":             "Portfolio allocation is within acceptable bounds.",
    }

    # ── 1. Sector concentration check ────────────────────────────────────
    if sector_allocation:
        top_sector = max(sector_allocation, key=lambda s: sector_allocation[s])
        top_pct    = sector_allocation[top_sector]

        if top_pct > _SECTOR_TARGET_MAX_PCT:
            action["rebalance"]        = True
            action["reduce_sector"]    = top_sector
            action["current_exposure"] = round(top_pct, 1)
            action["target_exposure"]  = f"{int(_SECTOR_TARGET_MAX_PCT)}%"

            # Tickers in the overweight sector that DecisionAgent already flagged EXIT
            sector_map: Dict[str, str] = {
                pos["ticker"]: pos["sector"] for pos in portfolio
            }
            action["priority_exits"] = [
                ticker
                for ticker, d in decisions.items()
                if d.get("action") == "EXIT"
                and sector_map.get(ticker) == top_sector
            ]

            severity = "critically" if top_pct >= _SECTOR_CRITICAL_PCT else "significantly"
            action["summary"] = (
                f"{top_sector} is {severity} overweight at {top_pct:.1f}% "
                f"(target ≤ {int(_SECTOR_TARGET_MAX_PCT)}%). "
                + (
                    f"Priority exits: {', '.join(action['priority_exits'])}."
                    if action["priority_exits"]
                    else "Consider trimming positions to rebalance."
                )
            )

    # ── 2. Diversification check ──────────────────────────────────────────
    _BROAD_SECTORS = [
        "Technology", "Healthcare", "Financials", "Consumer Discretionary",
        "Consumer Staples", "Industrials", "Energy", "Utilities",
        "Real Estate", "Materials", "Communication Services",
    ]
    present = set(sector_allocation.keys())
    if len(present) < _MIN_SECTORS:
        action["add_diversification"] = True
        action["missing_sectors"] = [s for s in _BROAD_SECTORS if s not in present][:4]
        diversify_note = (
            f"Only {len(present)} sector(s) present — consider adding: "
            + ", ".join(action["missing_sectors"]) + "."
        )
        action["summary"] = action["summary"].rstrip(".") + ". " + diversify_note

    return action
