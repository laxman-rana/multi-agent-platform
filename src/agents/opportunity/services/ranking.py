from typing import Any, Dict, List, Tuple


class CandidateRanker:
    def __init__(
        self,
        max_signal_score: float = 15.0,  # updated to match new signal_engine ceiling
        candidate_min_score: int = 5,    # updated: Fix 5 — raised from 4
        max_ranked_candidates: int = 12,
    ) -> None:
        self._max_signal_score = max_signal_score
        self._candidate_min_score = candidate_min_score
        self._max_ranked_candidates = max_ranked_candidates

    def compute_opportunity_score(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        signal: Dict[str, Any],
        recent_signal_context: Dict[str, Any],
    ) -> float:
        quality_score = signal.get("quality_score", signal.get("score", 0))
        quality_norm = max(0.0, min(1.0, quality_score / self._max_signal_score))

        forward_pe = market_data.get("forward_pe")
        if forward_pe and forward_pe > 0:
            valuation_norm = max(0.0, min(1.0, (35.0 - min(forward_pe, 35.0)) / 35.0))
        else:
            valuation_norm = 0.4

        analyst_target = market_data.get("analyst_target")
        price = market_data.get("price", 0.0)
        if analyst_target and price and price > 0:
            analyst_upside_norm = min(1.0, max(0.0, ((analyst_target - price) / price) / 0.4))
        else:
            analyst_upside_norm = 0.0

        freshness = 0.0 if ticker in recent_signal_context else 1.0

        return round(
            0.7 * quality_norm
            + 0.15 * valuation_norm
            + 0.1 * analyst_upside_norm
            + 0.05 * freshness,
            4,
        )

    def rank_candidates(
        self,
        signals: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        eligible = [
            (ticker, signal)
            for ticker, signal in signals.items()
            if signal.get("quality_score", signal.get("score", 0)) >= self._candidate_min_score
        ]
        ranked = sorted(
            eligible,
            key=lambda item: (item[1].get("opportunity_score", 0.0), item[1].get("quality_score", 0)),
            reverse=True,
        )
        return ranked[: self._max_ranked_candidates]
