import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from src.agents.opportunity.engines.prefilter_engine import PreFilterEngine
from src.agents.opportunity.engines.signal_engine import SignalEngine
from src.agents.opportunity.providers.base import MarketDataProvider
from src.agents.opportunity.state import OpportunityState
from src.agents.opportunity.services.cooldown import CooldownPolicy
from src.agents.opportunity.services.portfolio_overlay import PortfolioOverlayPolicy
from src.agents.opportunity.services.ranking import CandidateRanker
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)

_COOLDOWN_MINUTES: int = 30
_COOLDOWN_UNIT: str = "minutes"
_CANDIDATE_MIN_SCORE: int = 5   # Fix 5: raised from 4 — score 4 reachable
                                # with profitability(+2) + one growth signal(+1)
                                # + lower_band(+1): cheap grower with no other
                                # quality markers (value-trap territory).
                                # Score 5 forces at least two independent quality
                                # signals beyond basic profitability.
_MAX_RANKED_CANDIDATES: int = 12
_MAX_SECTOR_EXPOSURE: float = 60.0
_MAX_POSITION_WEIGHT: float = 10.0
_MAX_FETCH_WORKERS: int = 10
_MAX_CONCURRENT_BUYS: int = 3
_MAX_SIGNAL_SCORE: float = 15.0  # updated: new theoretical max with tiered growth + FCF yield


class AlphaScannerAgent:
    """
    Orchestrates the quality-first scan for a single cycle:

      1. Parallel market-data fetch
      2. PreFilterEngine
      3. SignalEngine quality scoring
      4. Rank + threshold candidate selection
      5. Portfolio overlay guards
    """

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        prefilter: Optional[PreFilterEngine] = None,
        signal_engine: Optional[SignalEngine] = None,
        ranker: Optional[CandidateRanker] = None,
        cooldown_policy: Optional[CooldownPolicy] = None,
        portfolio_overlay_policy: Optional[PortfolioOverlayPolicy] = None,
        max_fetch_workers: int = _MAX_FETCH_WORKERS,
    ) -> None:
        self._market_data_provider = market_data_provider
        self._prefilter = prefilter or PreFilterEngine()
        self._signal_engine = signal_engine or SignalEngine()
        self._ranker = ranker or CandidateRanker(
            max_signal_score=_MAX_SIGNAL_SCORE,
            candidate_min_score=_CANDIDATE_MIN_SCORE,
            max_ranked_candidates=_MAX_RANKED_CANDIDATES,
        )
        self._cooldown_policy = cooldown_policy or CooldownPolicy(
            cooldown_minutes=_COOLDOWN_MINUTES,
            cooldown_unit=_COOLDOWN_UNIT,
        )
        self._portfolio_overlay_policy = portfolio_overlay_policy or PortfolioOverlayPolicy(
            max_sector_exposure=_MAX_SECTOR_EXPOSURE,
            max_position_weight=_MAX_POSITION_WEIGHT,
        )
        self._max_fetch_workers = max_fetch_workers

    def _fetch_market_data(self, watchlist: List[str], state: OpportunityState) -> None:
        workers = min(len(watchlist), self._max_fetch_workers) if watchlist else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._market_data_provider.fetch_one, ticker): ticker
                for ticker in watchlist
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    snapshot = future.result()
                    state.market_data[ticker] = snapshot.to_dict()
                except Exception as exc:
                    logger.warning("[AlphaScannerAgent] Fetch failed for %s: %s", ticker, exc)
                    state.scan_errors[ticker] = str(exc)

    def run(self, state: OpportunityState) -> OpportunityState:
        telemetry = get_telemetry_logger()
        step_latencies: Dict[str, float] = {}

        telemetry.log_event(
            "scan_start",
            {
                "watchlist_size": len(state.watchlist),
                "provider": type(self._market_data_provider).__name__,
            },
        )
        logger.info("[AlphaScannerAgent] Quality scan started — %d tickers", len(state.watchlist))

        fetch_t0 = time.monotonic()
        self._fetch_market_data(state.watchlist, state)
        step_latencies["fetch_ms"] = round((time.monotonic() - fetch_t0) * 1000, 1)

        prefilter_t0 = time.monotonic()
        state.prefiltered = [
            ticker
            for ticker, data in state.market_data.items()
            if self._prefilter.pre_filter(data)
        ]
        step_latencies["prefilter_ms"] = round((time.monotonic() - prefilter_t0) * 1000, 1)

        telemetry.log_event(
            "prefilter_complete",
            {
                "total": len(state.market_data),
                "passed": len(state.prefiltered),
                "filtered_out": len(state.market_data) - len(state.prefiltered),
                "latency_ms": step_latencies["prefilter_ms"],
            },
        )

        signal_t0 = time.monotonic()
        for ticker in state.prefiltered:
            signal = self._signal_engine.score(state.market_data[ticker])
            signal["opportunity_score"] = self._ranker.compute_opportunity_score(
                ticker,
                state.market_data[ticker],
                signal,
                state.recent_signal_context,
            )
            state.signals[ticker] = signal
        step_latencies["signal_ms"] = round((time.monotonic() - signal_t0) * 1000, 1)

        telemetry.log_event(
            "signals_generated",
            {
                "scored": len(state.signals),
                "elite_tier": sum(1 for s in state.signals.values() if s.get("quality_tier") == "elite"),
                "high_quality_tier": sum(1 for s in state.signals.values() if s.get("quality_tier") == "high_quality"),
                "watchlist_tier": sum(1 for s in state.signals.values() if s.get("quality_tier") == "watchlist"),
                "avoid_tier": sum(1 for s in state.signals.values() if s.get("quality_tier") == "avoid"),
                "latency_ms": step_latencies["signal_ms"],
            },
        )

        ranked_pool = self._ranker.rank_candidates(state.signals)
        candidates: List[str] = []
        skipped_cooldown: List[str] = []

        for ticker, signal in ranked_pool:
            market_data = state.market_data.get(ticker, {})
            if not self._cooldown_policy.is_cooled_down(ticker, state.recent_signals):
                if self._cooldown_policy.is_fresh_despite_cooldown(
                    ticker,
                    market_data,
                    state.recent_signal_context,
                ):
                    signal["override_reason"] = "fresh_catalyst"
                else:
                    skipped_cooldown.append(ticker)
                    continue
            candidates.append(ticker)

        state.candidates = candidates
        state.skipped_cooldown = skipped_cooldown
        state.blocked_no_cash = []

        telemetry.log_event(
            "candidates_filtered",
            {
                "candidate_count": len(candidates),
                "ranked_pool": len(ranked_pool),
                "filtered_count": len(state.signals) - len(ranked_pool),
                "skipped_cooldown": skipped_cooldown,
            },
        )

        state = self._portfolio_overlay_policy.apply(state)

        logger.info(
            "[AlphaScannerAgent] Ranked %d quality names, %d candidate(s) proceeding",
            len(ranked_pool),
            len(state.candidates),
        )
        return state
