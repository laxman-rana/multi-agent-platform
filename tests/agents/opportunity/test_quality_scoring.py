import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.opportunity.engines.signal_engine import SignalEngine
from src.agents.opportunity.nodes.alpha_scanner_agent import AlphaScannerAgent
from src.agents.opportunity.nodes.decision_node import DecisionNode
from src.agents.opportunity.nodes.news_node import _apply_fundamental_risk_penalty
from src.agents.opportunity.state import OpportunityState


def _market_data(**overrides):
    base = {
        "ticker": "TEST",
        "price": 100.0,
        "change_pct": 1.0,
        "volatility": 0.22,
        "pe_ratio": 28.0,
        "forward_pe": 24.0,
        "52w_high": 120.0,
        "52w_low": 80.0,
        "sector": "Technology",
        "market_cap": 50e9,
        "volume": 2_000_000,
        "avg_volume": 1_500_000,
        "analyst_rating": "buy",
        "analyst_count": 12,
        "analyst_target": 120.0,
        "profit_margins": 0.18,
        "operating_margins": 0.22,
        "return_on_equity": 0.25,
        "debt_to_equity": 0.45,
        "free_cash_flow": 5e9,
        "revenue_growth": 0.12,
        "earnings_growth": 0.15,
        "vol_pressure": "buying",
    }
    base.update(overrides)
    return base


class SignalEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SignalEngine()

    def test_high_quality_company_scores_above_weak_company(self):
        high_quality = self.engine.score(_market_data())
        weak_company = self.engine.score(
            _market_data(
                ticker="WEAK",
                forward_pe=58.0,
                profit_margins=-0.02,
                operating_margins=0.01,
                return_on_equity=0.04,
                debt_to_equity=2.6,
                free_cash_flow=-1e9,
                revenue_growth=-0.03,
                earnings_growth=-0.08,
                analyst_rating="hold",
                analyst_count=1,
                analyst_target=None,
                change_pct=4.0,
                volume=6_000_000,
            )
        )
        self.assertGreater(high_quality["quality_score"], weak_company["quality_score"])
        self.assertEqual(high_quality["quality_tier"], "elite")
        self.assertEqual(weak_company["quality_tier"], "avoid")

    def test_balance_sheet_quality_beats_volume_driven_low_quality_setup(self):
        strong_balance_sheet = self.engine.score(_market_data(ticker="SAFE", volume=1_000_000, avg_volume=1_000_000))
        volume_driven = self.engine.score(
            _market_data(
                ticker="LOUD",
                profit_margins=0.02,
                operating_margins=0.03,
                return_on_equity=0.05,
                debt_to_equity=2.2,
                free_cash_flow=-5e8,
                revenue_growth=0.01,
                earnings_growth=-0.02,
                forward_pe=50.0,
                change_pct=6.0,
                volume=8_000_000,
                avg_volume=2_000_000,
            )
        )
        self.assertGreater(strong_balance_sheet["quality_score"], volume_driven["quality_score"])

    def test_missing_metrics_degrade_gracefully(self):
        result = self.engine.score(
            _market_data(
                profit_margins=None,
                operating_margins=None,
                return_on_equity=None,
                debt_to_equity=None,
                free_cash_flow=None,
                revenue_growth=None,
                earnings_growth=None,
                analyst_target=None,
            )
        )
        self.assertIn("quality_score", result)
        self.assertIsInstance(result["quality_signals"], list)

    def test_extreme_valuation_and_near_high_are_penalized(self):
        result = self.engine.score(
            _market_data(**{
                "price": 118.0,
                "52w_high": 120.0,
                "52w_low": 80.0,
                "forward_pe": 60.0,
            })
        )
        self.assertLess(result["quality_score"], 9)
        self.assertIn("margin of safety", " ".join(result["quality_signals"]).lower())


class NewsPenaltyTests(unittest.TestCase):
    def test_negative_news_penalty_reduces_quality_score(self):
        state = OpportunityState(
            candidates=["RISK"],
            market_data={"RISK": _market_data(ticker="RISK", change_pct=-12.0)},
            signals={
                "RISK": {
                    "score": 6,
                    "quality_score": 6,
                    "tier": "strong_buy",
                    "quality_tier": "high_quality",
                    "signals": ["Healthy profit margin"],
                    "quality_signals": ["Healthy profit margin"],
                }
            },
            news_sentiment={"RISK": {"sentiment": "negative", "catalyst": "Guidance cut", "headline_count": 3}},
        )
        updated = _apply_fundamental_risk_penalty(state)
        self.assertEqual(updated.signals["RISK"]["quality_score"], 4)
        self.assertEqual(updated.signals["RISK"]["quality_tier"], "watchlist")


class PipelineSelectionTests(unittest.TestCase):
    @patch("src.agents.opportunity.nodes.alpha_scanner_agent._fetch_extended")
    def test_only_top_quality_names_become_candidates(self, fetch_mock):
        dataset = {
            "ELITE": _market_data(ticker="ELITE"),
            "MID": _market_data(
                ticker="MID",
                profit_margins=0.10,
                operating_margins=0.11,
                return_on_equity=0.12,
                debt_to_equity=0.9,
                free_cash_flow=1e8,
                revenue_growth=0.06,
                earnings_growth=0.04,
                analyst_rating="hold",
                analyst_count=0,
                volume=1_200_000,
                avg_volume=1_000_000,
            ),
            "LOUD": _market_data(
                ticker="LOUD",
                profit_margins=0.01,
                operating_margins=0.02,
                return_on_equity=0.03,
                debt_to_equity=2.3,
                free_cash_flow=-1e9,
                revenue_growth=-0.02,
                earnings_growth=-0.03,
                forward_pe=55.0,
                change_pct=8.0,
                volume=10_000_000,
                avg_volume=1_000_000,
            ),
        }
        fetch_mock.side_effect = lambda ticker: dataset[ticker]

        state = OpportunityState(watchlist=["ELITE", "MID", "LOUD"])
        final_state = AlphaScannerAgent().run(state)

        self.assertIn("ELITE", final_state.candidates)
        self.assertIn("MID", final_state.candidates)
        self.assertNotIn("LOUD", final_state.candidates)

    @patch("src.agents.opportunity.nodes.decision_node.get_provider")
    @patch("src.agents.opportunity.nodes.decision_node.OpportunityDecisionAgent.run")
    def test_final_output_includes_quality_fields(self, decision_run_mock, get_provider_mock):
        get_provider_mock.return_value = SimpleNamespace(max_concurrency=1)
        decision_run_mock.return_value = {
            "ticker": "ELITE",
            "score": 8,
            "quality_score": 8,
            "quality_tier": "elite",
            "type": "compounder",
            "action": "BUY",
            "confidence": "high",
            "entry_quality": "strong",
            "reason": "Elite quality score with durable profitability and disciplined valuation.",
        }

        state = OpportunityState(
            watchlist=["ELITE"],
            candidates=["ELITE"],
            market_data={"ELITE": _market_data(ticker="ELITE")},
            signals={
                "ELITE": {
                    "score": 8,
                    "quality_score": 8,
                    "tier": "strong_buy",
                    "quality_tier": "elite",
                    "type": "compounder",
                    "signals": ["Healthy profit margin"],
                    "quality_signals": ["Healthy profit margin"],
                    "opportunity_score": 0.88,
                }
            },
        )

        final_state = DecisionNode().run(state)
        self.assertEqual(len(final_state.buy_opportunities), 1)
        opportunity = final_state.buy_opportunities[0]
        self.assertEqual(opportunity["quality_score"], 8)
        self.assertEqual(opportunity["quality_tier"], "elite")
        self.assertIn("quality_signals", opportunity)
        self.assertEqual(opportunity["action"], "BUY SIGNAL")


if __name__ == "__main__":
    unittest.main()
