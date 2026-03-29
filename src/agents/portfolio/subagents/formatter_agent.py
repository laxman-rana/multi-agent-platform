from src.agents.portfolio.state import PortfolioState


_ACTION_ICON = {"EXIT": "🔴", "HOLD": "🟡", "DOUBLE_DOWN": "🟢"}
_SENTIMENT_ICON = {"positive": "↑", "negative": "↓", "neutral": "→"}


class FormatterAgent:
    """
    Final node in the graph.
    Produces a structured, human-readable portfolio analysis report
    and writes it to state.final_output.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        lines = self._build_report(state)
        state.final_output = "\n".join(lines)
        print("  Report generated.")
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_report(self, state: PortfolioState) -> list:
        lines = []
        lines += self._header(state)
        lines += self._summary(state)
        lines += self._sector_section(state)
        lines += self._decisions_section(state)
        if state.news:
            lines += self._news_section(state)
        lines += self._critic_section(state)
        lines += self._footer()
        return lines

    def _header(self, state: PortfolioState) -> list:
        p = state.user_profile
        return [
            "=" * 70,
            "  PORTFOLIO ANALYSIS REPORT",
            "=" * 70,
            f"  Investor   : {p.get('name')}",
            f"  Risk Level : {p.get('risk_tolerance', '').capitalize()}",
            f"  Horizon    : {p.get('investment_horizon')}",
            "",
        ]

    def _summary(self, state: PortfolioState) -> list:
        r = state.risk_metrics
        return [
            "── PORTFOLIO SUMMARY ──────────────────────────────────────────────",
            f"  Total Value      : ${r.get('total_portfolio_value', 0):>12,.2f}",
            f"  Unrealized P&L   : ${r.get('unrealized_pnl', 0):>+12,.2f}  ({r.get('unrealized_pnl_pct', 0):+.2f}%)",
            f"  Weighted Vol.    : {r.get('weighted_volatility', 0):.2%}",
            f"  Concentration    : {r.get('concentration_risk', '').upper()}",
            "",
        ]

    def _sector_section(self, state: PortfolioState) -> list:
        lines = ["── SECTOR ALLOCATION ───────────────────────────────────────────────"]
        for sector, pct in sorted(state.sector_allocation.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 5)
            lines.append(f"  {sector:<20}  {pct:5.1f}%  {bar}")
        lines.append("")
        return lines

    def _decisions_section(self, state: PortfolioState) -> list:
        lines = ["── STOCK DECISIONS ─────────────────────────────────────────────────"]
        for ticker, decision in state.decisions.items():
            icon = _ACTION_ICON.get(decision["action"], "⚪")
            insight = state.stock_insights.get(ticker, {})
            critic_entry = state.critic_feedback.get("per_ticker", {}).get(ticker, {})
            issues = critic_entry.get("issues", [])

            lines.append(
                f"  {icon}  {ticker:<6}  {decision['action']:<13}"
                f"[{decision['confidence'].upper()}]  "
                f"PnL: {decision.get('gain_pct', 0):+.1f}%  "
                f"@ ${insight.get('price', 0):.2f}"
            )
            lines.append(f"       → {decision['reason']}")

            for issue in issues:
                lines.append(f"       ⚠  {issue}")

        lines.append("")
        return lines

    def _news_section(self, state: PortfolioState) -> list:
        lines = ["── NEWS  (High-Volatility Tickers) ─────────────────────────────────"]
        for ticker, articles in state.news.items():
            lines.append(f"  {ticker}:")
            for article in articles[:3]:
                icon = _SENTIMENT_ICON.get(article.get("sentiment", "neutral"), "→")
                lines.append(f"    {icon}  {article['headline']}")
        lines.append("")
        return lines

    def _critic_section(self, state: PortfolioState) -> list:
        feedback = state.critic_feedback
        lines = []
        if feedback.get("warnings"):
            lines.append("── CRITIC WARNINGS ─────────────────────────────────────────────────")
            for w in feedback["warnings"]:
                lines.append(f"  ⚠  {w}")
            lines.append("")
        approved = feedback.get("approved", True)
        lines.append(f"  Overall Review : {'✅  APPROVED' if approved else '⛔  NEEDS REVIEW — low-confidence decisions present'}")
        return lines

    def _footer(self) -> list:
        return [
            "=" * 70,
            "  ⚠  DISCLAIMER: AI-generated decision support only.",
            "     Consult a licensed financial advisor before making any trades.",
            "=" * 70,
            "",
        ]
