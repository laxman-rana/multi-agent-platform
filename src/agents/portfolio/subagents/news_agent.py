import logging

from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.news_tools import get_news
from src.agents.portfolio.subagents.market_agent import VOLATILITY_THRESHOLD
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)


class NewsAgent:
    """
    Optional node — only reached when the conditional edge routes here
    because at least one ticker exceeded the volatility threshold.

    Fetches news and sentiment for every high-volatility position.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        news: dict = {}
        telemetry = get_telemetry_logger()

        for ticker, insight in state.stock_insights.items():
            if insight.get("volatility", 0) > VOLATILITY_THRESHOLD:
                articles = get_news(ticker) or []
                telemetry.log_tool_usage(
                    "get_news",
                    {"ticker": ticker, "volatility": insight.get("volatility")},
                    {"article_count": len(articles), "sentiments": [a["sentiment"] for a in articles]},
                )
                if articles:
                    news[ticker] = articles
                    logger.info("[NewsAgent] %s: %d articles fetched", ticker, len(articles))
                else:
                    logger.info("[NewsAgent] %s: 0 articles — skipping (no value added)", ticker)

        state.news = news

        if not news:
            logger.info("[NewsAgent] No articles fetched for any ticker — news context omitted.")

        return state
