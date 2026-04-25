import os

from src.agents.opportunity.providers.base import MarketDataProvider, NewsProvider
from src.agents.opportunity.providers.yahoo import YahooMarketDataAdapter, YahooNewsAdapter


def create_market_data_provider() -> MarketDataProvider:
    provider_name = os.getenv("OPPORTUNITY_MARKET_DATA_PROVIDER", "yahoo").lower()
    if provider_name == "yahoo":
        return YahooMarketDataAdapter()
    raise ValueError(f"Unsupported opportunity market-data provider '{provider_name}'")


def create_news_provider() -> NewsProvider:
    provider_name = os.getenv("OPPORTUNITY_NEWS_PROVIDER", "yahoo").lower()
    if provider_name == "yahoo":
        return YahooNewsAdapter()
    raise ValueError(f"Unsupported opportunity news provider '{provider_name}'")
