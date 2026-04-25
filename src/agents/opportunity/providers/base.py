from abc import ABC, abstractmethod
from typing import Dict, List

from src.agents.opportunity.providers.models import MarketSnapshot, NewsSnapshot


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_one(self, ticker: str) -> MarketSnapshot:
        raise NotImplementedError

    def fetch_many(self, tickers: List[str]) -> Dict[str, MarketSnapshot]:
        return {ticker: self.fetch_one(ticker) for ticker in tickers}


class NewsProvider(ABC):
    @abstractmethod
    def fetch_headlines(self, ticker: str, limit: int) -> NewsSnapshot:
        raise NotImplementedError
