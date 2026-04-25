import math
from typing import Any, List

import yfinance as yf

from src.agents.opportunity.providers.base import MarketDataProvider, NewsProvider
from src.agents.opportunity.providers.models import MarketSnapshot, NewsSnapshot


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class YahooMarketDataAdapter(MarketDataProvider):
    def fetch_one(self, ticker: str) -> MarketSnapshot:
        t = yf.Ticker(ticker)
        info = t.info

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if not price:
            raise ValueError(f"No price data returned for {ticker}")

        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0

        hist = t.history(period="30d")
        if len(hist) >= 2:
            daily_returns = hist["Close"].pct_change().dropna()
            volatility = round(float(daily_returns.std() * math.sqrt(252)), 4)
        else:
            volatility = 0.20

        volume = info.get("regularMarketVolume", 0)
        avg_volume = info.get("averageVolume", 0)
        vol_pressure = "neutral"
        if volume and avg_volume and volume >= avg_volume * 1.5:
            vol_pressure = "buying" if change_pct > 0 else ("selling" if change_pct < 0 else "neutral")

        return MarketSnapshot(
            ticker=ticker,
            price=round(float(price), 2),
            change_pct=change_pct,
            volatility=volatility,
            pe_ratio=_coerce_float(info.get("trailingPE")),
            forward_pe=_coerce_float(info.get("forwardPE")),
            high_52w=_coerce_float(info.get("fiftyTwoWeekHigh")) or 0.0,
            low_52w=_coerce_float(info.get("fiftyTwoWeekLow")) or 0.0,
            sector=info.get("sector", "Unknown"),
            market_cap=_coerce_float(info.get("marketCap")) or 0.0,
            volume=volume,
            avg_volume=avg_volume,
            analyst_rating=(info.get("recommendationKey") or "none").lower(),
            analyst_count=info.get("numberOfAnalystOpinions") or 0,
            analyst_target=_coerce_float(info.get("targetMeanPrice")),
            profit_margins=_coerce_float(info.get("profitMargins")),
            operating_margins=_coerce_float(info.get("operatingMargins")),
            return_on_equity=_coerce_float(info.get("returnOnEquity")),
            debt_to_equity=(_coerce_float(info.get("debtToEquity")) or 0.0) / 100.0 if info.get("debtToEquity") is not None else None,
            free_cash_flow=_coerce_float(info.get("freeCashflow")),
            revenue_growth=_coerce_float(info.get("revenueGrowth")),
            earnings_growth=_coerce_float(info.get("earningsGrowth")),
            vol_pressure=vol_pressure,
        )


class YahooNewsAdapter(NewsProvider):
    def fetch_headlines(self, ticker: str, limit: int) -> NewsSnapshot:
        try:
            items = yf.Ticker(ticker).news or []
            titles: List[str] = []
            for item in items[:limit]:
                title = ((item.get("content") or {}).get("title") or item.get("title") or "")
                if title:
                    titles.append(title)
            return NewsSnapshot(ticker=ticker, headlines=titles, provider="yahoo")
        except Exception:
            return NewsSnapshot(ticker=ticker, headlines=[], provider="yahoo")
