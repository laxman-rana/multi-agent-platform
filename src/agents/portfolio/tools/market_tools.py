import math
from typing import Any, Dict



def _fetch_live(ticker: str) -> Dict[str, Any]:
    """
    Fetch real-time market data from Yahoo Finance via yfinance.

    Fields returned match the shape expected by MarketAgent:
      price, change_pct, volatility (annualised, 30-day),
      pe_ratio, 52w_high, 52w_low
    """
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")

    if not price:
        raise ValueError(f"No price data returned for {ticker}")

    change_pct = (
        round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
    )

    # Annualised volatility from 30 trading days of daily returns
    hist = t.history(period="30d")
    if len(hist) >= 2:
        daily_returns = hist["Close"].pct_change().dropna()
        volatility = round(float(daily_returns.std() * math.sqrt(252)), 4)
    else:
        volatility = 0.20  # sensible default if history unavailable

    return {
        "price":      round(float(price), 2),
        "change_pct": change_pct,
        "volatility": volatility,
        # None when yfinance doesn't have data — displayed as N/A in the LLM prompt
        # so the model does not invent a value.
        "pe_ratio":   info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "52w_high":   info.get("fiftyTwoWeekHigh", 0.0),
        "52w_low":    info.get("fiftyTwoWeekLow", 0.0),
    }


def get_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Return live market data for a given ticker via Yahoo Finance (yfinance).
    Raises ValueError if no price data is available.
    """
    return _fetch_live(ticker)
