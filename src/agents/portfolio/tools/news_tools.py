from typing import Dict, List


def _score_sentiment(headline: str) -> str:
    """Classify a headline as positive, negative, or neutral using VADER."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    score = SentimentIntensityAnalyzer().polarity_scores(headline)["compound"]
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def _fetch_live(ticker: str, max_articles: int = 5) -> List[Dict[str, str]]:
    """
    Fetch recent news headlines from Yahoo Finance via yfinance and
    score their sentiment with VADER (offline, no API key required).
    """
    import yfinance as yf

    raw = yf.Ticker(ticker).news or []
    articles = []
    for item in raw[:max_articles]:
        headline = item.get("title", "").strip()
        if not headline:
            continue
        articles.append({
            "headline": headline,
            "sentiment": _score_sentiment(headline),
        })
    return articles


def get_news(ticker: str) -> List[Dict[str, str]]:
    """
    Return live news articles with VADER sentiment labels for a given ticker.
    Source: Yahoo Finance (yfinance). Returns an empty list if unavailable.
    """
    try:
        return _fetch_live(ticker)
    except Exception:
        return []
