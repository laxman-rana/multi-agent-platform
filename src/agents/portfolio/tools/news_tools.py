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


def _fetch_rss(ticker: str, max_articles: int = 5) -> List[Dict[str, str]]:
    """
    Fallback: fetch headlines via RSS (Google News + Yahoo Finance RSS feeds).
    Used when yfinance returns no articles. Scores sentiment with VADER so the
    output format is identical to _fetch_live.
    """
    import feedparser

    sources = [
        f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
    ]
    for url in sources:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        entries = feed.get("entries", [])
        if not entries:
            continue
        articles = []
        for entry in entries[:max_articles]:
            headline = (entry.get("title") or "").strip()
            if not headline:
                continue
            articles.append({
                "headline": headline,
                "sentiment": _score_sentiment(headline),
            })
        if articles:
            return articles
    return []


def get_news(ticker: str) -> List[Dict[str, str]]:
    """
    Return live news articles with VADER sentiment labels for a given ticker.

    Strategy:
      1. Try yfinance (primary — no API key required).
      2. Fall back to RSS (Google News / Yahoo Finance) if yfinance returns empty.
      3. Return [] if both sources fail or are empty.
    """
    try:
        articles = _fetch_live(ticker)
        if articles:
            return articles
    except Exception:
        pass

    try:
        return _fetch_rss(ticker)
    except Exception:
        return []


def compute_news_score(articles: List[Dict[str, str]]) -> int:
    """
    Aggregate article sentiments into a single signal: +1, 0, or -1.

    Each article contributes:
      positive  → +1
      neutral   →  0
      negative  → -1

    The raw sum is normalised to the range [-1, +1] via sign():
      sum > 0  → +1
      sum == 0 →  0
      sum < 0  → -1

    Returns 0 when the article list is empty.
    """
    if not articles:
        return 0
    _MAP = {"positive": 1, "neutral": 0, "negative": -1}
    total = sum(_MAP.get(a.get("sentiment", "neutral"), 0) for a in articles)
    if total > 0:
        return 1
    if total < 0:
        return -1
    return 0
