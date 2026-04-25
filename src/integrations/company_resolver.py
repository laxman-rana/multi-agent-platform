from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

import yfinance as yf

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "a",
    "an",
    "buy",
    "can",
    "for",
    "give",
    "hello",
    "help",
    "hi",
    "i",
    "me",
    "please",
    "recommend",
    "recommendation",
    "scan",
    "should",
    "stock",
    "stocks",
    "tell",
    "the",
    "today",
    "what",
}

def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _extract_company_phrases(message: str) -> list[str]:
    normalized_message = _normalize_text(message)
    if not normalized_message:
        return []

    tokens = [token for token in normalized_message.split() if token not in _STOP_WORDS]
    if not tokens:
        return []

    phrases: list[str] = []
    seen: set[str] = set()

    full_phrase = " ".join(tokens)
    if full_phrase:
        seen.add(full_phrase)
        phrases.append(full_phrase)

    for size in (3, 2, 1):
        for index in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[index:index + size])
            if phrase and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)

    return phrases


def _search_yahoo(query: str, max_results: int = 5) -> list[dict]:
    search = yf.Search(query=query, max_results=max_results)
    quotes = getattr(search, "quotes", None) or []
    return [quote for quote in quotes if isinstance(quote, dict)]


def _score_quote(query: str, quote: dict) -> float:
    symbol = str(quote.get("symbol") or "").upper()
    short_name = str(quote.get("shortname") or quote.get("longname") or "")
    normalized_query = _normalize_text(query)
    normalized_name = _normalize_text(short_name)
    normalized_symbol = _normalize_text(symbol)

    score = 0.0
    if quote.get("quoteType") == "EQUITY":
        score += 2.0
    if normalized_name:
        score += SequenceMatcher(None, normalized_query, normalized_name).ratio() * 4.0
    if normalized_symbol:
        score += SequenceMatcher(None, normalized_query, normalized_symbol).ratio() * 2.0
    if normalized_query and normalized_query in normalized_name:
        score += 1.5
    if "." not in symbol and "-" not in symbol:
        score += 0.25
    return score


def _resolve_with_yahoo(phrases: list[str]) -> list[str]:
    ranked: list[tuple[float, str]] = []
    seen: set[str] = set()

    for phrase in phrases:
        try:
            quotes = _search_yahoo(phrase)
        except Exception as exc:
            logger.warning("Yahoo resolver failed for query '%s': %s", phrase, exc)
            continue

        if not quotes:
            continue

        best_quote = max(quotes, key=lambda quote: _score_quote(phrase, quote))
        symbol = str(best_quote.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            continue

        score = _score_quote(phrase, best_quote)
        if score < 3.2:
            continue

        seen.add(symbol)
        ranked.append((score, symbol))

    ranked.sort(reverse=True)
    return [symbol for _, symbol in ranked]


def resolve_company_names(message: str) -> list[str]:
    """Resolve company names in free text to ticker symbols.

    Primary strategy:
    - extract likely company phrases from the user message
    - query Yahoo Finance search dynamically
    - rank candidate quotes and keep the strongest equity matches

    This resolver is intentionally dynamic only. If Yahoo search cannot resolve
    the name, the supervisor can still reason over the original natural-language
    request without local hardcoded aliases.
    """
    phrases = _extract_company_phrases(message)
    if not phrases:
        return []

    return _resolve_with_yahoo(phrases)
