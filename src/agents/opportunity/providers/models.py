from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    price: float
    change_pct: float
    volatility: float
    pe_ratio: float | None = None
    forward_pe: float | None = None
    market_cap: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    sector: str = "Unknown"
    volume: int = 0
    avg_volume: int = 0
    analyst_rating: str = "none"
    analyst_count: int = 0
    analyst_target: float | None = None
    profit_margins: float | None = None
    operating_margins: float | None = None
    return_on_equity: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    vol_pressure: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["52w_high"] = data.pop("high_52w")
        data["52w_low"] = data.pop("low_52w")
        return data


@dataclass(frozen=True)
class NewsSnapshot:
    ticker: str
    headlines: List[str] = field(default_factory=list)
    provider: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
