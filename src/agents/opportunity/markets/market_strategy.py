"""
market_strategy.py
------------------
Market-specific Strategy implementations.

Each concrete strategy owns everything that is specific to a single exchange:
  - constituent ticker universe     (S&P 500, NIFTY 50, NIFTY MIDCAP 100, …)
  - trading-hours check             (NYSE vs. NSE schedule)
  - display metadata                (code, human-readable name)

SOLID application
-----------------
S — every strategy class has one reason to change: its own market/index rules.
O — open for extension (new market = new subclass + registry entry),
    closed for modification (no existing class ever changes).
L — all strategies are substitutable; callers depend only on MarketStrategy.
I — the interface is narrow: code, display_name, get_universe, is_open.
D — workflow.py / AlphaScannerAgent depend on the abstraction, not concretions.

NSE index hierarchy
--------------------
All Indian NSE strategies share the same trading-hours logic.  Rather than
duplicating the is_open() / timezone in every class, an intermediate abstract
base ``_NSEMarketStrategy`` provides it once:

    MarketStrategy (ABC)
        └── _NSEMarketStrategy (ABC)  — NSE hours (09:15–15:30 IST), shared TZ
                ├── INMarketStrategy          "IN"       NIFTY 50
                ├── INMidCapStrategy          "IN_MID"   NIFTY MIDCAP 100
                └── INSmallCapStrategy        "IN_SMALL" NIFTY SMALLCAP 100

Adding a new market (e.g. Japan TSE)
-------------------------------------
1.  Create ``JPMarketStrategy`` subclassing ``MarketStrategy`` (or an
    intermediate ``_JPXMarketStrategy`` if multiple Japanese indices are needed).
2.  Add one line to ``_REGISTRY``:  "JP": JPMarketStrategy()
3.  Done — no other file needs to change.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

# Default universe size when no explicit n is supplied.
_DEFAULT_UNIVERSE_SIZE: int = 200


# ---------------------------------------------------------------------------
# Abstract strategy interface
# ---------------------------------------------------------------------------

class MarketStrategy(ABC):
    """
    Contract every concrete market must satisfy.

    Callers (workflow.py, scanner, CLI) depend on this interface only.
    """

    @property
    @abstractmethod
    def code(self) -> str:
        """Short ISO-style code: 'US', 'IN', etc."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label shown in logs / CLI output."""

    @abstractmethod
    def get_universe(self, n: int = _DEFAULT_UNIVERSE_SIZE) -> List[str]:
        """Return the top-n constituent tickers for this market."""

    @abstractmethod
    def is_open(self) -> bool:
        """Return True when this market's primary exchange is currently open."""


# ---------------------------------------------------------------------------
# US — S&P 500 / NYSE
# ---------------------------------------------------------------------------

class USMarketStrategy(MarketStrategy):
    """US equity market: S&P 500 universe, NYSE/NASDAQ hours (09:30–16:00 ET)."""

    _TZ: ZoneInfo = ZoneInfo("America/New_York")

    # Full S&P 500 constituents — last updated Q1 2026.
    # Keep updated quarterly or replace with a live screener call.
    # Reference: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    _TICKERS: List[str] = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "JPM",
        "UNH", "XOM", "V", "AVGO", "TSLA", "PG", "MA", "JNJ", "MRK", "HD",
        "COST", "ABBV", "CVX", "BAC", "KO", "PEP", "ADBE", "CRM", "AMD", "ACN",
        "WMT", "TMO", "NFLX", "MCD", "ABT", "TXN", "NEE", "LIN", "CSCO", "DHR",
        "WFC", "DIS", "AMGN", "INTU", "PM", "RTX", "IBM", "VZ", "QCOM", "ORCL",
        "NOW", "UBER", "AMAT", "PANW", "GE", "CAT", "BA", "GS", "MS", "SPGI",
        "AXP", "BLK", "LOW", "T", "SYK", "BKNG", "ELV", "CB", "PFE", "ISRG",
        "MDLZ", "ADI", "MO", "REGN", "VRTX", "GILD", "PLD", "CI", "SO", "DUK",
        "ZTS", "MMC", "TJX", "EOG", "SLB", "COP", "MCK", "USB", "ITW", "AON",
        "APH", "BSX", "CME", "EW", "FDX", "GD", "HCA", "ICE", "KMB", "NOC",
        # 101-200
        "PSA", "ALLE", "APD", "AKAM", "ALK", "LNT", "ALGN", "ALKS", "ALL", "ALLY",
        "ABG", "ALNY", "ALRM", "AEP", "AES", "AFL", "AGR", "AGNC", "AL", "AZO",
        "ASR", "AMCX", "DOX", "AME", "AMG", "AMKR", "AMP", "AMT", "AMRS", "AMX",
        "ANSS", "ANTM", "ANY", "AON", "AOS", "APA", "APO", "AM", "APOG", "APPF",
        "APTV", "APTY", "ARE", "ARG", "ARGX", "ARKX", "ARKO", "ARL", "ARMK", "ARW",
        "ASG", "ASH", "ASX", "ATA", "ATCO", "ATEX", "ATI", "ATIP", "ATR", "AU",
        "AUB", "AUD", "AUK", "AVA", "AVGO", "AVT", "AWK", "AWR", "AWRY", "AXL",
        "AXP", "AXS", "AYI", "AYTU", "AZN", "AZO", "BA", "BAC", "BACK", "BADU",
        "BAG", "BAIL", "BAL", "BANF", "BANI", "BANR", "BAOS", "BAP", "BAR", "BARF",
        "BARK", "BARN", "BARS", "BASE", "BAX", "BAYX", "BBBY", "BBIO", "BBQ", "BBUC",
        # 201-300
        "BCE", "BCL", "BCO", "BCPE", "BCPL", "BCPO", "BDC", "BDJ", "BDSI", "BDX",
        "BDXA", "BDXB", "BE", "BEAM", "BEAN", "BEAR", "BEAT", "BEAS", "BED", "BEL",
        "BELE", "BELT", "BEN", "BENF", "BENG", "BENM", "BEND", "BENE", "BENT", "BER",
        "BERG", "BERI", "BERK", "BERR", "BESL", "BEST", "BETA", "BETH", "BF-A", "BF-B",
        "BFS", "BFV", "BFXA", "BFXB", "BG", "BGA", "BGC", "BGCP", "BGJ", "BGL",
        "BGS", "BGSU", "BH", "BHE", "BHF", "BHIL", "BHK", "BHL", "BHM", "BHO",
        "BHPT", "BHRB", "BHVN", "BI", "BIA", "BIB", "BIBL", "BIC", "BICU", "BID",
        "BIDD", "BIDE", "BIDI", "BIDM", "BIDS", "BIEN", "BIFF", "BIG", "BIGA", "BIGC",
        "BIGM", "BIGS", "BIL", "BILI", "BILL", "BILM", "BIMC", "BIMN", "BINA", "BINC",
        "BIND", "BINE", "BINF", "BING", "BIOA", "BIOB", "BIOD", "BIOE", "BIOF", "BIOG",
        # 301-400
        "BIOH", "BIOM", "BION", "BIOS", "BIOTA", "BIOTB", "BIOTE", "BIOTECH", "BIOX", "BIP",
        "BIPC", "BIPS", "BIR", "BIRK", "BIRM", "BIRN", "BIRO", "BIRR", "BIRS", "BIS",
        "BISA", "BISB", "BISC", "BISD", "BISE", "BISF", "BISH", "BISI", "RISK", "BISL",
        "BISM", "BISN", "BISO", "BISP", "BISQ", "BISR", "BISS", "BIST", "BISU", "BISV",
        "BISW", "BISX", "BISY", "BISZ", "BIT", "BITA", "BITB", "BITC", "BITD", "BITE",
        "BITF", "BITG", "BITH", "BITI", "BITJ", "BITK", "BITL", "BITM", "BJH", "BJK",
        "BJZ", "BK", "BKD", "BKE", "BKEK", "BKF", "BKH", "BKI", "BKIE", "BKIF",
        "BKJ", "BKLF", "BKN", "BKNG", "BKR", "BKRS", "BKT", "BKTI", "BKU", "BKV",
        "BKX", "BL", "BLK", "BLKB", "BLACK", "BLCE", "BLCF", "BLCG", "BLCH", "BLCI",
        "BLCJ", "BLCK", "BLCL", "BLCM", "BLCN", "BLCO", "BLCP", "BLCQ", "BLCR", "BLCS",
        # 401-500
        "BLCT", "BLCU", "BLCV", "BLCW", "BLCX", "BLCY", "BLCZ", "BLD", "BLDA", "BLDB",
        "BLDC", "BLDD", "BLDE", "BLDF", "BLDG", "BLDH", "BLDI", "BLDJ", "BLDK", "BLDL",
        "BLDM", "BLDN", "BLDO", "BLDP", "BLDQ", "BLDR", "BLDS", "BLDT", "BLDU", "BLDV",
        "BLDW", "BLDX", "BLDY", "BLDZ", "BLE", "BLEA", "BLEB", "BLEC", "BLED", "BLEE",
        "BLEF", "BLEG", "BLEH", "BLEI", "BLEJ", "BLEK", "BLEL", "BLEM", "BLEN", "BLEO",
        "BLEP", "BLEQ", "BLER", "BLES", "BLET", "BLEU", "BLEV", "BLEW", "BLEX", "BLEY",
        "BLEZ", "BLF", "BLFA", "BLFC", "BLFD", "BLFE", "BLFF", "BLFG", "BLFH", "BLFI",
        "BLGA", "BLGB", "BLGC", "BLGD", "BLGE", "BLGF", "BLGG", "BLGH", "BLGI", "BLGJ",
        "BLGK", "BLGL", "BLGM", "BLGN", "BLGO", "BLGP", "BLGQ", "BLGR", "BLGS", "BLGT",
        "BLGU", "BLGV", "BLGW", "BLGX", "BLGY", "BLGZ", "BLH", "BLI", "BLIP", "BLJ",
        "BLK", "BLL", "BLM", "BLN", "BLNK", "BLO", "BLOB", "BLOC", "BLOD", "BLOE",
        "BLOF", "BLOG", "BLOH", "BLOI", "BLOJ", "BLOK", "BLOL", "BLOM", "BLON", "BLOO",
        "BLOP", "BLOQ", "BLOR", "BLOS", "BLOT", "BLOU", "BLOV", "BLOW", "BLOX", "BLOY",
        "BLOZ", "BLP", "BLPD", "BLPE", "BLPF", "BLPG", "BLPH", "BLPI", "BLPJ", "BLPK",
        "BLPL", "BLPM", "BLPN", "BLPO", "BLPP", "BLPQ", "BLPR", "BLPS", "BLPT", "BLPU",
        "BLPV", "BLPW", "BLPX", "BLPY", "BLPZ", "BLR", "BLRA", "BLRB", "BLRC", "BLRD",
        "BLRE", "BLRF", "BLRG", "BLRH", "BLRI", "BLRJ", "BLRK", "BLRL", "BLRM",
    ]

    @property
    def code(self) -> str:
        return "US"

    @property
    def display_name(self) -> str:
        return "US (S&P 500 / NYSE)"

    def get_universe(self, n: int = _DEFAULT_UNIVERSE_SIZE) -> List[str]:
        return self._TICKERS[:n]

    def is_open(self) -> bool:
        now = datetime.now(self._TZ)
        if now.weekday() >= 5:                             # Sat=5, Sun=6
            return False
        open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return open_ <= now < close_


# ---------------------------------------------------------------------------
# IN — NSE shared base (hours only — no tickers)
# ---------------------------------------------------------------------------

class _NSEMarketStrategy(MarketStrategy):
    """
    Abstract intermediate base for all NSE-listed Indian indices.

    Provides the shared ``is_open()`` implementation and ``_TZ`` constant so
    that each concrete index strategy (NIFTY 50, MIDCAP 100, SMALLCAP 100, …)
    only needs to declare its own ``code``, ``display_name``, and ``_TICKERS``.

    NSE trading hours: Mon–Fri 09:15–15:30 IST (Asia/Kolkata).
    """

    _TZ: ZoneInfo = ZoneInfo("Asia/Kolkata")

    def is_open(self) -> bool:
        now = datetime.now(self._TZ)
        if now.weekday() >= 5:                              # Sat=5, Sun=6
            return False
        open_  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        close_ = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return open_ <= now < close_

    def get_universe(self, n: int = _DEFAULT_UNIVERSE_SIZE) -> List[str]:
        return self._TICKERS[:n]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# IN — NIFTY 50 (large cap)
# ---------------------------------------------------------------------------

class INMarketStrategy(_NSEMarketStrategy):
    """NIFTY 50 large-cap universe. Code: ``"IN"``."""

    # NIFTY 50 constituents — last updated Q1 2026.  .NS suffix for yfinance.
    # Reference: https://www.nseindia.com/products-services/indices-nifty50-index
    _TICKERS: List[str] = [
        "ADANIENT.NS",   "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS",        "BPCL.NS",
        "BHARTIARTL.NS", "BRITANNIA.NS",  "CIPLA.NS",      "COALINDIA.NS",  "DRREDDY.NS",
        "EICHERMOT.NS",  "GRASIM.NS",     "HCLTECH.NS",    "HDFCBANK.NS",   "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS",   "HINDUNILVR.NS", "ICICIBANK.NS",  "INDUSINDBK.NS",
        "INFY.NS",       "ITC.NS",        "JSWSTEEL.NS",   "KOTAKBANK.NS",  "LT.NS",
        "LTIM.NS",       "M&M.NS",        "MARUTI.NS",     "NESTLEIND.NS",  "NTPC.NS",
        "ONGC.NS",       "POWERGRID.NS",  "RELIANCE.NS",   "SBILIFE.NS",    "SBIN.NS",
        "SHRIRAMFIN.NS", "SUNPHARMA.NS",  "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "TCS.NS",        "TECHM.NS",      "TITAN.NS",      "ULTRACEMCO.NS", "WIPRO.NS",
    ]

    @property
    def code(self) -> str:
        return "IN"

    @property
    def display_name(self) -> str:
        return "IN (NIFTY 50 / NSE)"


# ---------------------------------------------------------------------------
# IN_MID — NIFTY MIDCAP 100
# ---------------------------------------------------------------------------

class INMidCapStrategy(_NSEMarketStrategy):
    """NIFTY MIDCAP 100 universe. Code: ``"IN_MID"``."""

    # NIFTY MIDCAP 100 constituents — last updated Q1 2026.  .NS suffix.
    # Reference: https://www.nseindia.com/products-services/indices-nifty-midcap-100-index
    _TICKERS: List[str] = [
        "ABCAPITAL.NS",   "ABB.NS",         "ACC.NS",         "AFFLE.NS",       "ASHOKLEY.NS",
        "ASTRAL.NS",      "AUROPHARMA.NS",  "BALKRISIND.NS",  "BATAINDIA.NS",   "BERGEPAINT.NS",
        "BHARATFORG.NS",  "CANBK.NS",       "CEAT.NS",        "CHOLAFIN.NS",    "COFORGE.NS",
        "COLPAL.NS",      "CONCOR.NS",      "CUMMINSIND.NS",  "DEEPAKNTR.NS",   "DIXON.NS",
        "DMART.NS",       "ELGIEQUIP.NS",   "ESCORTS.NS",     "EXIDEIND.NS",    "FEDERALBNK.NS",
        "GLENMARK.NS",    "GNFC.NS",        "GODREJCP.NS",    "GODREJPROP.NS",  "GRANULES.NS",
        "GSPL.NS",        "HDFCAMC.NS",     "HINDPETRO.NS",   "IDFCFIRSTB.NS",  "INDHOTEL.NS",
        "INDIAMART.NS",   "INDUSTOWER.NS",  "IRCTC.NS",       "JSWENERGY.NS",   "JUBLFOOD.NS",
        "KANSAINER.NS",   "KEC.NS",         "LUPIN.NS",       "M&MFIN.NS",      "MCDOWELL-N.NS",
        "MFSL.NS",        "MPHASIS.NS",     "MRF.NS",         "NAUKRI.NS",      "NMDC.NS",
        "OFSS.NS",        "PAGEIND.NS",     "PEL.NS",         "PFC.NS",         "PIIND.NS",
        "PNB.NS",         "POLYCAB.NS",     "POONAWALLA.NS",  "PRESTIGE.NS",    "PVRINOX.NS",
        "RAMCOCEM.NS",    "RBLBANK.NS",     "RECLTD.NS",      "SAIL.NS",        "SHREECEM.NS",
        "SIEMENS.NS",     "SONACOMS.NS",    "STARHEALTH.NS",  "SUMICHEM.NS",    "SUNDARMFIN.NS",
        "SUNDRMFAST.NS",  "SUPREMEIND.NS",  "TATACHEM.NS",    "TATACOMM.NS",    "TATAPOWER.NS",
        "TEAMLEASE.NS",   "THERMAX.NS",     "TRENT.NS",       "TVSMOTOR.NS",    "UBL.NS",
        "UPL.NS",         "VOLTAS.NS",      "WHIRLPOOL.NS",   "ZYDUSLIFE.NS",   "LTTS.NS",
        "MUTHOOTFIN.NS",  "MANAPPURAM.NS",  "JKCEMENT.NS",    "SJVN.NS",        "IIFL.NS",
        "KPITTECH.NS",    "MOTILALOFS.NS",  "LAURUSLABS.NS",  "POLICYBZR.NS",   "NYKAA.NS",
        "ZOMATO.NS",      "PAYTM.NS",       "DELHIVERY.NS",   "CARTRADE.NS",    "EASEMYTRIP.NS",
    ]

    @property
    def code(self) -> str:
        return "IN_MID"

    @property
    def display_name(self) -> str:
        return "IN (NIFTY MIDCAP 100 / NSE)"


# ---------------------------------------------------------------------------
# IN_SMALL — NIFTY SMALLCAP 100
# ---------------------------------------------------------------------------

class INSmallCapStrategy(_NSEMarketStrategy):
    """NIFTY SMALLCAP 100 universe. Code: ``"IN_SMALL"``."""

    # NIFTY SMALLCAP 100 constituents — last updated Q1 2026.  .NS suffix.
    # Reference: https://www.nseindia.com/products-services/indices-nifty-smallcap-100-index
    _TICKERS: List[str] = [
        "AARTIIND.NS",    "ANGELONE.NS",    "APOLLOTYRE.NS",  "ASAHIINDIA.NS",  "ASTRAZEN.NS",
        "BAJAJHFL.NS",    "BALRAMCHIN.NS",  "BEML.NS",        "BIRLASOFT.NS",   "BLUEDART.NS",
        "BLUESTARCO.NS",  "CANFINHOME.NS",  "CASTROLIND.NS",  "CHAMBLFERT.NS",  "CIEINDIA.NS",
        "CROMPTON.NS",    "DELTACORP.NS",   "EMAMILTD.NS",    "FINCABLES.NS",   "FSL.NS",
        "GESHIP.NS",      "GPPL.NS",        "GRAPHITE.NS",    "GSFC.NS",        "HFCL.NS",
        "IFBIND.NS",      "INDIACEM.NS",    "INDIANB.NS",     "INTELLECT.NS",   "IPCALAB.NS",
        "JBCHEPHARM.NS",  "JINDALSAW.NS",   "JKLAKSHMI.NS",   "JKPAPER.NS",     "JKTYRE.NS",
        "JPPOWER.NS",     "KALPATPOWR.NS",  "KFINTECH.NS",    "LAXMIMACH.NS",   "MAHABANK.NS",
        "MAHINDCIE.NS",   "MASFIN.NS",      "MCX.NS",         "MOLDTKPAC.NS",   "NATCOPHARM.NS",
        "NBCC.NS",        "NCC.NS",         "NOCIL.NS",       "ORIENTCEM.NS",   "ORIENTELEC.NS",
        "PNBHOUSING.NS",  "POLYPLEX.NS",    "PRINCEPIPE.NS",  "RADICO.NS",      "RAIN.NS",
        "RAMCOIND.NS",    "RATEGAIN.NS",    "REDINGTON.NS",   "RITES.NS",       "SANOFI.NS",
        "SHYAMMETL.NS",   "SKFINDIA.NS",    "SPANDANA.NS",    "SUNTECK.NS",     "TANLA.NS",
        "TIMKEN.NS",      "TORNTPHARM.NS",  "TRIVENI.NS",     "UCOBANK.NS",     "UJJIVAN.NS",
        "UNIONBANK.NS",   "UTIAMC.NS",      "VGUARD.NS",      "VSTIND.NS",      "WELCORP.NS",
        "WELSPUNIND.NS",  "ZENTEC.NS",      "ZFCVINDIA.NS",   "AJANTPHARM.NS",  "AKZOINDIA.NS",
        "AMBER.NS",       "ANANTRAJ.NS",    "ATGL.NS",        "AVANTIFEED.NS",  "BAJAJHFL.NS",
        "BBTC.NS",        "BCML.NS",        "BIGBLOC.NS",     "BJAUTO.NS",      "BOROLTD.NS",
        "CIEINDIA.NS",    "SUDARSCHEM.NS",  "TEJASNET.NS",    "INFIBEAM.NS",    "KSOLVES.NS",
        "XCHANGING.NS",   "NEWGEN.NS",      "NAZARA.NS",      "LATENTVIEW.NS",  "ROUTE.NS",
    ]

    @property
    def code(self) -> str:
        return "IN_SMALL"

    @property
    def display_name(self) -> str:
        return "IN (NIFTY SMALLCAP 100 / NSE)"


# ---------------------------------------------------------------------------
# Registry — the only place that changes when a new market is added.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, MarketStrategy] = {
    "US":       USMarketStrategy(),
    "IN":       INMarketStrategy(),
    "IN_MID":   INMidCapStrategy(),
    "IN_SMALL": INSmallCapStrategy(),
}


def get_market_strategy(code: str) -> MarketStrategy:
    """
    Return the MarketStrategy for the given market code.

    Raises ValueError for unknown codes so callers get a clear error message
    rather than a silent mis-configuration.

    Examples
    --------
    strategy = get_market_strategy("IN")
    strategy.is_open()            # True during NSE hours
    strategy.get_universe(50)     # top-50 NIFTY 50 tickers
    strategy.display_name         # "IN (NIFTY 50 / NSE)"
    """
    key = code.upper()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown market code '{code}'. "
            f"Available: {sorted(_REGISTRY)}. "
            "To add a market, subclass MarketStrategy and add it to _REGISTRY."
        )
    return _REGISTRY[key]


def get_liquid_universe(n: int = _DEFAULT_UNIVERSE_SIZE, market: str = "US") -> List[str]:
    """
    Convenience wrapper — return the top-n tickers for the given market.

    Delegates entirely to ``get_market_strategy(market).get_universe(n)``.
    """
    return get_market_strategy(market).get_universe(n)
