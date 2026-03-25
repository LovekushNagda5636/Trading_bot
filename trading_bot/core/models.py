"""
Core data models for the trading bot.
Defines all entities used across the system: instruments, candles, signals,
orders, positions, trades, option chains, and account information.

Supports: Equity (NSE/BSE), F&O (NFO/BFO), Commodity (MCX), Currency (CDS).

Design rules:
  - float everywhere (no Decimal) — avoids JSON serialization pain,
    matches Angel One API responses, and works with pandas/numpy.
  - Every model has .to_dict() for API/dashboard serialization.
  - Validation in __post_init__ — fail fast on bad data.
  - No external dependencies — only stdlib + dataclasses.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import uuid
import json


# ═══════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════

class Exchange(Enum):
    """Supported exchanges."""
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"      # NSE Futures & Options
    BFO = "BFO"      # BSE Futures & Options
    MCX = "MCX"      # Multi Commodity Exchange
    CDS = "CDS"      # Currency Derivatives Segment
    NCDEX = "NCDEX"  # National Commodity & Derivatives Exchange


class MarketSegment(Enum):
    """Market segment classification."""
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    COMMODITY = "COMMODITY"
    CURRENCY = "CURRENCY"
    INDEX = "INDEX"


class OrderSide(Enum):
    """Order direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types supported by Angel One."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "STOPLOSS_LIMIT"
    SL_M = "STOPLOSS_MARKET"


class ProductType(Enum):
    """Product / margin types."""
    INTRADAY = "INTRADAY"        # MIS — squared off same day
    DELIVERY = "DELIVERY"        # CNC — delivery based
    CARRYFORWARD = "CARRYFORWARD"  # NRML — futures/options carry


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    TRIGGER_PENDING = "TRIGGER_PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionStatus(Enum):
    """Position lifecycle."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


class SignalType(Enum):
    """Signal strength levels (directional)."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TradeType(Enum):
    """Concrete trade type (what to actually do)."""
    # Directional
    LONG = "LONG"
    SHORT = "SHORT"

    # Options — single leg
    CALL_BUY = "CALL_BUY"
    CALL_SELL = "CALL_SELL"
    PUT_BUY = "PUT_BUY"
    PUT_SELL = "PUT_SELL"

    # Options — multi-leg
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    IRON_CONDOR = "IRON_CONDOR"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    RATIO_SPREAD = "RATIO_SPREAD"
    BUTTERFLY = "BUTTERFLY"

    # Commodity
    COMMODITY_LONG = "COMMODITY_LONG"
    COMMODITY_SHORT = "COMMODITY_SHORT"

    # Currency
    CURRENCY_LONG = "CURRENCY_LONG"
    CURRENCY_SHORT = "CURRENCY_SHORT"


class MarketRegime(Enum):
    """Market regime classification for strategy selection."""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    SLIGHTLY_BULLISH = "SLIGHTLY_BULLISH"
    SIDEWAYS = "SIDEWAYS"
    SLIGHTLY_BEARISH = "SLIGHTLY_BEARISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class TimeFrame(Enum):
    """Candle intervals matching Angel One API values."""
    ONE_MINUTE = "ONE_MINUTE"
    THREE_MINUTE = "THREE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    TEN_MINUTE = "TEN_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    ONE_DAY = "ONE_DAY"


class ExitReason(Enum):
    """Why a position was closed — used by the learning engine."""
    TARGET_1_HIT = "TARGET_1_HIT"
    TARGET_2_HIT = "TARGET_2_HIT"
    TARGET_3_HIT = "TARGET_3_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TRAILING_SL_HIT = "TRAILING_SL_HIT"
    TIME_BASED_EXIT = "TIME_BASED_EXIT"
    EOD_SQUAREOFF = "EOD_SQUAREOFF"
    MANUAL_EXIT = "MANUAL_EXIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    NO_PROFIT_TIMEOUT = "NO_PROFIT_TIMEOUT"
    MARGIN_CALL = "MARGIN_CALL"


class MistakeType(Enum):
    """Mistake classification for the self-learning engine."""
    TIGHT_SL = "TIGHT_SL"
    WIDE_SL = "WIDE_SL"
    WRONG_DIRECTION = "WRONG_DIRECTION"
    LATE_ENTRY = "LATE_ENTRY"
    WEAK_SIGNAL = "WEAK_SIGNAL"
    LOW_VOLATILITY_ENTRY = "LOW_VOLATILITY_ENTRY"
    PREMATURE_EXIT = "PREMATURE_EXIT"
    OVER_SIZED = "OVER_SIZED"
    UNDER_SIZED = "UNDER_SIZED"
    BAD_REGIME = "BAD_REGIME"
    CORRELATED_POSITIONS = "CORRELATED_POSITIONS"
    NONE = "NONE"


# ═══════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════

def _enum_value(val: Any) -> Any:
    """Return .value if Enum, else return as-is."""
    return val.value if isinstance(val, Enum) else val


def _dt_iso(val: Any) -> Any:
    """Return ISO string if datetime, else return as-is."""
    return val.isoformat() if isinstance(val, datetime) else val


# ═══════════════════════════════════════════════════════════════
# INSTRUMENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class Instrument:
    """
    Universal instrument model.
    Covers equity, futures, options, commodity, and currency.
    """
    symbol: str                              # Trading symbol (e.g. "RELIANCE-EQ", "NIFTY24MARFUT")
    token: str                               # Angel One instrument token
    exchange: str                            # NSE, NFO, MCX, CDS
    name: str                                # Human-readable name (e.g. "RELIANCE")
    segment: MarketSegment = MarketSegment.EQUITY
    lot_size: int = 1
    tick_size: float = 0.05

    # F&O fields
    expiry: Optional[str] = None             # "28MAR2026" format from Angel One
    strike: Optional[float] = None           # Strike price (already divided by 100)
    option_type: Optional[str] = None        # "CE" or "PE"
    underlying: Optional[str] = None         # Underlying symbol for derivatives
    instrument_type: Optional[str] = None    # EQ, FUTSTK, FUTIDX, OPTSTK, OPTIDX, FUTCOM, FUTCUR

    # Commodity fields
    margin_pct: Optional[float] = None       # Margin requirement as percentage
    contract_value: Optional[float] = None   # Notional value per lot

    # Metadata
    isin: Optional[str] = None

    def is_equity(self) -> bool:
        return self.segment == MarketSegment.EQUITY

    def is_future(self) -> bool:
        return self.segment == MarketSegment.FUTURES

    def is_option(self) -> bool:
        return self.segment == MarketSegment.OPTIONS

    def is_commodity(self) -> bool:
        return self.segment == MarketSegment.COMMODITY

    def is_currency(self) -> bool:
        return self.segment == MarketSegment.CURRENCY

    def is_index(self) -> bool:
        return self.segment == MarketSegment.INDEX

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segment"] = self.segment.value
        return d


# ═══════════════════════════════════════════════════════════════
# CANDLE / TICK DATA
# ═══════════════════════════════════════════════════════════════

@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None   # Open Interest for F&O / commodity

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(
                f"Candle validation failed: high ({self.high}) < low ({self.low}) "
                f"at {self.timestamp}"
            )
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_pct(self) -> float:
        return (self.body / self.open * 100) if self.open > 0 else 0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def range_pct(self) -> float:
        return (self.range / self.low * 100) if self.low > 0 else 0

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

    def to_dict(self) -> dict:
        return {
            "timestamp": _dt_iso(self.timestamp),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "oi": self.oi,
        }


@dataclass
class Tick:
    """Real-time tick from WebSocket or LTP API."""
    token: str
    symbol: str
    exchange: str
    timestamp: datetime
    ltp: float                  # Last traded price
    volume: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0          # Previous close
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    oi: Optional[int] = None   # Open interest
    change_pct: float = 0.0

    def __post_init__(self):
        if self.ltp < 0:
            raise ValueError(f"LTP cannot be negative: {self.ltp}")

    @property
    def spread(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.ltp > 0 and self.spread > 0:
            return self.spread / self.ltp * 100
        return 0.0

    @property
    def bid_ask_ratio(self) -> float:
        if self.ask_qty > 0:
            return self.bid_qty / self.ask_qty
        return 0.0

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": _dt_iso(self.timestamp),
            "ltp": self.ltp,
            "volume": self.volume,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "spread_pct": round(self.spread_pct, 4),
            "oi": self.oi,
            "change_pct": self.change_pct,
        }


@dataclass
class MarketDepth:
    """Level-2 market depth (top 5 bids/asks)."""
    bids: List[Dict[str, float]] = field(default_factory=list)
    asks: List[Dict[str, float]] = field(default_factory=list)
    total_bid_qty: int = 0
    total_ask_qty: int = 0

    @property
    def bid_ask_imbalance(self) -> float:
        """Positive = more buying pressure, Negative = more selling pressure."""
        total = self.total_bid_qty + self.total_ask_qty
        if total == 0:
            return 0.0
        return (self.total_bid_qty - self.total_ask_qty) / total


# ═══════════════════════════════════════════════════════════════
# OPTION CHAIN
# ═══════════════════════════════════════════════════════════════

@dataclass
class OptionGreeks:
    """Option greeks for risk assessment."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    iv: float = 0.0    # Implied Volatility (%)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptionChainEntry:
    """Single strike in the option chain."""
    strike: float
    expiry: str
    option_type: str          # "CE" or "PE"
    token: str
    symbol: str
    ltp: float = 0.0
    oi: int = 0
    oi_change: int = 0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    iv: float = 0.0
    greeks: Optional[OptionGreeks] = None
    lot_size: int = 1

    @property
    def spread_pct(self) -> float:
        if self.ltp > 0 and self.ask > self.bid:
            return (self.ask - self.bid) / self.ltp * 100
        return 0.0

    @property
    def is_liquid(self) -> bool:
        """Check if option has reasonable liquidity."""
        return self.oi >= 500 and self.volume >= 100 and self.spread_pct < 5.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.greeks:
            d["greeks"] = self.greeks.to_dict()
        return d


@dataclass
class OptionChain:
    """Complete option chain for a symbol at one expiry."""
    underlying: str
    spot_price: float
    expiry: str
    entries: List[OptionChainEntry] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def calls(self) -> List[OptionChainEntry]:
        return [e for e in self.entries if e.option_type == "CE"]

    @property
    def puts(self) -> List[OptionChainEntry]:
        return [e for e in self.entries if e.option_type == "PE"]

    @property
    def pcr_oi(self) -> float:
        """Put-Call Ratio by Open Interest."""
        call_oi = sum(e.oi for e in self.calls)
        put_oi = sum(e.oi for e in self.puts)
        return round(put_oi / call_oi, 3) if call_oi > 0 else 0.0

    @property
    def pcr_volume(self) -> float:
        """Put-Call Ratio by Volume."""
        call_vol = sum(e.volume for e in self.calls)
        put_vol = sum(e.volume for e in self.puts)
        return round(put_vol / call_vol, 3) if call_vol > 0 else 0.0

    @property
    def max_pain(self) -> float:
        """Strike where maximum options expire worthless."""
        strikes = sorted(set(e.strike for e in self.entries))
        if not strikes:
            return self.spot_price

        min_pain_value = float("inf")
        max_pain_strike = self.spot_price

        for strike in strikes:
            pain = 0.0
            for c in self.calls:
                if strike > c.strike:
                    pain += (strike - c.strike) * c.oi
            for p in self.puts:
                if strike < p.strike:
                    pain += (p.strike - strike) * p.oi

            if pain < min_pain_value:
                min_pain_value = pain
                max_pain_strike = strike

        return max_pain_strike

    def get_atm_strike(self) -> float:
        """Get at-the-money strike nearest to spot."""
        strikes = sorted(set(e.strike for e in self.entries))
        if not strikes:
            return round(self.spot_price / 50) * 50
        return min(strikes, key=lambda s: abs(s - self.spot_price))

    def get_max_oi_strike(self, option_type: str = "CE") -> float:
        """Strike with highest OI (support for PE, resistance for CE)."""
        entries = self.calls if option_type == "CE" else self.puts
        if not entries:
            return self.spot_price
        return max(entries, key=lambda e: e.oi).strike

    def to_summary(self) -> dict:
        """Compact summary for dashboard display."""
        return {
            "underlying": self.underlying,
            "spot_price": self.spot_price,
            "expiry": self.expiry,
            "pcr_oi": self.pcr_oi,
            "pcr_volume": self.pcr_volume,
            "max_pain": self.max_pain,
            "atm_strike": self.get_atm_strike(),
            "max_call_oi_strike": self.get_max_oi_strike("CE"),
            "max_put_oi_strike": self.get_max_oi_strike("PE"),
            "total_calls": len(self.calls),
            "total_puts": len(self.puts),
            "total_call_oi": sum(e.oi for e in self.calls),
            "total_put_oi": sum(e.oi for e in self.puts),
            "timestamp": _dt_iso(self.timestamp),
        }


# ═══════════════════════════════════════════════════════════════
# TRADING SIGNAL
# ═══════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """
    Trading signal from a single strategy.
    This is the core output of every strategy's generate_signal() method.
    """
    # Identity
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    exchange: str = "NSE"
    segment: MarketSegment = MarketSegment.EQUITY

    # Signal direction & type
    signal_type: SignalType = SignalType.NEUTRAL
    trade_type: TradeType = TradeType.LONG

    # Price levels
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0

    # Quality metrics
    confidence: float = 0.0       # 0–100 scale
    risk_reward_ratio: float = 0.0

    # Strategy info
    strategy_name: str = ""
    reason: str = ""
    indicators_used: List[str] = field(default_factory=list)

    # Execution hints
    suggested_quantity: int = 0
    product_type: ProductType = ProductType.INTRADAY
    timeframe: str = "FIVE_MINUTE"

    # Options specific
    option_strike: Optional[float] = None
    option_type: Optional[str] = None    # CE / PE
    option_expiry: Optional[str] = None
    option_premium: Optional[float] = None
    greeks: Optional[OptionGreeks] = None
    legs: List[Dict[str, Any]] = field(default_factory=list)

    # Risk estimates
    max_loss: Optional[float] = None
    max_profit: Optional[float] = None
    breakeven: Optional[float] = None

    # Metadata
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Auto-calculate risk:reward if not set
        if self.risk_reward_ratio == 0 and self.entry_price > 0 and self.stop_loss > 0:
            risk = abs(self.entry_price - self.stop_loss)
            if risk > 0 and self.target_2 > 0:
                reward = abs(self.target_2 - self.entry_price)
                self.risk_reward_ratio = round(reward / risk, 2)

    @property
    def is_buy(self) -> bool:
        return "BUY" in self.signal_type.value

    @property
    def is_sell(self) -> bool:
        return "SELL" in self.signal_type.value

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_loss) if self.entry_price and self.stop_loss else 0.0

    def validate(self) -> Tuple[bool, str]:
        """
        Validate signal integrity. Returns (is_valid, reason).
        Called before the signal enters the scoring/execution pipeline.
        """
        if self.entry_price <= 0:
            return False, "Entry price must be positive"
        if self.stop_loss <= 0:
            return False, "Stop loss must be positive"
        if self.confidence < 0 or self.confidence > 100:
            return False, f"Confidence out of range: {self.confidence}"

        if self.is_buy:
            if self.stop_loss >= self.entry_price:
                return False, f"BUY signal: SL ({self.stop_loss}) >= entry ({self.entry_price})"
            if self.target_1 > 0 and self.target_1 <= self.entry_price:
                return False, f"BUY signal: T1 ({self.target_1}) <= entry ({self.entry_price})"
        elif self.is_sell:
            if self.stop_loss <= self.entry_price:
                return False, f"SELL signal: SL ({self.stop_loss}) <= entry ({self.entry_price})"
            if self.target_1 > 0 and self.target_1 >= self.entry_price:
                return False, f"SELL signal: T1 ({self.target_1}) >= entry ({self.entry_price})"

        if self.risk_reward_ratio < 0.5 and self.signal_type != SignalType.NEUTRAL:
            return False, f"Risk:reward too low: {self.risk_reward_ratio}"

        return True, "OK"

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "segment": _enum_value(self.segment),
            "signal_type": _enum_value(self.signal_type),
            "trade_type": _enum_value(self.trade_type),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "target_3": self.target_3,
            "confidence": round(self.confidence, 1),
            "risk_reward_ratio": self.risk_reward_ratio,
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "suggested_quantity": self.suggested_quantity,
            "product_type": _enum_value(self.product_type),
            "timeframe": self.timeframe,
            "option_strike": self.option_strike,
            "option_type": self.option_type,
            "option_expiry": self.option_expiry,
            "max_loss": self.max_loss,
            "max_profit": self.max_profit,
            "breakeven": self.breakeven,
            "timestamp": _dt_iso(self.timestamp),
        }


@dataclass
class CombinedSignal:
    """
    Aggregated signal from multiple strategies.
    Produced by the scanner after running all applicable strategies on a symbol.
    """
    symbol: str
    overall_signal: SignalType
    recommended_trade: TradeType
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    overall_confidence: float       # 0–100
    risk_reward_ratio: float
    consensus_score: float          # -100 to +100
    strategy_agreement: str         # "STRONG (80%)", "WEAK (40%)"
    suggested_quantity: int
    max_risk: float                 # Max ₹ risk for this trade
    reasons: List[str]
    individual_signals: List[Signal] = field(default_factory=list)

    # Market context
    segment: MarketSegment = MarketSegment.EQUITY
    exchange: str = "NSE"
    market_regime: Optional[MarketRegime] = None

    # Option chain data (if applicable)
    option_chain_summary: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: Optional[datetime] = None
    scan_duration_ms: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def is_actionable(self) -> bool:
        """Signal is worth acting on."""
        return (
            self.overall_signal != SignalType.NEUTRAL
            and self.overall_confidence >= 50
            and self.risk_reward_ratio >= 1.0
        )

    @property
    def buy_sell_breakdown(self) -> Dict[str, int]:
        """Count of buy vs sell signals."""
        buy = sum(1 for s in self.individual_signals if s.is_buy)
        sell = sum(1 for s in self.individual_signals if s.is_sell)
        neutral = len(self.individual_signals) - buy - sell
        return {"buy": buy, "sell": sell, "neutral": neutral}

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal": _enum_value(self.overall_signal),
            "trade_type": _enum_value(self.recommended_trade),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "target_3": self.target_3,
            "confidence": round(self.overall_confidence, 1),
            "risk_reward": self.risk_reward_ratio,
            "consensus_score": round(self.consensus_score, 1),
            "strategy_agreement": self.strategy_agreement,
            "quantity": self.suggested_quantity,
            "max_risk": round(self.max_risk, 2),
            "reasons": self.reasons,
            "segment": _enum_value(self.segment),
            "exchange": self.exchange,
            "market_regime": _enum_value(self.market_regime) if self.market_regime else None,
            "option_chain_summary": self.option_chain_summary,
            "timestamp": _dt_iso(self.timestamp),
            "scan_duration_ms": self.scan_duration_ms,
            "breakdown": self.buy_sell_breakdown,
            "individual_strategies": [
                {
                    "name": s.strategy_name,
                    "signal": _enum_value(s.signal_type),
                    "confidence": round(s.confidence, 1),
                    "reason": s.reason,
                }
                for s in self.individual_signals
            ],
        }


# ═══════════════════════════════════════════════════════════════
# ORDER
# ═══════════════════════════════════════════════════════════════

@dataclass
class Order:
    """Order placed with broker."""
    # Identity
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    broker_order_id: Optional[str] = None

    # Instrument
    symbol: str = ""
    token: str = ""
    exchange: str = "NSE"

    # Order details
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    product_type: ProductType = ProductType.INTRADAY
    quantity: int = 0
    price: float = 0.0
    trigger_price: float = 0.0

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    rejection_reason: str = ""

    # Tracking
    strategy_name: str = ""
    signal_id: Optional[str] = None
    tag: str = ""

    # Parent/child (for bracket/cover orders)
    parent_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    # Timestamps
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_open(self) -> bool:
        return self.status in (
            OrderStatus.PENDING, OrderStatus.OPEN,
            OrderStatus.TRIGGER_PENDING, OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED, OrderStatus.CANCELLED,
            OrderStatus.REJECTED, OrderStatus.EXPIRED,
        )

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    def to_angel_one_params(self) -> dict:
        """Convert to Angel One SmartAPI placeOrder parameters."""
        variety = "NORMAL"
        if self.order_type in (OrderType.SL, OrderType.SL_M):
            variety = "STOPLOSS"

        product_map = {
            ProductType.INTRADAY: "INTRADAY",
            ProductType.DELIVERY: "DELIVERY",
            ProductType.CARRYFORWARD: "CARRYFORWARD",
        }

        params = {
            "variety": variety,
            "tradingsymbol": self.symbol,
            "symboltoken": self.token,
            "transactiontype": self.side.value,
            "exchange": self.exchange,
            "ordertype": self.order_type.value,
            "producttype": product_map.get(self.product_type, "INTRADAY"),
            "duration": "DAY",
            "quantity": str(self.quantity),
        }

        if self.order_type == OrderType.LIMIT:
            params["price"] = str(self.price)
        elif self.order_type == OrderType.SL:
            params["price"] = str(self.price)
            params["triggerprice"] = str(self.trigger_price)
        elif self.order_type == OrderType.SL_M:
            params["triggerprice"] = str(self.trigger_price)

        return params

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "side": _enum_value(self.side),
            "order_type": _enum_value(self.order_type),
            "product_type": _enum_value(self.product_type),
            "quantity": self.quantity,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "status": _enum_value(self.status),
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "remaining": self.remaining_quantity,
            "strategy_name": self.strategy_name,
            "placed_at": _dt_iso(self.placed_at),
            "filled_at": _dt_iso(self.filled_at),
        }


# ═══════════════════════════════════════════════════════════════
# POSITION
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Live trading position."""
    # Identity
    position_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    token: str = ""
    exchange: str = "NSE"
    segment: MarketSegment = MarketSegment.EQUITY

    # Position details
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    average_price: float = 0.0
    ltp: float = 0.0

    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Risk levels
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    trailing_stop: Optional[float] = None
    trailing_activated: bool = False

    # Status
    status: PositionStatus = PositionStatus.OPEN
    strategy_name: str = ""

    # Related orders
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    # Timestamps
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Options specific
    option_strike: Optional[float] = None
    option_type: Optional[str] = None
    option_expiry: Optional[str] = None
    greeks: Optional[Dict[str, float]] = None

    # Tracking
    max_favorable_excursion: float = 0.0   # Best unrealized P&L during trade
    max_adverse_excursion: float = 0.0     # Worst unrealized P&L during trade
    signal_confidence: float = 0.0
    market_regime_at_entry: str = ""

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def is_long(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_short(self) -> bool:
        return self.side == OrderSide.SELL

    @property
    def pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl

    @property
    def pnl_pct(self) -> float:
        if self.average_price == 0:
            return 0.0
        if self.is_long:
            return (self.ltp - self.average_price) / self.average_price * 100
        else:
            return (self.average_price - self.ltp) / self.average_price * 100

    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.ltp

    @property
    def holding_duration_minutes(self) -> float:
        if self.entry_time is None:
            return 0.0
        end = self.exit_time or datetime.now()
        return (end - self.entry_time).total_seconds() / 60

    @property
    def is_in_profit(self) -> bool:
        return self.pnl_pct > 0

    def update_price(self, current_price: float) -> None:
        """Update LTP and recalculate P&L, MFE, MAE."""
        self.ltp = current_price
        self.last_updated = datetime.now()

        if self.quantity != 0 and self.average_price > 0:
            if self.is_long:
                self.unrealized_pnl = (current_price - self.average_price) * abs(self.quantity)
            else:
                self.unrealized_pnl = (self.average_price - current_price) * abs(self.quantity)

        # Track excursions
        if self.unrealized_pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = self.unrealized_pnl
        if self.unrealized_pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = self.unrealized_pnl

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "segment": _enum_value(self.segment),
            "side": _enum_value(self.side),
            "quantity": self.quantity,
            "average_price": self.average_price,
            "ltp": self.ltp,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "market_value": round(self.market_value, 2),
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "trailing_stop": self.trailing_stop,
            "trailing_activated": self.trailing_activated,
            "status": _enum_value(self.status),
            "strategy_name": self.strategy_name,
            "entry_time": _dt_iso(self.entry_time),
            "holding_minutes": round(self.holding_duration_minutes, 1),
            "mfe": round(self.max_favorable_excursion, 2),
            "mae": round(self.max_adverse_excursion, 2),
            "signal_confidence": self.signal_confidence,
        }


# ═══════════════════════════════════════════════════════════════
# TRADE RECORD (for journal / learning)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """
    Complete trade record for the journal and self-learning engine.
    Created when a position is closed.
    """
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    exchange: str = "NSE"
    segment: str = "EQUITY"

    # Entry
    entry_side: str = "BUY"
    entry_price: float = 0.0
    entry_quantity: int = 0
    entry_time: Optional[datetime] = None
    entry_order_id: str = ""

    # Exit
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_order_id: str = ""
    exit_reason: str = ""        # ExitReason value

    # P&L
    gross_pnl: float = 0.0
    charges: float = 0.0        # Brokerage + STT + exchange charges
    net_pnl: float = 0.0
    pnl_pct: float = 0.0

    # Strategy context
    strategy_name: str = ""
    signal_confidence: float = 0.0
    signal_type: str = ""        # SignalType value

    # Risk context
    stop_loss: float = 0.0
    target: float = 0.0
    risk_reward_planned: float = 0.0
    risk_reward_actual: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Market context at entry
    market_regime: str = ""
    volatility_at_entry: float = 0.0
    atr_at_entry: float = 0.0
    volume_ratio_at_entry: float = 0.0

    # Learning
    mistake_type: str = ""       # MistakeType value
    lesson_learned: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def holding_duration_minutes(self) -> float:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return 0.0

    @property
    def slippage_pct(self) -> float:
        """Estimate total slippage percentage."""
        if self.gross_pnl == 0:
            return 0.0
        return abs(self.charges / self.gross_pnl) * 100 if self.gross_pnl != 0 else 0.0

    def classify_mistake(self) -> str:
        """
        Auto-classify the mistake type based on trade outcome.
        Called by the learning engine after position close.
        """
        if self.is_winner:
            # Check if we left too much on the table
            if self.max_favorable_excursion > 0:
                captured_ratio = self.gross_pnl / self.max_favorable_excursion
                if captured_ratio < 0.4:
                    self.mistake_type = MistakeType.PREMATURE_EXIT.value
                    self.lesson_learned = (
                        f"Captured only {captured_ratio:.0%} of max potential. "
                        f"Consider wider targets or trailing stops."
                    )
                    return self.mistake_type
            self.mistake_type = MistakeType.NONE.value
            return self.mistake_type

        # Losing trade analysis
        risk = abs(self.entry_price - self.stop_loss) if self.stop_loss else 0

        # SL was too tight — price reversed after hitting SL
        if (self.exit_reason in (ExitReason.STOP_LOSS_HIT.value, ExitReason.TRAILING_SL_HIT.value)
                and self.max_favorable_excursion > 0):
            self.mistake_type = MistakeType.TIGHT_SL.value
            self.lesson_learned = (
                f"SL hit but trade showed MFE of ₹{self.max_favorable_excursion:.2f}. "
                f"Consider widening SL by {abs(self.max_adverse_excursion):.2f}."
            )

        # Wrong direction entirely
        elif self.max_favorable_excursion <= 0:
            self.mistake_type = MistakeType.WRONG_DIRECTION.value
            self.lesson_learned = (
                f"Trade never went positive. MAE=₹{self.max_adverse_excursion:.2f}. "
                f"Strategy '{self.strategy_name}' may be wrong for regime '{self.market_regime}'."
            )

        # Late entry (after 2 PM)
        elif self.entry_time and self.entry_time.hour >= 14:
            self.mistake_type = MistakeType.LATE_ENTRY.value
            self.lesson_learned = (
                f"Entered at {self.entry_time.strftime('%H:%M')}. "
                f"Avoid entries after 14:00 for this setup."
            )

        # Weak signal
        elif self.signal_confidence < 55:
            self.mistake_type = MistakeType.WEAK_SIGNAL.value
            self.lesson_learned = (
                f"Signal confidence was only {self.signal_confidence:.0f}%. "
                f"Raise minimum threshold."
            )

        # Low volatility — no room for profit
        elif self.atr_at_entry > 0 and risk > 0:
            if self.atr_at_entry < risk * 0.5:
                self.mistake_type = MistakeType.LOW_VOLATILITY_ENTRY.value
                self.lesson_learned = (
                    f"ATR ({self.atr_at_entry:.2f}) was much less than risk ({risk:.2f}). "
                    f"Skip low-volatility setups."
                )
            else:
                self.mistake_type = MistakeType.WIDE_SL.value
                self.lesson_learned = "SL was too wide relative to ATR."
        else:
            self.mistake_type = MistakeType.BAD_REGIME.value
            self.lesson_learned = (
                f"Strategy '{self.strategy_name}' underperformed in '{self.market_regime}' regime."
            )

        return self.mistake_type

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "segment": self.segment,
            "entry_side": self.entry_side,
            "entry_price": self.entry_price,
            "entry_quantity": self.entry_quantity,
            "entry_time": _dt_iso(self.entry_time),
            "exit_price": self.exit_price,
            "exit_time": _dt_iso(self.exit_time),
            "exit_reason": self.exit_reason,
            "gross_pnl": round(self.gross_pnl, 2),
            "charges": round(self.charges, 2),
            "net_pnl": round(self.net_pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "is_winner": self.is_winner,
            "strategy_name": self.strategy_name,
            "signal_confidence": self.signal_confidence,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "risk_reward_planned": self.risk_reward_planned,
            "risk_reward_actual": self.risk_reward_actual,
            "mfe": round(self.max_favorable_excursion, 2),
            "mae": round(self.max_adverse_excursion, 2),
            "holding_minutes": round(self.holding_duration_minutes, 1),
            "market_regime": self.market_regime,
            "mistake_type": self.mistake_type,
            "lesson_learned": self.lesson_learned,
            "tags": self.tags,
        }


# ═══════════════════════════════════════════════════════════════
# ACCOUNT / PORTFOLIO
# ═══════════════════════════════════════════════════════════════

@dataclass
class AccountInfo:
    """Broker account summary."""
    client_id: str = ""
    available_margin: float = 0.0
    used_margin: float = 0.0
    total_balance: float = 0.0
    day_pnl: float = 0.0
    total_pnl: float = 0.0
    available_cash: float = 0.0

    # Commodity margin (separate for MCX)
    commodity_margin_available: float = 0.0
    commodity_margin_used: float = 0.0

    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def buying_power(self) -> float:
        return self.available_margin

    @property
    def margin_utilization_pct(self) -> float:
        total = self.available_margin + self.used_margin
        if total <= 0:
            return 0.0
        return (self.used_margin / total) * 100

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "available_margin": self.available_margin,
            "used_margin": self.used_margin,
            "total_balance": self.total_balance,
            "day_pnl": round(self.day_pnl, 2),
            "buying_power": self.buying_power,
            "margin_utilization_pct": round(self.margin_utilization_pct, 1),
            "commodity_margin_available": self.commodity_margin_available,
            "last_updated": _dt_iso(self.last_updated),
        }


@dataclass
class RiskSnapshot:
    """Point-in-time snapshot of portfolio risk."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_capital: float = 0.0
    used_capital: float = 0.0
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0
    open_positions: int = 0
    max_positions: int = 5
    daily_loss_used: float = 0.0
    daily_loss_limit: float = 0.0
    remaining_loss_capacity: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    can_trade: bool = True
    reason_if_blocked: str = ""

    # Per-segment exposure
    equity_exposure: float = 0.0
    futures_exposure: float = 0.0
    options_exposure: float = 0.0
    commodity_exposure: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": _dt_iso(self.timestamp),
            "total_capital": self.total_capital,
            "used_capital": round(self.used_capital, 2),
            "day_pnl": round(self.day_pnl, 2),
            "day_pnl_pct": round(self.day_pnl_pct, 2),
            "open_positions": self.open_positions,
            "max_positions": self.max_positions,
            "daily_loss_used": round(self.daily_loss_used, 2),
            "daily_loss_limit": round(self.daily_loss_limit, 2),
            "remaining_loss_capacity": round(self.remaining_loss_capacity, 2),
            "drawdown_pct": round(self.current_drawdown_pct, 2),
            "consecutive_losses": self.consecutive_losses,
            "can_trade": self.can_trade,
            "reason_if_blocked": self.reason_if_blocked,
            "exposure": {
                "equity": round(self.equity_exposure, 2),
                "futures": round(self.futures_exposure, 2),
                "options": round(self.options_exposure, 2),
                "commodity": round(self.commodity_exposure, 2),
            },
        }


# ═══════════════════════════════════════════════════════════════
# LEARNING PARAMETERS (persisted to data/models/)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AdaptiveParameters:
    """
    Self-learning parameters that are updated after each trading session.
    Persisted to data/models/adaptive_params.json.
    """
    # Strategy weights per regime
    strategy_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Score thresholds per regime
    min_score_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BULLISH": 55.0,
        "BULLISH": 60.0,
        "SIDEWAYS": 70.0,
        "BEARISH": 60.0,
        "STRONG_BEARISH": 55.0,
        "HIGH_VOLATILITY": 65.0,
        "LOW_VOLATILITY": 70.0,
    })

    # Stop-loss multipliers per regime
    sl_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BULLISH": 1.5,
        "BULLISH": 1.5,
        "SIDEWAYS": 1.2,
        "BEARISH": 1.5,
        "STRONG_BEARISH": 1.5,
        "HIGH_VOLATILITY": 2.0,
        "LOW_VOLATILITY": 1.0,
    })

    # Target multipliers per regime
    target_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BULLISH": 2.5,
        "BULLISH": 2.0,
        "SIDEWAYS": 1.5,
        "BEARISH": 2.0,
        "STRONG_BEARISH": 2.5,
        "HIGH_VOLATILITY": 3.0,
        "LOW_VOLATILITY": 1.2,
    })

    # Hours to avoid (learned from losing patterns)
    avoid_hours: List[int] = field(default_factory=list)

    # Best hours (learned from winning patterns)
    best_hours: List[int] = field(default_factory=lambda: [10, 11, 14])

    # Position limit per regime
    max_positions_per_regime: Dict[str, int] = field(default_factory=lambda: {
        "STRONG_BULLISH": 5,
        "BULLISH": 4,
        "SIDEWAYS": 2,
        "BEARISH": 3,
        "STRONG_BEARISH": 2,
        "HIGH_VOLATILITY": 2,
        "LOW_VOLATILITY": 3,
    })

    # Discovered winning strategy combinations
    winning_combos: List[Dict[str, Any]] = field(default_factory=list)

    # Global counters
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    last_updated: Optional[str] = None

    @property
    def overall_win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades * 100

    def get_sl_multiplier(self, regime: str) -> float:
        return self.sl_multipliers.get(regime, 1.5)

    def get_target_multiplier(self, regime: str) -> float:
        return self.target_multipliers.get(regime, 2.0)

    def get_min_score(self, regime: str) -> float:
        return self.min_score_thresholds.get(regime, 60.0)

    def get_max_positions(self, regime: str) -> int:
        return self.max_positions_per_regime.get(regime, 3)

    def get_strategy_weight(self, strategy_name: str, regime: str) -> float:
        regime_weights = self.strategy_weights.get(regime, {})
        return regime_weights.get(strategy_name, 1.0)

    def should_avoid_hour(self, hour: int) -> bool:
        return hour in self.avoid_hours

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str = "data/models/adaptive_params.json") -> None:
        """Persist to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str = "data/models/adaptive_params.json") -> "AdaptiveParameters":
        """Load from disk, or return defaults."""
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()
            # ═══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════
# These aliases ensure existing code that imports old model names
# continues to work without modification. As you enhance each file,
# migrate it to use the new names, then remove these aliases.

# Old name → New name
Trade = TradeRecord
TradeSide = OrderSide
TradingSignal = Signal

# Old MarketTick → new Tick (structural difference, so we provide a shim)
class MarketTick:
    """Backward-compatible shim for code that uses MarketTick."""
    def __init__(
        self,
        symbol: str = "",
        exchange: Exchange = Exchange.NSE,
        timestamp: datetime = None,
        price: float = 0.0,
        volume: int = 0,
        bid_price: float = 0.0,
        ask_price: float = 0.0,
        bid_size: int = 0,
        ask_size: int = 0,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp or datetime.now()
        self.price = price
        self.volume = volume
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size


# Old OHLC → new Candle (field names differ)
class OHLC:
    """Backward-compatible shim for code that uses OHLC."""
    def __init__(
        self,
        timestamp: datetime = None,
        open_price: float = 0.0,
        high_price: float = 0.0,
        low_price: float = 0.0,
        close_price: float = 0.0,
        volume: int = 0,
    ):
        self.timestamp = timestamp or datetime.now()
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        # Also expose Candle-style names
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price


# Old OHLCData shim
class OHLCData:
    """Backward-compatible shim for code that uses OHLCData."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'timestamp'):
            self.timestamp = datetime.now()


# Old TickData shim
class TickData:
    """Backward-compatible shim for code that uses TickData."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'timestamp'):
            self.timestamp = datetime.now()


# Old MarketData shim
class MarketData:
    """Backward-compatible shim for code that uses MarketData."""
    def __init__(
        self,
        symbol: str = "",
        exchange: Exchange = Exchange.NSE,
        **kwargs,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.ohlc_1min = kwargs.get('ohlc_1min', [])
        self.ohlc_5min = kwargs.get('ohlc_5min', [])
        self.ohlc_15min = kwargs.get('ohlc_15min', [])
        self.ohlc_1hour = kwargs.get('ohlc_1hour', [])
        self.current_tick = kwargs.get('current_tick', None)
        self.last_updated = kwargs.get('last_updated', datetime.now())


# Old TradeDecision → maps to Order concept
class TradeDecision:
    """Backward-compatible shim for code that uses TradeDecision."""
    def __init__(self, **kwargs):
        self.decision_id = kwargs.get('decision_id', str(uuid.uuid4())[:12])
        self.symbol = kwargs.get('symbol', '')
        self.action = kwargs.get('action', OrderSide.BUY)
        self.quantity = kwargs.get('quantity', 0)
        self.order_type = kwargs.get('order_type', OrderType.MARKET)
        self.price = kwargs.get('price', None)
        self.stop_loss = kwargs.get('stop_loss', None)
        self.take_profit = kwargs.get('take_profit', None)
        self.strategy_id = kwargs.get('strategy_id', '')
        self.timestamp = kwargs.get('timestamp', datetime.now())


# Old IndicatorValues shim
class IndicatorValues:
    """Backward-compatible shim for code that uses IndicatorValues."""
    def __init__(self, **kwargs):
        self.timestamp = kwargs.get('timestamp', datetime.now())
        self.sma_20 = kwargs.get('sma_20', None)
        self.sma_50 = kwargs.get('sma_50', None)
        self.ema_12 = kwargs.get('ema_12', None)
        self.ema_26 = kwargs.get('ema_26', None)
        self.rsi = kwargs.get('rsi', None)
        self.macd_line = kwargs.get('macd_line', None)
        self.macd_signal = kwargs.get('macd_signal', None)
        self.macd_histogram = kwargs.get('macd_histogram', None)
        self.bb_upper = kwargs.get('bb_upper', None)
        self.bb_middle = kwargs.get('bb_middle', None)
        self.bb_lower = kwargs.get('bb_lower', None)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def has_sufficient_data(self) -> bool:
        return any([
            self.sma_20 is not None,
            self.ema_12 is not None,
            self.rsi is not None,
            self.macd_line is not None,
        ])


# Old RiskLimits shim
class RiskLimits:
    """Backward-compatible shim for code that uses RiskLimits."""
    def __init__(self, **kwargs):
        self.max_position_size = kwargs.get('max_position_size', 100000)
        self.max_daily_loss = kwargs.get('max_daily_loss', 10000)
        self.max_portfolio_exposure = kwargs.get('max_portfolio_exposure', 500000)
        self.max_positions_per_symbol = kwargs.get('max_positions_per_symbol', 1)
        self.max_total_positions = kwargs.get('max_total_positions', 10)
        self.sebi_position_limits = kwargs.get('sebi_position_limits', {})

    def validate_position_size(self, symbol: str, value: float) -> bool:
        if value > self.max_position_size:
            return False
        sebi_limit = self.sebi_position_limits.get(symbol)
        if sebi_limit and value > sebi_limit:
            return False
        return True