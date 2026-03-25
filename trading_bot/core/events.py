"""
Event system for the trading bot.

Production-grade event bus covering the full trading pipeline for
Equity, F&O (NFO), Commodity (MCX), and Currency (CDS) segments.

Features
--------
- 40+ event types spanning data, signals, orders, positions, risk, regime,
  learning, scanner, session, and system categories
- Correlation / causation IDs for tracing an entire signal → order → fill chain
- Priority-based dispatch (CRITICAL → HIGH → NORMAL → LOW → BACKGROUND)
- Thread-safe handler registry with RLock
- Type-indexed O(matched) dispatch for hot-path tick data
- Symbol / segment / source filters on subscriptions
- One-shot subscriptions (auto-remove after first match)
- Both synchronous and asynchronous handler support
- Event history ring-buffer for auditing and replay
- Dispatch metrics (counts, errors, per-type breakdown)
- Wildcard subscriptions (EventType.ALL)
- Factory helper with automatic chain propagation
"""

import asyncio
import logging
import threading
import time as _time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Deque, Dict, List, Optional, Set, Union,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Event Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EventType(str, Enum):
    """Every event type used across the trading system."""

    # ── Market Data ──
    TICK = "tick"
    CANDLE_CLOSE = "candle_close"
    MARKET_DATA = "market_data"                          # legacy / general
    MARKET_DEPTH = "market_depth"
    OPTION_CHAIN_UPDATE = "option_chain_update"

    # ── Signals ──
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_SCORED = "signal_scored"
    TRADING_SIGNAL = "trading_signal"                     # legacy

    # ── Trade Lifecycle ──
    TRADE_DECISION = "trade_decision"
    RISK_APPROVED_TRADE = "risk_approved_trade"
    RISK_REJECTED = "risk_rejected"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_STATUS = "order_status"                         # legacy / general

    # ── Position Management ──
    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    STOP_LOSS_UPDATED = "stop_loss_updated"
    PARTIAL_EXIT = "partial_exit"
    POSITION_CLOSED = "position_closed"

    # ── Risk Management ──
    RISK_ALERT = "risk_alert"
    CIRCUIT_BREAKER = "circuit_breaker"
    DRAWDOWN_ALERT = "drawdown_alert"
    MARGIN_ALERT = "margin_alert"
    DAILY_LIMIT_REACHED = "daily_limit_reached"
    CONSECUTIVE_LOSS_ALERT = "consecutive_loss_alert"

    # ── Market Regime ──
    REGIME_CHANGE = "regime_change"
    VOLATILITY_SHIFT = "volatility_shift"
    SECTOR_ROTATION = "sector_rotation"

    # ── Scanner ──
    SCAN_COMPLETE = "scan_complete"
    OPPORTUNITY_FOUND = "opportunity_found"

    # ── Learning Engine ──
    TRADE_JOURNALED = "trade_journaled"
    LEARNING_CYCLE_COMPLETE = "learning_cycle_complete"
    STRATEGY_WEIGHTS_UPDATED = "strategy_weights_updated"
    MISTAKE_CLASSIFIED = "mistake_classified"

    # ── Session ──
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SQUARE_OFF_WARNING = "square_off_warning"
    HEARTBEAT = "heartbeat"

    # ── System ──
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONNECTION_STATUS = "connection_status"

    # ── Wildcard (subscribe to everything) ──
    ALL = "*"


class EventPriority:
    """Lower number  =  higher priority  =  processed first."""
    CRITICAL = 0          # shutdown, circuit-breaker, halt
    HIGH = 10             # order fills, risk alerts, regime change
    NORMAL = 50           # signals, decisions, position updates
    LOW = 80              # scan results, learning, heartbeat
    BACKGROUND = 100      # metrics, analytics, logging


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Base Event
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Event:
    """
    Root class for every event flowing through the system.

    Direct fields cover the metadata envelope.  Subclasses add
    domain-specific payload fields.  The ``data`` dict holds
    arbitrary extras that don't warrant a named field.
    """

    event_type: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""                     # component that created this event

    # ── scope ──
    symbol: str = ""
    segment: str = ""                    # NSE, NFO, MCX, CDS, BSE, BFO

    # ── chain tracing ──
    correlation_id: str = ""             # shared across an entire signal-to-exit chain
    causation_id: str = ""               # event_id of the direct parent event

    # ── dispatch control ──
    priority: int = EventPriority.NORMAL

    # ── open payload ──
    data: Dict[str, Any] = field(default_factory=dict)

    # ── helpers ──

    def chain(self, **overrides) -> Dict[str, Any]:
        """
        Return constructor kwargs for a child event that inherits
        correlation_id, symbol, and segment from this event.

        Usage::

            signal = SignalEvent(symbol="RELIANCE", ...)
            decision = TradeDecisionEvent(
                **signal.chain(),
                direction="BUY",
                quantity=5,
            )
        """
        base: Dict[str, Any] = {
            "correlation_id": self.correlation_id or self.event_id,
            "causation_id": self.event_id,
            "symbol": self.symbol,
            "segment": self.segment,
        }
        base.update(overrides)
        return base

    @property
    def source_component(self) -> str:
        """Backward-compatible alias for ``source``."""
        return self.source

    @source_component.setter
    def source_component(self, value: str) -> None:
        self.source = value

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since the event was created."""
        return (datetime.now() - self.timestamp).total_seconds()

    def __repr__(self) -> str:
        tag = self.symbol or ""
        src = f", src={self.source}" if self.source else ""
        return f"{self.__class__.__name__}({self.event_type}{', ' + tag if tag else ''}{src})"


# Backward-compatible alias
BaseEvent = Event


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Market Data Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class TickEvent(Event):
    """Real-time tick from Angel One WebSocket.  Hot-path — keep lean."""
    event_type: str = field(default=EventType.TICK, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    ltp: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    oi: int = 0                          # open interest (F&O / MCX)
    change_pct: float = 0.0
    last_trade_qty: int = 0
    avg_trade_price: float = 0.0
    total_buy_qty: int = 0
    total_sell_qty: int = 0


@dataclass
class CandleEvent(Event):
    """Emitted when a candle completes (all timeframes)."""
    event_type: str = field(default=EventType.CANDLE_CLOSE, init=False)
    priority: int = field(default=EventPriority.NORMAL, init=False)

    timeframe: str = "5m"                # 1m, 3m, 5m, 15m, 1h, 1d
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    oi: int = 0
    candle_timestamp: Optional[datetime] = None


@dataclass
class MarketDataEvent(Event):
    """General market-data update (backward compatible with v1)."""
    event_type: str = field(default=EventType.MARKET_DATA, init=False)

    exchange: str = ""
    price: float = 0.0
    volume: int = 0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    oi: int = 0


@dataclass
class MarketDepthEvent(Event):
    """Top-5 bid/ask book snapshot."""
    event_type: str = field(default=EventType.MARKET_DEPTH, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    bids: List[Dict[str, Any]] = field(default_factory=list)   # [{price, qty}, …]
    asks: List[Dict[str, Any]] = field(default_factory=list)
    total_buy_qty: int = 0
    total_sell_qty: int = 0
    imbalance_ratio: float = 0.0         # (buy − sell) / (buy + sell)


@dataclass
class OptionChainEvent(Event):
    """Snapshot of the full option chain for an underlying."""
    event_type: str = field(default=EventType.OPTION_CHAIN_UPDATE, init=False)
    priority: int = field(default=EventPriority.NORMAL, init=False)

    underlying: str = ""
    spot_price: float = 0.0
    expiry: str = ""
    chain_data: List[Dict[str, Any]] = field(default_factory=list)
    pcr: float = 0.0                     # put-call ratio
    max_pain: float = 0.0
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    iv_percentile: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Signal Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SignalEvent(Event):
    """
    Trading signal from strategy engine or scanner.

    Carries entry/SL/target levels, confidence, and instrument details
    for Equity, Futures, and Options.
    """
    event_type: str = field(default=EventType.SIGNAL_GENERATED, init=False)
    priority: int = field(default=EventPriority.NORMAL, init=False)

    direction: str = ""                  # BUY / SELL
    strategy_id: str = ""
    strategy_name: str = ""
    confidence: float = 0.0              # 0.0 – 1.0

    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    risk_reward: float = 0.0

    # multi-strategy consensus
    strategies_agreed: List[str] = field(default_factory=list)
    composite_score: float = 0.0

    # context
    regime: str = ""
    timeframe: str = "5m"
    instrument_type: str = ""            # EQUITY, FUTURES, OPTIONS_CE, OPTIONS_PE

    # option-specific
    strike: float = 0.0
    expiry: str = ""
    option_type: str = ""                # CE / PE

    # commodity-specific
    lot_size: int = 0
    tick_size: float = 0.0

    # pass-through: strategy engine may attach a full signal model object
    signal: Any = None

    # v1 backward-compat fields
    signal_type: str = ""                # BUY / SELL / HOLD
    strength: float = 0.0
    indicators_used: List[str] = field(default_factory=list)
    price_target: Optional[float] = None


# Backward-compatible alias
TradingSignalEvent = SignalEvent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Trade-Lifecycle Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class TradeDecisionEvent(Event):
    """Decision to enter or exit — pending risk approval."""
    event_type: str = field(default=EventType.TRADE_DECISION, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    direction: str = ""                  # BUY / SELL
    quantity: int = 0
    order_type: str = "LIMIT"            # MARKET / LIMIT / SL / SL-M
    price: float = 0.0
    trigger_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0

    strategy_id: str = ""
    confidence: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    position_value: float = 0.0

    product_type: str = "INTRADAY"       # INTRADAY / DELIVERY / CARRYFORWARD
    instrument_type: str = "EQUITY"
    exchange: str = "NSE"

    # options / futures
    strike: float = 0.0
    expiry: str = ""
    option_type: str = ""

    # v1 compat
    action: str = ""
    take_profit: Optional[float] = None


@dataclass
class RiskAssessmentEvent(Event):
    """Risk manager's verdict on a proposed trade."""
    event_type: str = field(default=EventType.RISK_APPROVED_TRADE, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    approved: bool = False
    original_quantity: int = 0
    adjusted_quantity: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    position_size_pct: float = 0.0
    portfolio_risk_after: float = 0.0
    kelly_fraction: float = 0.0
    regime: str = ""


@dataclass
class OrderEvent(Event):
    """Order lifecycle update from broker."""
    event_type: str = field(default=EventType.ORDER_STATUS, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    order_id: str = ""
    broker_order_id: str = ""
    direction: str = ""                  # BUY / SELL
    quantity: int = 0
    filled_quantity: int = 0
    pending_quantity: int = 0
    price: float = 0.0
    trigger_price: float = 0.0
    average_price: float = 0.0
    status: str = ""                     # PENDING, PLACED, OPEN, PARTIALLY_FILLED,
    #                                      FILLED, REJECTED, CANCELLED
    order_type: str = ""                 # MARKET / LIMIT / SL / SL-M
    product_type: str = ""               # INTRADAY / DELIVERY
    reject_reason: str = ""
    exchange_timestamp: Optional[datetime] = None

    # v1 compat
    action: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Position Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PositionEvent(Event):
    """Position open / update / close notification."""
    event_type: str = field(default=EventType.POSITION_UPDATED, init=False)
    priority: int = field(default=EventPriority.NORMAL, init=False)

    position_id: str = ""
    direction: str = ""                  # LONG / SHORT
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss: float = 0.0

    unrealised_pnl: float = 0.0
    unrealised_pnl_pct: float = 0.0
    realised_pnl: float = 0.0

    holding_minutes: int = 0
    strategy_id: str = ""
    instrument_type: str = ""

    # stop-loss updates
    previous_stop: float = 0.0
    stop_update_reason: str = ""
    trailing_activated: bool = False

    # exit details (when closing)
    exit_price: float = 0.0
    exit_reason: str = ""                # stop_loss, target, trailing, time_based,
    #                                      force_square_off, manual, circuit_breaker
    exit_type: str = ""
    trade_charges: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Risk Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RiskEvent(Event):
    """Risk-management alert or circuit-breaker trigger."""
    event_type: str = field(default=EventType.RISK_ALERT, init=False)
    priority: int = field(default=EventPriority.CRITICAL, init=False)

    alert_type: str = ""                 # daily_loss, drawdown, margin,
    #                                      circuit_breaker, consecutive_losses,
    #                                      correlation, sector_concentration
    severity: str = "WARNING"            # INFO, WARNING, CRITICAL, HALT
    current_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    action_taken: str = ""               # reduced_size, paused_trading, halted

    # portfolio snapshot at time of alert
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    open_positions: int = 0
    total_exposure: float = 0.0
    margin_used_pct: float = 0.0
    consecutive_losses: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Regime Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RegimeEvent(Event):
    """Market-regime change notification."""
    event_type: str = field(default=EventType.REGIME_CHANGE, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    previous_regime: str = ""
    new_regime: str = ""
    confidence: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0
    index_symbol: str = ""               # NIFTY, BANKNIFTY, …
    indicators: Dict[str, Any] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scanner Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ScanEvent(Event):
    """Scanner sweep result."""
    event_type: str = field(default=EventType.SCAN_COMPLETE, init=False)
    priority: int = field(default=EventPriority.LOW, init=False)

    scan_type: str = ""                  # equity, fno, commodity, full
    stocks_scanned: int = 0
    signals_found: int = 0
    top_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    scan_duration_seconds: float = 0.0
    regime: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Learning Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LearningEvent(Event):
    """End-of-day or per-trade learning-cycle result."""
    event_type: str = field(default=EventType.LEARNING_CYCLE_COMPLETE, init=False)
    priority: int = field(default=EventPriority.LOW, init=False)

    cycle_type: str = ""                 # per_trade, end_of_day, weekly
    trades_analysed: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    strategy_updates: Dict[str, Any] = field(default_factory=dict)
    parameter_updates: Dict[str, Any] = field(default_factory=dict)
    mistakes_found: List[Dict[str, Any]] = field(default_factory=list)
    report_path: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Session Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SessionEvent(Event):
    """Session lifecycle marker (open, close, warnings)."""
    event_type: str = field(default=EventType.SESSION_START, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    session_type: str = ""               # market_open, market_close,
    #                                      pre_open, square_off_warning, eod
    capital: float = 0.0
    positions_count: int = 0
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0
    trades_today: int = 0
    win_rate_today: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  System Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SystemEvent(Event):
    """System-level error or warning."""
    event_type: str = field(default=EventType.SYSTEM_ERROR, init=False)
    priority: int = field(default=EventPriority.CRITICAL, init=False)

    severity: str = "ERROR"              # INFO, WARNING, ERROR, CRITICAL
    component: str = ""
    message: str = ""
    error_type: str = ""
    traceback_str: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionEvent(Event):
    """Broker / WebSocket / DB connection status."""
    event_type: str = field(default=EventType.CONNECTION_STATUS, init=False)
    priority: int = field(default=EventPriority.HIGH, init=False)

    connection_type: str = ""            # broker_api, websocket, database
    status: str = ""                     # connected, disconnected, reconnecting, failed
    attempt: int = 0
    max_attempts: int = 0
    error_message: str = ""
    latency_ms: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Abstract Handler (backward-compatible)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EventHandler(ABC):
    """
    Abstract base for class-based event handlers.

    Implementations may be synchronous or asynchronous —
    the EventBus detects which and dispatches accordingly.
    """

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Internal types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class _Subscription:
    """One handler registered for one-or-more event types."""
    handler: Callable
    sub_id: str
    event_types: Set[str]
    is_async: bool = False
    symbol_filter: Optional[str] = None
    segment_filter: Optional[str] = None
    source_filter: Optional[str] = None
    once: bool = False


@dataclass
class _Metrics:
    total_emitted: int = 0
    total_dispatched: int = 0
    total_errors: int = 0
    events_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_handler: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_emit_epoch: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Event Bus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EventBus:
    """
    Central publish-subscribe event bus.

    Fast path
    ---------
    Subscriptions are indexed by event-type string, so ``emit()`` does
    **O(matched handlers)** work — no scanning of unrelated subscriptions.

    Thread safety
    -------------
    The handler registry is guarded by an ``RLock``.  ``emit()`` itself
    iterates a snapshot so handlers may subscribe/unsubscribe during dispatch
    without deadlocking.

    Sync / async
    -------------
    * ``emit()``       — synchronous dispatch (main trading loop)
    * ``emit_async()`` — asynchronous dispatch (WebSocket data feed)
    * ``publish()``    — enqueue for background async processing

    Both sync and async handlers are accepted; the bus detects and
    dispatches each appropriately.
    """

    def __init__(self, history_size: int = 5000):
        self._lock = threading.RLock()

        # type-indexed for O(matched) dispatch
        self._type_index: Dict[str, List[_Subscription]] = defaultdict(list)
        self._wildcard_subs: List[_Subscription] = []
        self._sub_by_id: Dict[str, _Subscription] = {}

        # auditing
        self._history: Deque[Event] = deque(maxlen=max(history_size, 100))
        self._m = _Metrics()

        # async queue (activated by start())
        self._running = False
        self._async_queue: Optional[asyncio.Queue] = None
        self._processor_task: Optional[asyncio.Task] = None

    # ────────────────────────────── subscribe ──

    def subscribe(
        self,
        event_type: Union[EventType, str, List[Union[EventType, str]]],
        handler: Union[Callable, EventHandler],
        *,
        symbol: Optional[str] = None,
        segment: Optional[str] = None,
        source: Optional[str] = None,
        once: bool = False,
    ) -> str:
        """
        Register a handler for one-or-more event types.

        Parameters
        ----------
        event_type : EventType, str, or list thereof
            The type(s) to listen for.  Use ``EventType.ALL`` (or ``"*"``)
            to receive every event.
        handler : callable or EventHandler
            Receives a single ``Event`` argument.  May be sync or async.
        symbol : str, optional
            Only deliver events whose ``symbol`` matches.
        segment : str, optional
            Only deliver events whose ``segment`` matches (NSE, NFO, MCX …).
        source : str, optional
            Only deliver events whose ``source`` matches.
        once : bool
            Auto-unsubscribe after the first matching dispatch.

        Returns
        -------
        str
            Subscription ID for later ``unsubscribe()``.
        """
        actual: Callable
        if isinstance(handler, EventHandler):
            actual = handler.handle_event
        elif callable(handler):
            actual = handler
        else:
            raise TypeError(f"handler must be callable or EventHandler, got {type(handler)}")

        if isinstance(event_type, (list, tuple, set)):
            raw_types = event_type
        else:
            raw_types = [event_type]

        types_set: Set[str] = set()
        for t in raw_types:
            types_set.add(t.value if isinstance(t, EventType) else str(t))

        sub_id = uuid.uuid4().hex[:10]
        sub = _Subscription(
            handler=actual,
            sub_id=sub_id,
            event_types=types_set,
            is_async=asyncio.iscoroutinefunction(actual),
            symbol_filter=symbol,
            segment_filter=segment,
            source_filter=source,
            once=once,
        )

        with self._lock:
            self._sub_by_id[sub_id] = sub
            if "*" in types_set:
                self._wildcard_subs.append(sub)
            else:
                for et in types_set:
                    self._type_index[et].append(sub)

        logger.debug(
            "Subscribed %s → %s  id=%s  filters=sym:%s/seg:%s",
            getattr(actual, "__qualname__", repr(actual)),
            types_set,
            sub_id,
            symbol or "*",
            segment or "*",
        )
        return sub_id

    # ────────────────────────────── unsubscribe ──

    def unsubscribe(
        self,
        sub_id_or_handler: Union[str, Callable, EventHandler],
    ) -> bool:
        """
        Remove a subscription by ID, callable reference, or EventHandler.

        Returns ``True`` if at least one subscription was removed.
        """
        with self._lock:
            if isinstance(sub_id_or_handler, str):
                sub = self._sub_by_id.pop(sub_id_or_handler, None)
                if sub:
                    self._purge(sub)
                    return True
                return False

            ref = sub_id_or_handler
            if isinstance(ref, EventHandler):
                ref = ref.handle_event

            to_remove = [
                sid for sid, s in self._sub_by_id.items()
                if s.handler is ref
            ]
            for sid in to_remove:
                s = self._sub_by_id.pop(sid)
                self._purge(s)

            return len(to_remove) > 0

    def _purge(self, sub: _Subscription) -> None:
        """Remove *sub* from every index it appears in.  Caller holds lock."""
        if "*" in sub.event_types:
            self._wildcard_subs = [
                s for s in self._wildcard_subs if s.sub_id != sub.sub_id
            ]
        else:
            for et in sub.event_types:
                if et in self._type_index:
                    self._type_index[et] = [
                        s for s in self._type_index[et] if s.sub_id != sub.sub_id
                    ]

    # ────────────────────────────── emit (sync) ──

    def emit(self, event: Event) -> int:
        """
        Dispatch *event* synchronously to every matching handler.

        * Sync handlers are called inline.
        * Async handlers are scheduled on the running event-loop (if one
          exists) or executed via ``asyncio.run()`` as a fallback.
        * Handler exceptions are caught, logged, and counted — dispatch
          continues to subsequent handlers.

        Returns the number of handlers successfully invoked.
        """
        et = self._normalise_type(event)

        self._m.total_emitted += 1
        self._m.events_by_type[et] += 1
        self._history.append(event)

        # snapshot under lock to avoid mutation during iteration
        with self._lock:
            candidates = list(self._type_index.get(et, []))
            candidates.extend(self._wildcard_subs)

        dispatched = 0
        spent: List[str] = []

        for sub in candidates:
            if not self._passes_filters(sub, event):
                continue

            try:
                if sub.is_async:
                    self._dispatch_async_from_sync(sub.handler, event)
                else:
                    sub.handler(event)
                dispatched += 1
            except Exception:
                self._m.total_errors += 1
                name = getattr(sub.handler, "__qualname__", repr(sub.handler))
                self._m.errors_by_handler[name] = (
                    self._m.errors_by_handler.get(name, 0) + 1
                )
                logger.exception(
                    "Handler error [%s] on %s(%s)",
                    name, et, getattr(event, "symbol", ""),
                )

            if sub.once:
                spent.append(sub.sub_id)

        if spent:
            with self._lock:
                for sid in spent:
                    s = self._sub_by_id.pop(sid, None)
                    if s:
                        self._purge(s)

        self._m.total_dispatched += dispatched
        self._m.last_emit_epoch = _time.monotonic()
        return dispatched

    # ────────────────────────────── emit_async ──

    async def emit_async(self, event: Event) -> int:
        """
        Dispatch *event* asynchronously.

        Async handlers are ``await``-ed; sync handlers are called directly.
        """
        et = self._normalise_type(event)

        self._m.total_emitted += 1
        self._m.events_by_type[et] += 1
        self._history.append(event)

        with self._lock:
            candidates = list(self._type_index.get(et, []))
            candidates.extend(self._wildcard_subs)

        dispatched = 0
        spent: List[str] = []

        for sub in candidates:
            if not self._passes_filters(sub, event):
                continue
            try:
                if sub.is_async:
                    await sub.handler(event)
                else:
                    sub.handler(event)
                dispatched += 1
            except Exception:
                self._m.total_errors += 1
                logger.exception("Async handler error on %s", et)

            if sub.once:
                spent.append(sub.sub_id)

        if spent:
            with self._lock:
                for sid in spent:
                    s = self._sub_by_id.pop(sid, None)
                    if s:
                        self._purge(s)

        self._m.total_dispatched += dispatched
        return dispatched

    # ────────────────────────────── async queue ──

    async def publish(self, event: Event) -> None:
        """
        Enqueue *event* for async processing.

        Requires ``start()`` to have been called; otherwise falls back
        to ``emit_async()``.
        """
        if self._async_queue is not None and self._running:
            await self._async_queue.put(event)
        else:
            await self.emit_async(event)

    async def start(self) -> None:
        """Start the background async queue processor."""
        if self._running:
            return
        self._running = True
        self._async_queue = asyncio.Queue()
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("EventBus async processor started")

    async def stop(self) -> None:
        """Stop the background async queue processor gracefully."""
        self._running = False
        if self._processor_task is not None:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        self._processor_task = None
        self._async_queue = None
        logger.info("EventBus async processor stopped")

    async def _process_queue(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._async_queue.get(), timeout=1.0,
                )
                await self.emit_async(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Async queue processor error")

    # ────────────────────────────── internal helpers ──

    @staticmethod
    def _normalise_type(event: Event) -> str:
        et = event.event_type
        return et.value if isinstance(et, EventType) else str(et)

    @staticmethod
    def _passes_filters(sub: _Subscription, event: Event) -> bool:
        if sub.symbol_filter and getattr(event, "symbol", "") != sub.symbol_filter:
            return False
        if sub.segment_filter and getattr(event, "segment", "") != sub.segment_filter:
            return False
        if sub.source_filter and getattr(event, "source", "") != sub.source_filter:
            return False
        return True

    @staticmethod
    def _dispatch_async_from_sync(handler: Callable, event: Event) -> None:
        """Best-effort scheduling of an async handler from a sync context."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(handler(event))
        except RuntimeError:
            try:
                asyncio.run(handler(event))
            except Exception:
                logger.exception("Failed to run async handler from sync context")

    # ────────────────────────────── queries ──

    @property
    def metrics(self) -> Dict[str, Any]:
        """Snapshot of dispatch metrics."""
        return {
            "total_emitted": self._m.total_emitted,
            "total_dispatched": self._m.total_dispatched,
            "total_errors": self._m.total_errors,
            "events_by_type": dict(self._m.events_by_type),
            "errors_by_handler": dict(self._m.errors_by_handler),
            "active_subscriptions": len(self._sub_by_id),
            "history_size": len(self._history),
        }

    @property
    def subscription_count(self) -> int:
        return len(self._sub_by_id)

    @property
    def history(self) -> List[Event]:
        """Full event history (oldest first)."""
        return list(self._history)

    def recent_events(
        self,
        event_type: Optional[Union[EventType, str]] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Query recent events with optional filters (newest first)."""
        et_str: Optional[str] = None
        if event_type is not None:
            et_str = event_type.value if isinstance(event_type, EventType) else event_type

        results: List[Event] = []
        for ev in reversed(self._history):
            if et_str and self._normalise_type(ev) != et_str:
                continue
            if symbol and getattr(ev, "symbol", "") != symbol:
                continue
            results.append(ev)
            if len(results) >= limit:
                break
        return results

    def last_event(
        self,
        event_type: Union[EventType, str],
        symbol: Optional[str] = None,
    ) -> Optional[Event]:
        """Most recent event of the given type."""
        events = self.recent_events(event_type=event_type, symbol=symbol, limit=1)
        return events[0] if events else None

    def has_subscribers(self, event_type: Union[EventType, str]) -> bool:
        et = event_type.value if isinstance(event_type, EventType) else event_type
        return bool(self._type_index.get(et)) or bool(self._wildcard_subs)

    # ────────────────────────────── housekeeping ──

    def clear_history(self) -> None:
        self._history.clear()

    def reset_metrics(self) -> None:
        self._m = _Metrics()

    def clear_all(self) -> None:
        """Remove every subscription, history entry, and metric. For testing."""
        with self._lock:
            self._type_index.clear()
            self._wildcard_subs.clear()
            self._sub_by_id.clear()
        self._history.clear()
        self._m = _Metrics()

    def __repr__(self) -> str:
        return (
            f"EventBus(subs={len(self._sub_by_id)}, "
            f"emitted={self._m.total_emitted}, "
            f"errors={self._m.total_errors})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Factory helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_event(
    event_type: Union[EventType, str],
    *,
    source: str = "",
    symbol: str = "",
    segment: str = "",
    parent: Optional[Event] = None,
    **extra,
) -> Event:
    """
    Convenience factory that auto-propagates the correlation chain
    from *parent* and puts arbitrary *extra* kwargs into ``data``.

    Usage::

        tick = TickEvent(symbol="RELIANCE", ltp=2450)
        generic = create_event(
            EventType.OPPORTUNITY_FOUND,
            source="scanner",
            parent=tick,
            score=87.5,
            strategies=["ORB", "VWAP"],
        )
    """
    chain: Dict[str, Any] = parent.chain() if parent else {}

    et = event_type.value if isinstance(event_type, EventType) else event_type

    return Event(
        event_type=et,
        source=source,
        symbol=symbol or chain.pop("symbol", ""),
        segment=segment or chain.pop("segment", ""),
        correlation_id=chain.pop("correlation_id", ""),
        causation_id=chain.pop("causation_id", ""),
        data=extra,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Module-level singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

event_bus = EventBus()