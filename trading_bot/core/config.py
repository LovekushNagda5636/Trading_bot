"""
Configuration management for the trading bot.

Handles all configuration for Equity, F&O (NFO), Commodity (MCX), and Currency (CDS)
trading with:
- Segment-specific market hours, transaction costs, and risk limits
- Angel One SmartAPI credential loading with env-var overrides for secrets
- Thread-safe runtime updates with validation and rollback
- Hot-reload, backup/restore, and change-notification hooks
- Self-learning engine parameter management
"""

import os
import json
import copy
import hashlib
import shutil
import threading
import logging
from dataclasses import dataclass, field, fields, asdict
from datetime import time, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Segment(str, Enum):
    """Exchange segments supported by Angel One."""
    NSE = "NSE"           # NSE Cash
    BSE = "BSE"           # BSE Cash
    NFO = "NFO"           # NSE F&O
    MCX = "MCX"           # Multi Commodity Exchange
    CDS = "CDS"           # Currency Derivatives
    BFO = "BFO"           # BSE F&O


class InstrumentType(str, Enum):
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS_CE = "OPTIONS_CE"
    OPTIONS_PE = "OPTIONS_PE"
    INDEX = "INDEX"
    COMMODITY = "COMMODITY"


class MarketRegime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class PositionSizingMethod(str, Enum):
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"


class StopLossMethod(str, Enum):
    ATR = "atr"
    PERCENTAGE = "percentage"
    SUPPORT_RESISTANCE = "support_resistance"
    COMBINED = "combined"


class TrailingMethod(str, Enum):
    FIXED = "fixed"
    ATR = "atr"
    PARABOLIC = "parabolic"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper – safe dataclass construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _safe_construct(cls: Type, data: dict) -> Any:
    """
    Build a dataclass instance from *data*, silently dropping keys that
    do not exist as fields and coercing values to the declared type.
    """
    if not data or not isinstance(data, dict):
        return cls()

    known = {f.name: f for f in fields(cls)}
    kwargs = {}

    for key, value in data.items():
        if key not in known:
            continue
        expected_type = type(getattr(cls(), key))          # runtime default type
        try:
            if expected_type is bool and isinstance(value, str):
                kwargs[key] = value.lower() in ("true", "1", "yes")
            elif expected_type is bool:
                kwargs[key] = bool(value)
            elif isinstance(value, dict) or isinstance(value, list):
                kwargs[key] = value                         # keep dicts/lists as-is
            else:
                kwargs[key] = expected_type(value)
        except (ValueError, TypeError):
            kwargs[key] = value                             # fallback – store raw

    return cls(**kwargs)


def _dataclass_to_dict(obj: Any) -> dict:
    """Recursively convert a dataclass to a JSON-serialisable dict."""
    if not hasattr(obj, "__dataclass_fields__"):
        return obj

    result: Dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if hasattr(value, "__dataclass_fields__"):
            result[f.name] = _dataclass_to_dict(value)
        elif isinstance(value, Decimal):
            result[f.name] = float(value)
        elif isinstance(value, Enum):
            result[f.name] = value.value
        elif isinstance(value, time):
            result[f.name] = value.strftime("%H:%M")
        elif isinstance(value, list):
            result[f.name] = [
                _dataclass_to_dict(v) if hasattr(v, "__dataclass_fields__") else v
                for v in value
            ]
        else:
            result[f.name] = value
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Market Hours
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class MarketHours:
    """Immutable trading-hours definition for a single exchange segment."""
    pre_open_start: time = time(9, 0)
    pre_open_end: time = time(9, 8)
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    post_close_end: time = time(16, 0)
    no_new_entry_after: time = time(14, 30)
    square_off_warning: time = time(14, 45)
    force_square_off: time = time(15, 20)

    # ── queries ──

    def is_market_open(self, now: time) -> bool:
        """True while the continuous session is running."""
        return self.market_open <= now <= self.market_close

    def is_pre_open(self, now: time) -> bool:
        return self.pre_open_start <= now < self.market_open

    def is_entry_allowed(self, now: time) -> bool:
        """True while new positions may be opened."""
        return self.market_open <= now <= self.no_new_entry_after

    def should_tighten_stops(self, now: time) -> bool:
        return now >= self.square_off_warning

    def should_force_exit(self, now: time) -> bool:
        return now >= self.force_square_off

    def minutes_to_close(self, now: time) -> int:
        """Minutes remaining until market close, negative if already closed."""
        now_mins = now.hour * 60 + now.minute
        close_mins = self.market_close.hour * 60 + self.market_close.minute
        return close_mins - now_mins


NSE_HOURS = MarketHours(
    pre_open_start=time(9, 0),
    pre_open_end=time(9, 8),
    market_open=time(9, 15),
    market_close=time(15, 30),
    post_close_end=time(16, 0),
    no_new_entry_after=time(14, 30),
    square_off_warning=time(14, 45),
    force_square_off=time(15, 20),
)

NFO_HOURS = MarketHours(
    pre_open_start=time(9, 0),
    pre_open_end=time(9, 8),
    market_open=time(9, 15),
    market_close=time(15, 30),
    post_close_end=time(16, 0),
    no_new_entry_after=time(14, 30),
    square_off_warning=time(14, 45),
    force_square_off=time(15, 20),
)

MCX_HOURS = MarketHours(
    pre_open_start=time(9, 0),
    pre_open_end=time(9, 0),
    market_open=time(9, 0),
    market_close=time(23, 30),
    post_close_end=time(23, 55),
    no_new_entry_after=time(23, 0),
    square_off_warning=time(23, 10),
    force_square_off=time(23, 25),
)

CDS_HOURS = MarketHours(
    pre_open_start=time(9, 0),
    pre_open_end=time(9, 0),
    market_open=time(9, 0),
    market_close=time(17, 0),
    post_close_end=time(17, 30),
    no_new_entry_after=time(16, 30),
    square_off_warning=time(16, 40),
    force_square_off=time(16, 55),
)

SEGMENT_HOURS: Dict[str, MarketHours] = {
    Segment.NSE: NSE_HOURS,
    Segment.BSE: NSE_HOURS,
    Segment.NFO: NFO_HOURS,
    Segment.MCX: MCX_HOURS,
    Segment.CDS: CDS_HOURS,
    Segment.BFO: NFO_HOURS,
    # Also accept plain strings that older code may pass:
    "NSE": NSE_HOURS,
    "BSE": NSE_HOURS,
    "NFO": NFO_HOURS,
    "MCX": MCX_HOURS,
    "CDS": CDS_HOURS,
    "BFO": NFO_HOURS,
    "EQUITY": NSE_HOURS,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Transaction Costs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class TransactionCosts:
    """
    Full round-trip cost model for a given segment / product combination.
    All percentage fields are expressed as decimals (0.01 = 1 %).
    """
    brokerage_flat: float = 20.0               # ₹ per executed order
    brokerage_pct: float = 0.0003              # 0.03 %  (cheaper of flat / pct used)
    stt_buy_pct: float = 0.0                   # equity intraday: 0 on buy
    stt_sell_pct: float = 0.00025              # 0.025 % on sell
    exchange_txn_pct: float = 0.0000345
    gst_pct: float = 0.18                      # on (brokerage + exchange charges)
    sebi_per_crore: float = 10.0               # ₹ 10 per crore turnover
    stamp_duty_buy_pct: float = 0.00015        # state dependent, conservative

    def round_trip_cost(
        self,
        buy_value: float,
        sell_value: float,
        orders: int = 2,
    ) -> float:
        """Total charges for a complete buy → sell (or sell → buy) cycle."""
        turnover = buy_value + sell_value

        brokerage = min(
            orders * self.brokerage_flat,
            turnover * self.brokerage_pct,
        )
        stt = buy_value * self.stt_buy_pct + sell_value * self.stt_sell_pct
        exchange = turnover * self.exchange_txn_pct
        gst = (brokerage + exchange) * self.gst_pct
        sebi = turnover * self.sebi_per_crore / 1e7
        stamp = buy_value * self.stamp_duty_buy_pct

        return brokerage + stt + exchange + gst + sebi + stamp

    def breakeven_move_pct(self, position_value: float) -> float:
        """Minimum price-change % needed to cover costs on a round trip."""
        if position_value <= 0:
            return 0.0
        cost = self.round_trip_cost(position_value, position_value)
        return (cost / position_value) * 100.0


EQUITY_INTRADAY_COSTS = TransactionCosts(
    brokerage_flat=20.0, brokerage_pct=0.0003,
    stt_buy_pct=0.0, stt_sell_pct=0.00025,
    exchange_txn_pct=0.0000345, gst_pct=0.18,
    sebi_per_crore=10.0, stamp_duty_buy_pct=0.00015,
)

EQUITY_DELIVERY_COSTS = TransactionCosts(
    brokerage_flat=20.0, brokerage_pct=0.0003,
    stt_buy_pct=0.001, stt_sell_pct=0.001,
    exchange_txn_pct=0.0000345, gst_pct=0.18,
    sebi_per_crore=10.0, stamp_duty_buy_pct=0.00015,
)

FNO_FUTURES_COSTS = TransactionCosts(
    brokerage_flat=20.0, brokerage_pct=0.0003,
    stt_buy_pct=0.0, stt_sell_pct=0.000125,
    exchange_txn_pct=0.000002, gst_pct=0.18,
    sebi_per_crore=10.0, stamp_duty_buy_pct=0.00002,
)

FNO_OPTIONS_COSTS = TransactionCosts(
    brokerage_flat=20.0, brokerage_pct=0.0003,
    stt_buy_pct=0.0, stt_sell_pct=0.000625,
    exchange_txn_pct=0.0000053, gst_pct=0.18,
    sebi_per_crore=10.0, stamp_duty_buy_pct=0.00003,
)

MCX_COMMODITY_COSTS = TransactionCosts(
    brokerage_flat=20.0, brokerage_pct=0.0003,
    stt_buy_pct=0.0, stt_sell_pct=0.0001,
    exchange_txn_pct=0.000026, gst_pct=0.18,
    sebi_per_crore=10.0, stamp_duty_buy_pct=0.00002,
)

CDS_CURRENCY_COSTS = TransactionCosts(
    brokerage_flat=20.0, brokerage_pct=0.0003,
    stt_buy_pct=0.0, stt_sell_pct=0.0,
    exchange_txn_pct=0.0000009, gst_pct=0.18,
    sebi_per_crore=10.0, stamp_duty_buy_pct=0.00001,
)

SEGMENT_COSTS: Dict[str, TransactionCosts] = {
    "EQUITY_INTRADAY": EQUITY_INTRADAY_COSTS,
    "EQUITY_DELIVERY": EQUITY_DELIVERY_COSTS,
    "FNO_FUTURES": FNO_FUTURES_COSTS,
    "FNO_OPTIONS": FNO_OPTIONS_COSTS,
    "MCX": MCX_COMMODITY_COSTS,
    "CDS": CDS_CURRENCY_COSTS,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Risk-Limits Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RiskLimitsConfig:
    """Portfolio-wide and per-position risk guardrails."""

    # ── portfolio level ──
    max_portfolio_risk_pct: float = 0.02         # 2 % of capital at risk at any moment
    max_daily_loss_pct: float = 0.03             # 3 % daily loss → stop trading
    max_drawdown_pct: float = 0.10               # 10 % peak-to-trough
    circuit_breaker_loss_pct: float = 0.02       # 2 % → pause, reduce size
    hard_stop_loss_pct: float = 0.05             # 5 % → halt all activity
    min_account_balance: float = 10000.0

    # ── position level ──
    max_position_risk_pct: float = 0.01          # 1 % risk per trade
    max_position_size_pct: float = 0.20          # single position ≤ 20 % of capital
    min_position_size_pct: float = 0.02
    max_positions: int = 5
    max_positions_per_sector: int = 2
    max_correlated_positions: int = 3
    max_correlation: float = 0.70

    # ── segment-level caps ──
    max_equity_exposure_pct: float = 0.60
    max_fno_exposure_pct: float = 0.40
    max_commodity_exposure_pct: float = 0.30
    max_options_premium_pct: float = 0.10

    # ── consecutive-loss circuit-breaker ──
    max_consecutive_losses: int = 3
    consecutive_loss_size_reduction: float = 0.50    # halve size after streak
    recovery_trades_at_reduced_size: int = 2         # stay small for N wins

    # ── daily limits ──
    daily_trade_limit: int = 15
    daily_options_trade_limit: int = 10

    # ── signal quality gates ──
    min_risk_reward_ratio: float = 1.5
    min_signal_confidence: float = 0.60

    # ── regime multipliers ──
    regime_position_multipliers: Dict[str, float] = field(default_factory=lambda: {
        MarketRegime.TRENDING_BULL.value: 1.2,
        MarketRegime.TRENDING_BEAR.value: 0.8,
        MarketRegime.SIDEWAYS.value: 0.6,
        MarketRegime.HIGH_VOLATILITY.value: 0.5,
        MarketRegime.LOW_VOLATILITY.value: 0.7,
    })

    def regime_multiplier(self, regime: str) -> float:
        return self.regime_position_multipliers.get(regime, 1.0)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.max_daily_loss_pct <= 0:
            errors.append("max_daily_loss_pct must be > 0")
        if self.max_position_size_pct > 0.50:
            errors.append("max_position_size_pct should not exceed 50 %")
        if self.min_risk_reward_ratio < 1.0:
            errors.append("min_risk_reward_ratio must be ≥ 1.0")
        if self.max_positions < 1:
            errors.append("max_positions must be ≥ 1")
        if self.max_consecutive_losses < 1:
            errors.append("max_consecutive_losses must be ≥ 1")
        if not (0 < self.max_portfolio_risk_pct <= 0.10):
            errors.append("max_portfolio_risk_pct should be between 0 and 10 %")
        if self.circuit_breaker_loss_pct >= self.hard_stop_loss_pct:
            errors.append("circuit_breaker must trigger before hard_stop")
        return errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Position Sizing Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PositionSizingConfig:
    method: str = PositionSizingMethod.KELLY.value

    # Kelly
    kelly_fraction: float = 0.25                 # use 25 % of full Kelly
    min_win_rate_for_kelly: float = 0.45
    min_trades_for_kelly: int = 30

    # Fixed
    fixed_risk_per_trade_pct: float = 0.02       # 2 % risk when using fixed

    # Volatility targeting
    volatility_target_annual: float = 0.15       # 15 % annualised
    risk_free_rate: float = 0.065                # RBI repo rate

    # Bounds
    max_position_value_pct: float = 0.20
    min_position_value: float = 500.0            # ₹ min (transaction-cost floor)

    # Scaling
    scale_in_enabled: bool = False
    scale_in_max_adds: int = 2
    scale_in_size_pct: float = 0.50              # each add = 50 % of initial

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not (0 < self.kelly_fraction <= 1.0):
            errors.append("kelly_fraction must be (0, 1]")
        if not (0 < self.fixed_risk_per_trade_pct <= 0.05):
            errors.append("fixed_risk_per_trade_pct must be (0, 5 %]")
        if self.min_position_value < 0:
            errors.append("min_position_value must be ≥ 0")
        return errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stop-Loss Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class StopLossConfig:
    # ── initial stop ──
    method: str = StopLossMethod.ATR.value
    atr_period: int = 14
    atr_multiplier: float = 1.5
    percentage_stop: float = 0.0075              # 0.75 % fallback
    max_stop_distance_pct: float = 0.020
    min_stop_distance_pct: float = 0.003

    # ── trailing ──
    trailing_enabled: bool = True
    trailing_method: str = TrailingMethod.ATR.value
    trailing_activation_ratio: float = 1.0       # activate after 1 × risk in profit
    trailing_distance_pct: float = 0.004
    trailing_ratchet_pct: float = 0.005          # lock every 0.5 % gain
    trailing_atr_multiplier: float = 1.0

    # ── time-based ──
    no_profit_timeout_minutes: int = 30
    afternoon_tighten_time: str = "14:30"
    afternoon_tighten_factor: float = 0.70
    force_exit_time: str = "15:20"

    # ── volatility adjustment ──
    high_vol_multiplier: float = 1.3
    low_vol_multiplier: float = 0.8
    vol_threshold_high: float = 0.25
    vol_threshold_low: float = 0.10

    # ── partial exits ──
    partial_exit_enabled: bool = True
    partial_exit_fraction: float = 0.50
    target_1_atr_mult: float = 1.5
    target_2_atr_mult: float = 3.0
    target_3_atr_mult: float = 5.0

    # ── convenience ──

    def tighten_time(self) -> time:
        h, m = map(int, self.afternoon_tighten_time.split(":"))
        return time(h, m)

    def exit_time(self) -> time:
        h, m = map(int, self.force_exit_time.split(":"))
        return time(h, m)

    def volatility_multiplier(self, annualised_vol: float) -> float:
        """Return a stop-distance multiplier based on current volatility."""
        if annualised_vol >= self.vol_threshold_high:
            return self.high_vol_multiplier
        if annualised_vol <= self.vol_threshold_low:
            return self.low_vol_multiplier
        return 1.0

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.atr_multiplier <= 0:
            errors.append("atr_multiplier must be > 0")
        if self.max_stop_distance_pct < self.min_stop_distance_pct:
            errors.append("max_stop_distance must be ≥ min_stop_distance")
        if not (0 < self.partial_exit_fraction < 1):
            errors.append("partial_exit_fraction must be in (0, 1)")
        if self.trailing_activation_ratio <= 0:
            errors.append("trailing_activation_ratio must be > 0")
        return errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  F&O Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class FnOConfig:
    """Futures & Options specific parameters."""
    enabled: bool = True

    # ── indices to trade ──
    trade_nifty: bool = True
    trade_banknifty: bool = True
    trade_finnifty: bool = False
    trade_midcpnifty: bool = False
    trade_sensex: bool = False

    # ── stock F&O ──
    trade_stock_futures: bool = True
    trade_stock_options: bool = True
    max_stock_fno_positions: int = 3

    # ── lot sizes (updated from Angel One instrument master) ──
    index_lot_sizes: Dict[str, int] = field(default_factory=lambda: {
        "NIFTY": 25,
        "BANKNIFTY": 15,
        "FINNIFTY": 25,
        "MIDCPNIFTY": 50,
        "SENSEX": 10,
    })

    # ── option quality filters ──
    min_option_volume: int = 1000
    min_option_oi: int = 50000
    max_bid_ask_spread_pct: float = 0.02         # 2 % max spread
    max_premium_per_lot: float = 15000.0
    max_iv_percentile: float = 85.0              # don't buy expensive options
    min_iv_percentile: float = 15.0              # don't sell cheap options

    # ── strike selection ──
    max_otm_strikes: int = 5
    prefer_atm_for_directional: bool = True

    # ── expiry rules ──
    trade_weekly_expiry: bool = True
    trade_monthly_expiry: bool = True
    min_dte: int = 0                             # 0 = may trade on expiry day
    max_dte: int = 30
    avoid_new_entry_on_expiry_after: str = "14:00"

    # ── Greeks caps ──
    max_portfolio_delta: float = 0.80
    max_portfolio_gamma: float = 0.10
    max_daily_theta_decay: float = 500.0         # ₹
    min_vega_for_vol_trades: float = 5.0

    # ── margin management ──
    margin_buffer_pct: float = 0.20              # keep 20 % margin head-room
    max_margin_utilisation_pct: float = 0.80

    def expiry_cutoff_time(self) -> time:
        h, m = map(int, self.avoid_new_entry_on_expiry_after.split(":"))
        return time(h, m)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Commodity (MCX) Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CommodityConfig:
    enabled: bool = False

    # ── which commodities ──
    trade_gold: bool = True
    trade_silver: bool = True
    trade_crude_oil: bool = True
    trade_natural_gas: bool = True
    trade_copper: bool = False
    trade_zinc: bool = False
    trade_aluminium: bool = False
    trade_lead: bool = False
    trade_nickel: bool = False

    # ── contract specifications (lot × unit-qty in the lot) ──
    lot_sizes: Dict[str, int] = field(default_factory=lambda: {
        "GOLD": 100,        "GOLDM": 10,        "GOLDPETAL": 1,
        "GOLDGUINEA": 1,
        "SILVER": 30,       "SILVERM": 5,        "SILVERMIC": 1,
        "CRUDEOIL": 100,    "CRUDEOILM": 10,
        "NATURALGAS": 1250, "NATURALGASM": 250,
        "COPPER": 2500,     "COPPERM": 250,
        "ZINC": 5000,       "ZINCMINI": 1000,
        "ALUMINIUM": 5000,  "ALUMINI": 1000,
        "LEAD": 5000,       "LEADMINI": 1000,
        "NICKEL": 1500,     "NICKELMINI": 100,
    })

    tick_sizes: Dict[str, float] = field(default_factory=lambda: {
        "GOLD": 1.0,   "GOLDM": 1.0,   "GOLDPETAL": 1.0,
        "SILVER": 1.0,  "SILVERM": 1.0,  "SILVERMIC": 1.0,
        "CRUDEOIL": 1.0, "CRUDEOILM": 1.0,
        "NATURALGAS": 0.10, "NATURALGASM": 0.10,
        "COPPER": 0.05,  "ZINC": 0.05,  "ALUMINIUM": 0.05,
        "LEAD": 0.05,   "NICKEL": 0.10,
    })

    # ── per-commodity risk limits (max SL %) ──
    max_sl_pct: Dict[str, float] = field(default_factory=lambda: {
        "GOLD": 0.005,          # 0.5 %
        "GOLDM": 0.005,
        "SILVER": 0.010,        # 1.0 %
        "SILVERM": 0.010,
        "CRUDEOIL": 0.015,      # 1.5 %
        "NATURALGAS": 0.025,    # 2.5 % – very volatile
        "COPPER": 0.010,
        "ZINC": 0.010,
        "ALUMINIUM": 0.010,
        "LEAD": 0.010,
        "NICKEL": 0.015,
    })

    max_commodity_positions: int = 2
    max_commodity_exposure_pct: float = 0.30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class StrategyConfig:
    strategy_id: str = ""
    strategy_type: str = ""                      # breakout, trend, reversion …
    category: str = ""
    enabled: bool = False

    # ── allocation ──
    allocation_pct: float = 0.10
    max_concurrent_positions: int = 2

    # ── quality gates ──
    min_confidence: float = 0.60
    min_risk_reward: float = 1.5

    # ── applicability ──
    segments: List[str] = field(default_factory=lambda: [Segment.NSE.value, Segment.NFO.value])
    preferred_regimes: List[str] = field(
        default_factory=lambda: [MarketRegime.TRENDING_BULL.value, MarketRegime.TRENDING_BEAR.value]
    )
    avoid_regimes: List[str] = field(default_factory=list)

    # ── timeframes ──
    primary_timeframe: str = "5m"
    confirmation_timeframes: List[str] = field(default_factory=lambda: ["15m", "1h"])

    # ── strategy-specific knobs ──
    parameters: Dict[str, Any] = field(default_factory=dict)

    # ── adaptive (written by the learning engine) ──
    weight: float = 1.0
    win_rate: float = 0.50
    profit_factor: float = 1.0
    total_trades: int = 0

    def is_suitable_for_regime(self, regime: str) -> bool:
        if regime in self.avoid_regimes:
            return False
        if self.preferred_regimes and regime not in self.preferred_regimes:
            return False
        return True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.strategy_id:
            errors.append("strategy_id is required")
        if not (0 < self.allocation_pct <= 1.0):
            errors.append(f"allocation_pct={self.allocation_pct} must be in (0, 1]")
        if not (0 <= self.min_confidence <= 1.0):
            errors.append("min_confidence must be in [0, 1]")
        if self.min_risk_reward < 0:
            errors.append("min_risk_reward must be ≥ 0")
        return errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Angel One Broker Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class AngelOneConfig:
    # ── credentials (prefer env vars) ──
    api_key: str = ""
    client_id: str = ""
    password: str = ""
    totp_secret: str = ""

    # ── endpoints ──
    base_url: str = "https://apiconnect.angelbroking.com"
    ws_url: str = "wss://smartapisocket.angelone.in/smart-stream"

    # ── rate limits ──
    rate_limit_per_second: int = 10
    rate_limit_per_minute: int = 100
    request_timeout_seconds: int = 30

    # ── websocket ──
    ws_reconnect_interval_seconds: int = 5
    ws_max_reconnect_attempts: int = 10
    ws_heartbeat_interval: int = 30
    max_ws_subscriptions: int = 50

    # ── historical data ──
    max_candles_per_request: int = 2000

    # ── order defaults ──
    default_order_type: str = "LIMIT"
    default_product_type: str = "INTRADAY"       # MIS
    default_variety: str = "NORMAL"

    enabled: bool = True

    # Common key-name variations found in different config files
    _KEY_ALIASES: Dict[str, str] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        object.__setattr__(self, "_KEY_ALIASES", {
            "apiKey": "api_key", "api_key": "api_key",
            "clientId": "client_id", "client_id": "client_id",
            "clientCode": "client_id",
            "mpin": "password", "pin": "password", "password": "password",
            "totpSecret": "totp_secret", "totp_secret": "totp_secret",
            "totp": "totp_secret",
        })

    @classmethod
    def from_dict(cls, data: dict) -> "AngelOneConfig":
        """Construct from a dict that may use non-standard key names."""
        if not data:
            return cls()

        # Flatten common wrappers
        for wrapper in ("angel_one", "angelone", "broker", "credentials"):
            if wrapper in data and isinstance(data[wrapper], dict):
                data = {**data, **data.pop(wrapper)}

        instance = cls()
        aliases = instance._KEY_ALIASES

        for raw_key, value in data.items():
            target_key = aliases.get(raw_key, raw_key)
            if hasattr(instance, target_key) and not target_key.startswith("_"):
                try:
                    expected = type(getattr(instance, target_key))
                    setattr(instance, target_key, expected(value))
                except (ValueError, TypeError):
                    setattr(instance, target_key, value)

        return instance

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.api_key:
            errors.append("Angel One api_key is required")
        if not self.client_id:
            errors.append("Angel One client_id is required")
        if not self.password:
            errors.append("Angel One password is required")
        return errors

    @property
    def credentials_available(self) -> bool:
        return bool(self.api_key and self.client_id and self.password)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scanner Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ScannerConfig:
    scan_interval_seconds: int = 300             # 5 min default
    enabled: bool = True

    # ── what to scan ──
    scan_equity: bool = True
    scan_fno_stocks: bool = True
    scan_index_options: bool = True
    scan_commodities: bool = False

    # ── universe filters ──
    fno_only: bool = True
    nifty50_only: bool = False
    nifty200_only: bool = False
    min_price: float = 50.0
    max_price: float = 50000.0
    min_volume_ratio: float = 1.5                # vs 20-day average

    # ── custom lists ──
    custom_watchlist: List[str] = field(default_factory=list)
    excluded_symbols: List[str] = field(default_factory=list)

    # ── well-known symbols ──
    index_symbols: List[str] = field(default_factory=lambda: [
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    ])
    commodity_symbols: List[str] = field(default_factory=lambda: [
        "GOLD", "GOLDM", "SILVER", "SILVERM",
        "CRUDEOIL", "CRUDEOILM", "NATURALGAS",
    ])

    # ── output ──
    max_signals: int = 20
    min_composite_score: float = 60.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Learning / Adaptive-Engine Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LearningConfig:
    enabled: bool = True

    # ── cycle triggers ──
    learn_after_each_trade: bool = True
    end_of_day_learning: bool = True
    weekly_full_retrain: bool = True

    # ── strategy weight adaptation ──
    weight_learning_rate: float = 0.10
    min_strategy_weight: float = 0.10
    max_strategy_weight: float = 3.0
    min_trades_for_adaptation: int = 10

    # ── score threshold learning ──
    initial_min_score: float = 35.0
    score_learning_rate: float = 0.05

    # ── stop-loss adaptation ──
    sl_learning_rate: float = 0.05
    sl_min_multiplier: float = 0.5
    sl_max_multiplier: float = 3.0

    # ── target adaptation ──
    target_learning_rate: float = 0.05
    target_min_multiplier: float = 0.8
    target_max_multiplier: float = 5.0

    # ── hour-based avoidance ──
    min_trades_per_hour: int = 5
    bad_hour_win_rate_threshold: float = 0.35

    # ── mistake classification thresholds ──
    tight_sl_reversal_pct: float = 0.50          # price recovered > 50 % of SL
    premature_exit_missed_pct: float = 0.30      # left > 30 % on table

    # ── persistence ──
    journal_dir: str = "data/journal"
    models_dir: str = "data/models"
    save_after_each_trade: bool = True
    backup_daily: bool = True
    max_journal_entries: int = 10000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dashboard / Alerts Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    refresh_interval_seconds: int = 5
    ws_enabled: bool = True

    # ── alert toggles ──
    alert_on_trade: bool = True
    alert_on_loss_limit: bool = True
    alert_on_system_error: bool = True
    alert_on_regime_change: bool = True

    # ── telegram ──
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── email ──
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Master Trading Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class TradingConfig:
    """
    Top-level configuration object that aggregates every sub-config.
    The single source of truth for the entire trading system.
    """

    # ── core ──
    environment: str = Environment.DEVELOPMENT.value
    trading_enabled: bool = False
    paper_trading: bool = True
    log_level: str = "INFO"

    # ── capital ──
    total_capital: float = 100000.0
    intraday_capital_pct: float = 0.80
    reserved_capital_pct: float = 0.20

    # ── sub-configurations ──
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    fno: FnOConfig = field(default_factory=FnOConfig)
    commodity: CommodityConfig = field(default_factory=CommodityConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    angel_one: AngelOneConfig = field(default_factory=AngelOneConfig)

    # ── strategies ──
    strategy_configs: List[StrategyConfig] = field(default_factory=list)

    # ── convenience ──

    @property
    def available_capital(self) -> float:
        return self.total_capital * self.intraday_capital_pct

    def active_strategies(self) -> List[StrategyConfig]:
        return [s for s in self.strategy_configs if s.enabled]

    def segment_hours(self, segment: str) -> MarketHours:
        return SEGMENT_HOURS.get(segment, NSE_HOURS)

    def transaction_costs(self, cost_key: str) -> TransactionCosts:
        return SEGMENT_COSTS.get(cost_key, EQUITY_INTRADAY_COSTS)

    # ── full validation ──

    def validate(self) -> List[str]:
        errors: List[str] = []

        if self.total_capital <= 0:
            errors.append("total_capital must be > 0")
        if self.total_capital < self.risk_limits.min_account_balance:
            errors.append(
                f"total_capital ₹{self.total_capital:,.0f} is below "
                f"min_account_balance ₹{self.risk_limits.min_account_balance:,.0f}"
            )
        if not (0 < self.intraday_capital_pct <= 1.0):
            errors.append("intraday_capital_pct must be in (0, 1]")
        if self.intraday_capital_pct + self.reserved_capital_pct > 1.0:
            errors.append("intraday + reserved capital exceeds 100 %")

        errors.extend(self.risk_limits.validate())
        errors.extend(self.position_sizing.validate())
        errors.extend(self.stop_loss.validate())

        if self.trading_enabled and not self.paper_trading:
            errors.extend(self.angel_one.validate())

        total_alloc = sum(s.allocation_pct for s in self.strategy_configs if s.enabled)
        if total_alloc > 1.0:
            errors.append(f"Total strategy allocation {total_alloc:.0%} exceeds 100 %")

        for strat in self.strategy_configs:
            for e in strat.validate():
                errors.append(f"Strategy '{strat.strategy_id}': {e}")

        return errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration Manager  (singleton, thread-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConfigManager:
    """
    Central configuration hub.

    Load order (later sources override earlier ones):
      1. Dataclass defaults
      2. config/angel_one_config.json
      3. config/trading_{environment}.json
      4. config/strategies.json
      5. Environment variables  (ANGEL_API_KEY, etc.)
    """

    _instance: Optional["ConfigManager"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialised = False
                cls._instance = inst
            return cls._instance

    def __init__(
        self,
        config_dir: str = "config",
        data_dir: str = "data",
    ):
        if self._initialised:
            return

        self._config_dir = Path(config_dir)
        self._data_dir = Path(data_dir)
        self._lock = threading.RLock()
        self._watchers: List[Callable[[str, Any], None]] = []
        self._file_hash: str = ""

        # ensure directories
        for d in (
            self._config_dir,
            self._data_dir,
            self._data_dir / "journal",
            self._data_dir / "models",
            self._data_dir / "backups",
        ):
            d.mkdir(parents=True, exist_ok=True)

        self._cfg = TradingConfig()
        self._load_all()

        self._initialised = True
        logger.info("ConfigManager ready  %s", self)

    # ────────────────────────────── public properties ──

    @property
    def config(self) -> TradingConfig:
        return self._cfg

    @property
    def risk(self) -> RiskLimitsConfig:
        return self._cfg.risk_limits

    @property
    def sizing(self) -> PositionSizingConfig:
        return self._cfg.position_sizing

    @property
    def stops(self) -> StopLossConfig:
        return self._cfg.stop_loss

    @property
    def fno(self) -> FnOConfig:
        return self._cfg.fno

    @property
    def commodity(self) -> CommodityConfig:
        return self._cfg.commodity

    @property
    def scanner(self) -> ScannerConfig:
        return self._cfg.scanner

    @property
    def learning(self) -> LearningConfig:
        return self._cfg.learning

    @property
    def angel_one(self) -> AngelOneConfig:
        return self._cfg.angel_one

    @property
    def dashboard(self) -> DashboardConfig:
        return self._cfg.dashboard

    # ────────────────────────────── load pipeline ──

    def _load_all(self) -> None:
        with self._lock:
            try:
                self._load_angel_one_config()
                self._load_trading_env_config()
                self._load_strategy_configs()
                self._apply_env_overrides()
                self._file_hash = self._compute_files_hash()

                errors = self._cfg.validate()
                for e in errors:
                    logger.warning("Config validation ⚠  %s", e)

                logger.info("Configuration loaded successfully")
            except Exception:
                logger.exception("Fatal error loading configuration")
                raise

    # ── angel_one_config.json ──

    def _load_angel_one_config(self) -> None:
        path = self._config_dir / "angel_one_config.json"
        if not path.exists():
            logger.warning("angel_one_config.json not found at %s", path)
            return

        try:
            raw = self._read_json(path)
        except Exception:
            logger.exception("Cannot parse %s", path)
            return

        # Broker credentials
        ao_data: dict = {}
        for wrapper in ("angel_one", "angelone", "broker", "credentials"):
            if wrapper in raw and isinstance(raw[wrapper], dict):
                ao_data.update(raw[wrapper])
        if not ao_data:
            ao_data = raw
        self._cfg.angel_one = AngelOneConfig.from_dict(ao_data)

        # Trading section
        trading = raw.get("trading", {})
        if trading:
            self._map_flat(self._cfg, trading, {
                "budget": ("total_capital", float),
                "paper_trading": ("paper_trading", bool),
            })
            self._map_flat(self._cfg.risk_limits, trading, {
                "max_positions": ("max_positions", int),
                "max_daily_loss_pct": ("max_daily_loss_pct", lambda v: float(v) / 100.0),
            })
            if "square_off_time" in trading:
                self._cfg.stop_loss.force_exit_time = str(trading["square_off_time"])

        # Dedicated sub-sections
        self._merge_section("risk_limits", raw, self._cfg.risk_limits)
        self._merge_section("position_sizing", raw, self._cfg.position_sizing)
        self._merge_section("stop_loss", raw, self._cfg.stop_loss)
        self._merge_section("fno", raw, self._cfg.fno)
        self._merge_section("futures_options", raw, self._cfg.fno)
        self._merge_section("commodity", raw, self._cfg.commodity)
        self._merge_section("mcx", raw, self._cfg.commodity)
        self._merge_section("scanner", raw, self._cfg.scanner)
        self._merge_section("learning", raw, self._cfg.learning)
        self._merge_section("dashboard", raw, self._cfg.dashboard)

        logger.info("Loaded angel_one_config.json")

    # ── trading_{env}.json ──

    def _load_trading_env_config(self) -> None:
        env = self._cfg.environment or os.getenv("TRADING_BOT_ENVIRONMENT", "development")
        path = self._config_dir / f"trading_{env}.json"
        if not path.exists():
            logger.debug("No environment config: %s", path)
            return

        try:
            raw = self._read_json(path)
        except Exception:
            logger.exception("Cannot parse %s", path)
            return

        # Top-level scalars
        for key in (
            "trading_enabled", "paper_trading", "total_capital",
            "intraday_capital_pct", "reserved_capital_pct", "environment", "log_level",
        ):
            if key in raw:
                current = getattr(self._cfg, key)
                setattr(self._cfg, key, type(current)(raw[key]))

        self._merge_section("risk_limits", raw, self._cfg.risk_limits)
        self._merge_section("position_sizing", raw, self._cfg.position_sizing)
        self._merge_section("stop_loss", raw, self._cfg.stop_loss)
        self._merge_section("fno", raw, self._cfg.fno)
        self._merge_section("commodity", raw, self._cfg.commodity)
        self._merge_section("scanner", raw, self._cfg.scanner)
        self._merge_section("learning", raw, self._cfg.learning)
        self._merge_section("dashboard", raw, self._cfg.dashboard)

        logger.info("Loaded trading_%s.json", env)

    # ── strategies.json ──

    def _load_strategy_configs(self) -> None:
        path = self._config_dir / "strategies.json"
        if not path.exists():
            return

        try:
            raw = self._read_json(path)
        except Exception:
            logger.exception("Cannot parse %s", path)
            return

        loaded: List[StrategyConfig] = []
        for entry in raw.get("strategies", []):
            # normalise allocation key
            if "allocation" in entry and "allocation_pct" not in entry:
                entry["allocation_pct"] = float(entry.pop("allocation"))
            try:
                cfg = _safe_construct(StrategyConfig, entry)
                errs = cfg.validate()
                if errs:
                    logger.warning("Strategy '%s' issues: %s", cfg.strategy_id, errs)
                loaded.append(cfg)
            except Exception as exc:
                logger.error("Bad strategy entry %s: %s", entry.get("strategy_id", "?"), exc)

        self._cfg.strategy_configs = loaded
        logger.info("Loaded %d strategy configs", len(loaded))

    # ── environment variable overrides ──

    _ENV_MAP: Dict[str, Tuple[str, str]] = {
        "ANGEL_API_KEY":       ("angel_one", "api_key"),
        "ANGEL_CLIENT_ID":     ("angel_one", "client_id"),
        "ANGEL_PASSWORD":      ("angel_one", "password"),
        "ANGEL_TOTP_SECRET":   ("angel_one", "totp_secret"),
        "TRADING_BOT_CAPITAL": ("",          "total_capital"),
        "TRADING_BOT_PAPER":   ("",          "paper_trading"),
        "TRADING_BOT_ENV":     ("",          "environment"),
        "TELEGRAM_BOT_TOKEN":  ("dashboard", "telegram_bot_token"),
        "TELEGRAM_CHAT_ID":    ("dashboard", "telegram_chat_id"),
    }

    def _apply_env_overrides(self) -> None:
        for env_var, (section, attr) in self._ENV_MAP.items():
            value = os.environ.get(env_var)
            if value is None:
                continue

            target = getattr(self._cfg, section) if section else self._cfg
            current = getattr(target, attr, None)
            if current is None:
                continue

            try:
                if isinstance(current, bool):
                    converted = value.lower() in ("true", "1", "yes")
                else:
                    converted = type(current)(value)
                setattr(target, attr, converted)

                is_secret = any(s in attr for s in ("key", "secret", "password", "token"))
                logger.debug(
                    "Env override %s → %s",
                    env_var,
                    "***" if is_secret else converted,
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Env override %s failed: %s", env_var, exc)

    # ────────────────────────────── saving ──

    def save_trading_config(self) -> bool:
        with self._lock:
            try:
                env = self._cfg.environment
                path = self._config_dir / f"trading_{env}.json"
                data = _dataclass_to_dict(self._cfg)
                # Strip credentials from env-specific file
                data.pop("angel_one", None)
                data.pop("strategy_configs", None)
                self._write_json_atomic(path, data)
                logger.info("Saved trading config → %s", path)
                return True
            except Exception:
                logger.exception("Failed to save trading config")
                return False

    def save_strategy_configs(self) -> bool:
        with self._lock:
            try:
                path = self._config_dir / "strategies.json"
                data = {
                    "strategies": [
                        _dataclass_to_dict(s)
                        for s in self._cfg.strategy_configs
                    ]
                }
                self._write_json_atomic(path, data)
                logger.info("Saved %d strategies → %s", len(self._cfg.strategy_configs), path)
                return True
            except Exception:
                logger.exception("Failed to save strategy configs")
                return False

    # ────────────────────────────── runtime updates ──

    def update(self, section: str, **kwargs) -> bool:
        """
        Update a sub-config at runtime.
        Validates after mutation; rolls back on failure.

        Usage:
            config_manager.update("risk_limits", max_positions=3, max_daily_loss_pct=0.04)
            config_manager.update("stop_loss", atr_multiplier=2.0)
            config_manager.update("trading", paper_trading=False)
        """
        with self._lock:
            target = self._cfg if section == "trading" else getattr(self._cfg, section, None)
            if target is None:
                logger.error("Unknown config section: %s", section)
                return False

            snapshot: Dict[str, Any] = {}
            for key, value in kwargs.items():
                if hasattr(target, key):
                    snapshot[key] = getattr(target, key)
                    setattr(target, key, value)
                else:
                    logger.warning("Ignored unknown key %s.%s", section, key)

            errors = self._cfg.validate()
            if errors:
                for key, old_val in snapshot.items():
                    setattr(target, key, old_val)
                logger.error("Update rolled back (%s): %s", section, errors)
                return False

            self.save_trading_config()
            self._notify("config_updated", {"section": section, "changes": kwargs})
            return True

    def update_strategy(self, strategy_id: str, **kwargs) -> bool:
        with self._lock:
            for strat in self._cfg.strategy_configs:
                if strat.strategy_id == strategy_id:
                    snapshot = {k: getattr(strat, k) for k in kwargs if hasattr(strat, k)}
                    for k, v in kwargs.items():
                        if hasattr(strat, k):
                            setattr(strat, k, v)

                    errs = strat.validate()
                    if errs:
                        for k, old in snapshot.items():
                            setattr(strat, k, old)
                        logger.error("Strategy update rolled back: %s", errs)
                        return False

                    self.save_strategy_configs()
                    self._notify("strategy_updated", {"id": strategy_id, **kwargs})
                    return True

            logger.error("Strategy not found: %s", strategy_id)
            return False

    def add_strategy(self, cfg: StrategyConfig) -> bool:
        with self._lock:
            errs = cfg.validate()
            if errs:
                logger.error("Strategy validation failed: %s", errs)
                return False
            ids = {s.strategy_id for s in self._cfg.strategy_configs}
            if cfg.strategy_id in ids:
                logger.error("Duplicate strategy_id: %s", cfg.strategy_id)
                return False
            self._cfg.strategy_configs.append(cfg)
            self.save_strategy_configs()
            self._notify("strategy_added", {"id": cfg.strategy_id})
            return True

    def remove_strategy(self, strategy_id: str) -> bool:
        with self._lock:
            before = len(self._cfg.strategy_configs)
            self._cfg.strategy_configs = [
                s for s in self._cfg.strategy_configs if s.strategy_id != strategy_id
            ]
            if len(self._cfg.strategy_configs) < before:
                self.save_strategy_configs()
                self._notify("strategy_removed", {"id": strategy_id})
                return True
            logger.warning("Strategy not found for removal: %s", strategy_id)
            return False

    # ────────────────────────────── reload / backup / restore ──

    def reload(self) -> bool:
        with self._lock:
            try:
                old_hash = self._file_hash
                self._cfg = TradingConfig()
                self._load_all()
                changed = self._file_hash != old_hash
                if changed:
                    self._notify("config_reloaded", None)
                logger.info("Reload complete (changed=%s)", changed)
                return True
            except Exception:
                logger.exception("Reload failed")
                return False

    def has_files_changed(self) -> bool:
        return self._compute_files_hash() != self._file_hash

    def backup(self, label: str = "") -> Optional[Path]:
        try:
            tag = label or datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self._data_dir / "backups" / f"config_{tag}"
            dest.mkdir(parents=True, exist_ok=True)
            for f in self._config_dir.glob("*.json"):
                shutil.copy2(f, dest / f.name)
            logger.info("Backup created → %s", dest)
            return dest
        except Exception:
            logger.exception("Backup failed")
            return None

    def restore(self, backup_path: Union[str, Path]) -> bool:
        backup_dir = Path(backup_path)
        if not backup_dir.is_dir():
            logger.error("Backup directory not found: %s", backup_dir)
            return False
        try:
            self.backup(label="pre_restore")
            for f in backup_dir.glob("*.json"):
                shutil.copy2(f, self._config_dir / f.name)
            self.reload()
            logger.info("Restored from %s", backup_dir)
            return True
        except Exception:
            logger.exception("Restore failed")
            return False

    # ────────────────────────────── watchers ──

    def add_watcher(self, callback: Callable[[str, Any], None]) -> None:
        self._watchers.append(callback)

    def remove_watcher(self, callback: Callable[[str, Any], None]) -> None:
        self._watchers = [w for w in self._watchers if w is not callback]

    def _notify(self, event: str, data: Any) -> None:
        for cb in self._watchers:
            try:
                cb(event, data)
            except Exception:
                logger.exception("Config watcher error")

    # ────────────────────────────── convenience queries ──

    def market_hours(self, segment: str = Segment.NSE.value) -> MarketHours:
        return SEGMENT_HOURS.get(segment, NSE_HOURS)

    def costs(self, cost_key: str = "EQUITY_INTRADAY") -> TransactionCosts:
        return SEGMENT_COSTS.get(cost_key, EQUITY_INTRADAY_COSTS)

    def is_entry_allowed(self, segment: str = Segment.NSE.value) -> bool:
        if not self._cfg.trading_enabled:
            return False
        return self.market_hours(segment).is_entry_allowed(datetime.now().time())

    def should_force_exit(self, segment: str = Segment.NSE.value) -> bool:
        return self.market_hours(segment).should_force_exit(datetime.now().time())

    def minutes_to_close(self, segment: str = Segment.NSE.value) -> int:
        return self.market_hours(segment).minutes_to_close(datetime.now().time())

    def breakeven_move_pct(
        self,
        position_value: float,
        cost_key: str = "EQUITY_INTRADAY",
    ) -> float:
        return self.costs(cost_key).breakeven_move_pct(position_value)

    def regime_multiplier(self, regime: str) -> float:
        return self._cfg.risk_limits.regime_multiplier(regime)

    def validate(self) -> List[str]:
        with self._lock:
            return self._cfg.validate()

    def to_dict(self) -> dict:
        with self._lock:
            return _dataclass_to_dict(self._cfg)

    # ────────────────────────────── internal helpers ──

    @staticmethod
    def _read_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _write_json_atomic(path: Path, data: dict) -> None:
        """Write via temp-file + rename to avoid partial writes."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
        tmp.replace(path)

    @staticmethod
    def _merge_section(key: str, source: dict, target: Any) -> None:
        """Copy matching keys from source[key] dict into a dataclass *target*."""
        section = source.get(key)
        if not isinstance(section, dict):
            return
        for k, v in section.items():
            if not hasattr(target, k):
                continue
            current = getattr(target, k)
            if isinstance(current, dict) and isinstance(v, dict):
                current.update(v)
            elif isinstance(current, list) and isinstance(v, list):
                setattr(target, k, v)
            else:
                try:
                    setattr(target, k, type(current)(v))
                except (ValueError, TypeError):
                    setattr(target, k, v)

    @staticmethod
    def _map_flat(
        target: Any,
        source: dict,
        mapping: Dict[str, Tuple[str, Any]],
    ) -> None:
        """Apply a {src_key → (dest_attr, converter)} mapping."""
        for src_key, (dest_attr, converter) in mapping.items():
            if src_key in source:
                try:
                    setattr(target, dest_attr, converter(source[src_key]))
                except (ValueError, TypeError) as exc:
                    logger.warning("Mapping %s → %s failed: %s", src_key, dest_attr, exc)

    def _compute_files_hash(self) -> str:
        h = hashlib.md5()
        for p in sorted(self._config_dir.glob("*.json")):
            try:
                h.update(p.read_bytes())
            except OSError:
                pass
        return h.hexdigest()

    def __repr__(self) -> str:
        c = self._cfg
        return (
            f"ConfigManager(env={c.environment}, capital=₹{c.total_capital:,.0f}, "
            f"paper={c.paper_trading}, strategies={len(c.strategy_configs)}, "
            f"fno={c.fno.enabled}, mcx={c.commodity.enabled})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Module-level singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config_manager: ConfigManager = ConfigManager()