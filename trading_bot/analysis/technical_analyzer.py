"""
Technical analyzer component for the trading bot.
Processes market data and generates trading signals based on technical indicators.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import structlog

from ..core.events import EventHandler, EventBus, MarketDataEvent, TradingSignalEvent, EventType
from ..core.models import TradingSignal, SignalType, IndicatorValues, MarketTick
from .indicators import indicator_calculator, TechnicalIndicators

logger = structlog.get_logger(__name__)


@dataclass
class SignalRule:
    """Configuration for a signal generation rule."""
    rule_id: str
    name: str
    enabled: bool = True
    min_strength: float = 0.5
    indicators_required: List[str] = None
    
    def __post_init__(self):
        if self.indicators_required is None:
            self.indicators_required = []


class TechnicalAnalyzer(EventHandler):
    """
    Technical analyzer that processes market data and generates trading signals.
    Implements various technical analysis strategies and signal generation rules.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.signal_rules: Dict[str, SignalRule] = {}
        self.active_symbols: Set[str] = set()
        self.signal_history: Dict[str, List[TradingSignal]] = {}
        self.processing_enabled = True
        
        # Default signal rules
        self._setup_default_rules()
        
        # Subscribe to market data events
        self.event_bus.subscribe(EventType.MARKET_DATA, self)
    
    def _setup_default_rules(self) -> None:
        """Setup default signal generation rules."""
        
        # Moving Average Crossover
        self.signal_rules["ma_crossover"] = SignalRule(
            rule_id="ma_crossover",
            name="Moving Average Crossover",
            enabled=True,
            min_strength=0.6,
            indicators_required=["sma_20", "sma_50"]
        )
        
        # RSI Overbought/Oversold
        self.signal_rules["rsi_levels"] = SignalRule(
            rule_id="rsi_levels",
            name="RSI Overbought/Oversold",
            enabled=True,
            min_strength=0.7,
            indicators_required=["rsi"]
        )
        
        # MACD Signal
        self.signal_rules["macd_signal"] = SignalRule(
            rule_id="macd_signal",
            name="MACD Signal Line Crossover",
            enabled=True,
            min_strength=0.6,
            indicators_required=["macd_line", "macd_signal"]
        )
        
        # Bollinger Bands
        self.signal_rules["bollinger_bands"] = SignalRule(
            rule_id="bollinger_bands",
            name="Bollinger Bands Breakout",
            enabled=True,
            min_strength=0.5,
            indicators_required=["bb_upper", "bb_lower"]
        )
    
    async def handle_event(self, event) -> None:
        """Handle incoming market data events."""
        if not self.processing_enabled:
            return
        
        if isinstance(event, MarketDataEvent):
            await self._process_market_data(event)
    
    async def _process_market_data(self, event: MarketDataEvent) -> None:
        """Process market data and generate signals if conditions are met."""
        try:
            symbol = event.symbol
            price = event.price
            
            # Update price data in indicator calculator
            indicator_calculator.update_price_data(symbol, price)
            
            # Add to active symbols
            self.active_symbols.add(symbol)
            
            # Check if we have sufficient data for analysis
            if not indicator_calculator.has_sufficient_data(symbol):
                logger.debug(f"Insufficient data for analysis: {symbol}")
                return
            
            # Calculate indicators
            indicators = indicator_calculator.calculate_indicators(symbol)
            if not indicators or not indicators.has_sufficient_data():
                logger.debug(f"No indicators available for {symbol}")
                return
            
            # Generate signals based on rules
            signals = await self._generate_signals(symbol, price, indicators)
            
            # Publish signals
            for signal in signals:
                await self._publish_signal(signal)
            
        except Exception as e:
            logger.error(f"Error processing market data for {event.symbol}: {e}")
    
    async def _generate_signals(
        self, 
        symbol: str, 
        current_price: Decimal, 
        indicators: IndicatorValues
    ) -> List[TradingSignal]:
        """Generate trading signals based on technical indicators."""
        signals = []
        
        try:
            # Moving Average Crossover Signal
            if self.signal_rules["ma_crossover"].enabled:
                ma_signal = self._check_ma_crossover(symbol, indicators)
                if ma_signal:
                    signals.append(ma_signal)
            
            # RSI Overbought/Oversold Signal
            if self.signal_rules["rsi_levels"].enabled:
                rsi_signal = self._check_rsi_levels(symbol, indicators)
                if rsi_signal:
                    signals.append(rsi_signal)
            
            # MACD Signal
            if self.signal_rules["macd_signal"].enabled:
                macd_signal = self._check_macd_signal(symbol, indicators)
                if macd_signal:
                    signals.append(macd_signal)
            
            # Bollinger Bands Signal
            if self.signal_rules["bollinger_bands"].enabled:
                bb_signal = self._check_bollinger_bands(symbol, current_price, indicators)
                if bb_signal:
                    signals.append(bb_signal)
            
            # Filter signals by minimum strength
            filtered_signals = [
                signal for signal in signals 
                if signal.strength >= self.signal_rules.get(
                    signal.indicators_used[0] if signal.indicators_used else "default", 
                    SignalRule("default", "Default")
                ).min_strength
            ]
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _check_ma_crossover(self, symbol: str, indicators: IndicatorValues) -> Optional[TradingSignal]:
        """Check for moving average crossover signals."""
        if not indicators.sma_20 or not indicators.sma_50:
            return None
        
        sma_20 = indicators.sma_20
        sma_50 = indicators.sma_50
        
        # Get previous indicators to detect crossover
        previous_indicators = self._get_previous_indicators(symbol)
        if not previous_indicators or not previous_indicators.sma_20 or not previous_indicators.sma_50:
            return None
        
        prev_sma_20 = previous_indicators.sma_20
        prev_sma_50 = previous_indicators.sma_50
        
        # Bullish crossover: SMA20 crosses above SMA50
        if prev_sma_20 <= prev_sma_50 and sma_20 > sma_50:
            strength = min(float((sma_20 - sma_50) / sma_50 * 100), 1.0)
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                indicators_used=["sma_20", "sma_50"],
                confidence=0.7
            )
        
        # Bearish crossover: SMA20 crosses below SMA50
        elif prev_sma_20 >= prev_sma_50 and sma_20 < sma_50:
            strength = min(float((sma_50 - sma_20) / sma_50 * 100), 1.0)
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                indicators_used=["sma_20", "sma_50"],
                confidence=0.7
            )
        
        return None
    
    def _check_rsi_levels(self, symbol: str, indicators: IndicatorValues) -> Optional[TradingSignal]:
        """Check for RSI overbought/oversold signals."""
        if indicators.rsi is None:
            return None
        
        rsi = indicators.rsi
        
        # Oversold condition (RSI < 30)
        if rsi < 30:
            strength = (30 - rsi) / 30  # Stronger signal as RSI gets lower
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                indicators_used=["rsi"],
                confidence=0.8
            )
        
        # Overbought condition (RSI > 70)
        elif rsi > 70:
            strength = (rsi - 70) / 30  # Stronger signal as RSI gets higher
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                indicators_used=["rsi"],
                confidence=0.8
            )
        
        return None
    
    def _check_macd_signal(self, symbol: str, indicators: IndicatorValues) -> Optional[TradingSignal]:
        """Check for MACD signal line crossover."""
        if not indicators.macd_line or not indicators.macd_signal:
            return None
        
        macd_line = indicators.macd_line
        signal_line = indicators.macd_signal
        
        # Get previous indicators to detect crossover
        previous_indicators = self._get_previous_indicators(symbol)
        if (not previous_indicators or 
            not previous_indicators.macd_line or 
            not previous_indicators.macd_signal):
            return None
        
        prev_macd = previous_indicators.macd_line
        prev_signal = previous_indicators.macd_signal
        
        # Bullish crossover: MACD crosses above signal line
        if prev_macd <= prev_signal and macd_line > signal_line:
            strength = min(float(abs(macd_line - signal_line) / signal_line * 10), 1.0)
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                indicators_used=["macd_line", "macd_signal"],
                confidence=0.75
            )
        
        # Bearish crossover: MACD crosses below signal line
        elif prev_macd >= prev_signal and macd_line < signal_line:
            strength = min(float(abs(signal_line - macd_line) / signal_line * 10), 1.0)
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                indicators_used=["macd_line", "macd_signal"],
                confidence=0.75
            )
        
        return None
    
    def _check_bollinger_bands(
        self, 
        symbol: str, 
        current_price: Decimal, 
        indicators: IndicatorValues
    ) -> Optional[TradingSignal]:
        """Check for Bollinger Bands breakout signals."""
        if not indicators.bb_upper or not indicators.bb_lower or not indicators.bb_middle:
            return None
        
        upper_band = indicators.bb_upper
        lower_band = indicators.bb_lower
        middle_band = indicators.bb_middle
        
        # Price breaks above upper band (potential sell signal - overbought)
        if current_price > upper_band:
            strength = min(float((current_price - upper_band) / middle_band * 10), 1.0)
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                indicators_used=["bb_upper", "bb_lower"],
                confidence=0.6
            )
        
        # Price breaks below lower band (potential buy signal - oversold)
        elif current_price < lower_band:
            strength = min(float((lower_band - current_price) / middle_band * 10), 1.0)
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                indicators_used=["bb_upper", "bb_lower"],
                confidence=0.6
            )
        
        return None
    
    def _get_previous_indicators(self, symbol: str) -> Optional[IndicatorValues]:
        """Get previous indicators for crossover detection."""
        # This is a simplified implementation
        # In practice, you'd maintain a history of indicator values
        return indicator_calculator.get_cached_indicators(symbol)
    
    async def _publish_signal(self, signal: TradingSignal) -> None:
        """Publish a trading signal to the event bus."""
        try:
            # Store signal in history
            if signal.symbol not in self.signal_history:
                self.signal_history[signal.symbol] = []
            
            self.signal_history[signal.symbol].append(signal)
            
            # Keep only last 100 signals per symbol
            if len(self.signal_history[signal.symbol]) > 100:
                self.signal_history[signal.symbol] = self.signal_history[signal.symbol][-100:]
            
            # Create signal event
            signal_event = TradingSignalEvent(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                strength=signal.strength,
                indicators_used=signal.indicators_used,
                price_target=signal.price_target,
                source_component="TechnicalAnalyzer"
            )
            
            # Publish to event bus
            await self.event_bus.publish(signal_event)
            
            logger.info(
                f"Generated {signal.signal_type.value} signal for {signal.symbol} "
                f"(strength: {signal.strength:.2f}, confidence: {signal.confidence:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Error publishing signal: {e}")
    
    def add_signal_rule(self, rule: SignalRule) -> None:
        """Add a custom signal generation rule."""
        self.signal_rules[rule.rule_id] = rule
        logger.info(f"Added signal rule: {rule.name}")
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a signal generation rule."""
        if rule_id in self.signal_rules:
            self.signal_rules[rule_id].enabled = True
            logger.info(f"Enabled signal rule: {rule_id}")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a signal generation rule."""
        if rule_id in self.signal_rules:
            self.signal_rules[rule_id].enabled = False
            logger.info(f"Disabled signal rule: {rule_id}")
            return True
        return False
    
    def get_signal_history(self, symbol: str, limit: int = 10) -> List[TradingSignal]:
        """Get recent signal history for a symbol."""
        if symbol not in self.signal_history:
            return []
        
        return self.signal_history[symbol][-limit:]
    
    def get_active_symbols(self) -> Set[str]:
        """Get set of symbols being analyzed."""
        return self.active_symbols.copy()
    
    def start_processing(self) -> None:
        """Enable signal processing."""
        self.processing_enabled = True
        logger.info("Technical analyzer processing enabled")
    
    def stop_processing(self) -> None:
        """Disable signal processing."""
        self.processing_enabled = False
        logger.info("Technical analyzer processing disabled")
    
    def get_current_indicators(self, symbol: str) -> Optional[IndicatorValues]:
        """Get current indicators for a symbol."""
        return indicator_calculator.get_cached_indicators(symbol)


# Create global technical analyzer instance (will be initialized with event_bus in main)
technical_analyzer = None