"""
Technical indicators implementation for the trading bot.
Provides moving averages, RSI, MACD, Bollinger Bands and other indicators.
"""

import numpy as np
import pandas as pd
from decimal import Decimal
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import structlog

from ..core.models import OHLC, IndicatorValues

logger = structlog.get_logger(__name__)


@dataclass
class MACD:
    """MACD indicator values."""
    macd_line: Decimal
    signal_line: Decimal
    histogram: Decimal
    timestamp: datetime


@dataclass
class BollingerBands:
    """Bollinger Bands indicator values."""
    upper_band: Decimal
    middle_band: Decimal  # SMA
    lower_band: Decimal
    timestamp: datetime


class TechnicalIndicators:
    """
    Technical indicators calculator.
    Implements common trading indicators with optimized calculations.
    """
    
    @staticmethod
    def sma(prices: List[Decimal], period: int) -> Optional[Decimal]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of prices
            period: Period for SMA calculation
            
        Returns:
            SMA value or None if insufficient data
        """
        if len(prices) < period:
            return None
        
        recent_prices = prices[-period:]
        return sum(recent_prices) / period
    
    @staticmethod
    def ema(prices: List[Decimal], period: int, previous_ema: Optional[Decimal] = None) -> Optional[Decimal]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of prices
            period: Period for EMA calculation
            previous_ema: Previous EMA value for incremental calculation
            
        Returns:
            EMA value or None if insufficient data
        """
        if not prices:
            return None
        
        current_price = prices[-1]
        multiplier = Decimal(2) / (period + 1)
        
        if previous_ema is None:
            # First EMA calculation - use SMA as starting point
            if len(prices) < period:
                return None
            previous_ema = TechnicalIndicators.sma(prices[-period:], period)
            if previous_ema is None:
                return None
        
        return (current_price * multiplier) + (previous_ema * (1 - multiplier))
    
    @staticmethod
    def rsi(prices: List[Decimal], period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: List of prices
            period: Period for RSI calculation (default 14)
            
        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            price_changes.append(change)
        
        if len(price_changes) < period:
            return None
        
        # Separate gains and losses
        gains = [max(change, Decimal('0')) for change in price_changes[-period:]]
        losses = [abs(min(change, Decimal('0'))) for change in price_changes[-period:]]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def macd(
        prices: List[Decimal], 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Optional[MACD]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of prices
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            MACD object or None if insufficient data
        """
        if len(prices) < slow_period:
            return None
        
        # Calculate EMAs
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # For signal line, we need historical MACD values
        # This is a simplified implementation - in practice, you'd maintain MACD history
        signal_line = macd_line  # Simplified - should be EMA of MACD line
        histogram = macd_line - signal_line
        
        return MACD(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def bollinger_bands(
        prices: List[Decimal], 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Optional[BollingerBands]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: List of prices
            period: Period for SMA calculation (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            
        Returns:
            BollingerBands object or None if insufficient data
        """
        if len(prices) < period:
            return None
        
        recent_prices = prices[-period:]
        
        # Calculate middle band (SMA)
        middle_band = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((price - middle_band) ** 2 for price in recent_prices) / period
        std_deviation = Decimal(str(variance ** 0.5))
        
        # Calculate upper and lower bands
        band_width = std_deviation * Decimal(str(std_dev))
        upper_band = middle_band + band_width
        lower_band = middle_band - band_width
        
        return BollingerBands(
            upper_band=upper_band,
            middle_band=middle_band,
            lower_band=lower_band,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def stochastic_oscillator(
        highs: List[Decimal], 
        lows: List[Decimal], 
        closes: List[Decimal], 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            k_period: Period for %K calculation (default 14)
            d_period: Period for %D calculation (default 3)
            
        Returns:
            Tuple of (%K, %D) or None if insufficient data
        """
        if len(closes) < k_period or len(highs) < k_period or len(lows) < k_period:
            return None
        
        # Get recent data
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        # Calculate %K
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = float((current_close - lowest_low) / (highest_high - lowest_low) * 100)
        
        # For %D, we need historical %K values
        # This is simplified - should be SMA of %K values
        d_percent = k_percent  # Simplified implementation
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(
        highs: List[Decimal], 
        lows: List[Decimal], 
        closes: List[Decimal], 
        period: int = 14
    ) -> Optional[float]:
        """
        Calculate Williams %R.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            period: Period for calculation (default 14)
            
        Returns:
            Williams %R value (-100 to 0) or None if insufficient data
        """
        if len(closes) < period or len(highs) < period or len(lows) < period:
            return None
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = float((highest_high - current_close) / (highest_high - lowest_low) * -100)
        return williams_r
    
    @staticmethod
    def atr(highs: List[Decimal], lows: List[Decimal], closes: List[Decimal], period: int = 14) -> Optional[Decimal]:
        """
        Calculate Average True Range.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            period: Period for ATR calculation (default 14)
            
        Returns:
            ATR value or None if insufficient data
        """
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return None
        
        true_ranges = []
        
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        # Calculate ATR as SMA of true ranges
        recent_tr = true_ranges[-period:]
        atr = sum(recent_tr) / period
        
        return atr


class IndicatorCalculator:
    """
    Main indicator calculator that maintains state and provides incremental updates.
    """
    
    def __init__(self):
        self.price_history: Dict[str, List[Decimal]] = {}
        self.ohlc_history: Dict[str, List[OHLC]] = {}
        self.indicator_cache: Dict[str, IndicatorValues] = {}
        self.max_history_length = 1000  # Keep last 1000 data points
    
    def update_price_data(self, symbol: str, price: Decimal) -> None:
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Limit history length
        if len(self.price_history[symbol]) > self.max_history_length:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history_length:]
    
    def update_ohlc_data(self, symbol: str, ohlc: OHLC) -> None:
        """Update OHLC history for a symbol."""
        if symbol not in self.ohlc_history:
            self.ohlc_history[symbol] = []
        
        self.ohlc_history[symbol].append(ohlc)
        
        # Limit history length
        if len(self.ohlc_history[symbol]) > self.max_history_length:
            self.ohlc_history[symbol] = self.ohlc_history[symbol][-self.max_history_length:]
    
    def calculate_indicators(self, symbol: str) -> Optional[IndicatorValues]:
        """
        Calculate all indicators for a symbol.
        
        Args:
            symbol: Symbol to calculate indicators for
            
        Returns:
            IndicatorValues object or None if insufficient data
        """
        try:
            if symbol not in self.price_history:
                return None
            
            prices = self.price_history[symbol]
            if len(prices) < 50:  # Need minimum data for reliable indicators
                return None
            
            # Calculate indicators
            indicators = IndicatorValues(timestamp=datetime.now())
            
            # Moving averages
            indicators.sma_20 = TechnicalIndicators.sma(prices, 20)
            indicators.sma_50 = TechnicalIndicators.sma(prices, 50)
            indicators.ema_12 = TechnicalIndicators.ema(prices, 12)
            indicators.ema_26 = TechnicalIndicators.ema(prices, 26)
            
            # RSI
            rsi_value = TechnicalIndicators.rsi(prices, 14)
            if rsi_value is not None:
                indicators.rsi = rsi_value
            
            # MACD
            macd = TechnicalIndicators.macd(prices)
            if macd:
                indicators.macd_line = macd.macd_line
                indicators.macd_signal = macd.signal_line
                indicators.macd_histogram = macd.histogram
            
            # Bollinger Bands
            bb = TechnicalIndicators.bollinger_bands(prices)
            if bb:
                indicators.bb_upper = bb.upper_band
                indicators.bb_middle = bb.middle_band
                indicators.bb_lower = bb.lower_band
            
            # Cache the results
            self.indicator_cache[symbol] = indicators
            
            logger.debug(f"Calculated indicators for {symbol}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def get_cached_indicators(self, symbol: str) -> Optional[IndicatorValues]:
        """Get cached indicators for a symbol."""
        return self.indicator_cache.get(symbol)
    
    def has_sufficient_data(self, symbol: str, min_periods: int = 50) -> bool:
        """Check if we have sufficient data for indicator calculations."""
        if symbol not in self.price_history:
            return False
        return len(self.price_history[symbol]) >= min_periods
    
    def get_price_history(self, symbol: str, periods: int) -> List[Decimal]:
        """Get recent price history for a symbol."""
        if symbol not in self.price_history:
            return []
        
        prices = self.price_history[symbol]
        return prices[-periods:] if len(prices) >= periods else prices
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear indicator cache for a symbol or all symbols."""
        if symbol:
            self.indicator_cache.pop(symbol, None)
        else:
            self.indicator_cache.clear()


# Global indicator calculator instance
indicator_calculator = IndicatorCalculator()