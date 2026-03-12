"""
Advanced feature engineering for ML trading models.
Creates comprehensive market features from raw data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import structlog
from dataclasses import dataclass

from ..core.models import MarketTick, OHLC, IndicatorValues
from ..analysis.indicators import TechnicalIndicators

logger = structlog.get_logger(__name__)


@dataclass
class MarketFeatures:
    """Comprehensive market features for ML models."""
    # Price features
    price_features: Dict[str, float]
    
    # Technical indicators
    technical_features: Dict[str, float]
    
    # Volume features
    volume_features: Dict[str, float]
    
    # Volatility features
    volatility_features: Dict[str, float]
    
    # Market microstructure
    microstructure_features: Dict[str, float]
    
    # Time-based features
    temporal_features: Dict[str, float]
    
    # Cross-asset features
    cross_asset_features: Dict[str, float]
    
    # Market regime features
    regime_features: Dict[str, float]
    
    # Sentiment features
    sentiment_features: Dict[str, float]
    
    timestamp: datetime


class FeatureEngineer:
    """
    Advanced feature engineering for trading ML models.
    Creates rich feature sets from market data.
    """
    
    def __init__(self):
        self.feature_cache: Dict[str, MarketFeatures] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.ohlc_history: Dict[str, List[OHLC]] = {}
        self.tick_history: Dict[str, List[MarketTick]] = {}
        
        # Feature configuration
        self.lookback_periods = [5, 10, 20, 50, 100, 200]
        self.volatility_windows = [10, 20, 50]
        self.max_history_length = 5000
        
        logger.info("Feature engineer initialized")
    
    def update_market_data(self, symbol: str, tick: MarketTick) -> None:
        """Update market data for feature calculation."""
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(float(tick.price))
        
        # Update volume history
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        self.volume_history[symbol].append(float(tick.volume))
        
        # Update tick history
        if symbol not in self.tick_history:
            self.tick_history[symbol] = []
        self.tick_history[symbol].append(tick)
        
        # Limit history length
        for history in [self.price_history[symbol], self.volume_history[symbol], self.tick_history[symbol]]:
            if len(history) > self.max_history_length:
                history[:] = history[-self.max_history_length:]
    
    def update_ohlc_data(self, symbol: str, ohlc: OHLC) -> None:
        """Update OHLC data for feature calculation."""
        if symbol not in self.ohlc_history:
            self.ohlc_history[symbol] = []
        
        self.ohlc_history[symbol].append(ohlc)
        
        # Limit history length
        if len(self.ohlc_history[symbol]) > self.max_history_length:
            self.ohlc_history[symbol] = self.ohlc_history[symbol][-self.max_history_length:]
    
    def extract_features(self, symbol: str) -> Optional[MarketFeatures]:
        """Extract comprehensive features for a symbol."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
                return None
            
            features = MarketFeatures(
                price_features=self._extract_price_features(symbol),
                technical_features=self._extract_technical_features(symbol),
                volume_features=self._extract_volume_features(symbol),
                volatility_features=self._extract_volatility_features(symbol),
                microstructure_features=self._extract_microstructure_features(symbol),
                temporal_features=self._extract_temporal_features(symbol),
                cross_asset_features=self._extract_cross_asset_features(symbol),
                regime_features=self._extract_regime_features(symbol),
                sentiment_features=self._extract_sentiment_features(symbol),
                timestamp=datetime.now()
            )
            
            # Cache features
            self.feature_cache[symbol] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return None
    
    def _extract_price_features(self, symbol: str) -> Dict[str, float]:
        """Extract price-based features."""
        prices = self.price_history[symbol]
        features = {}
        
        try:
            current_price = prices[-1]
            
            # Returns over different periods
            for period in self.lookback_periods:
                if len(prices) > period:
                    past_price = prices[-period-1]
                    returns = (current_price - past_price) / past_price
                    features[f'return_{period}d'] = returns
                    
                    # Log returns
                    features[f'log_return_{period}d'] = np.log(current_price / past_price)
            
            # Price momentum
            for period in [5, 10, 20]:
                if len(prices) > period:
                    momentum = sum(prices[-period:]) / period / current_price - 1
                    features[f'momentum_{period}d'] = momentum
            
            # Price acceleration
            if len(prices) > 20:
                recent_momentum = sum(prices[-10:]) / 10
                past_momentum = sum(prices[-20:-10]) / 10
                features['price_acceleration'] = (recent_momentum - past_momentum) / past_momentum
            
            # Price relative to moving averages
            for period in [20, 50, 100]:
                if len(prices) >= period:
                    ma = sum(prices[-period:]) / period
                    features[f'price_vs_ma_{period}'] = (current_price - ma) / ma
            
            # Price percentile over different periods
            for period in [50, 100, 200]:
                if len(prices) >= period:
                    recent_prices = prices[-period:]
                    percentile = (sum(1 for p in recent_prices if p < current_price) / len(recent_prices))
                    features[f'price_percentile_{period}d'] = percentile
            
        except Exception as e:
            logger.error(f"Error extracting price features: {e}")
        
        return features
    
    def _extract_technical_features(self, symbol: str) -> Dict[str, float]:
        """Extract technical indicator features."""
        features = {}
        
        try:
            prices = [Decimal(str(p)) for p in self.price_history[symbol]]
            
            if len(prices) < 50:
                return features
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(prices) >= period:
                    sma = TechnicalIndicators.sma(prices, period)
                    ema = TechnicalIndicators.ema(prices, period)
                    if sma:
                        features[f'sma_{period}'] = float(sma)
                    if ema:
                        features[f'ema_{period}'] = float(ema)
            
            # RSI
            rsi = TechnicalIndicators.rsi(prices, 14)
            if rsi:
                features['rsi'] = rsi
                features['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
                features['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            
            # MACD
            macd = TechnicalIndicators.macd(prices)
            if macd:
                features['macd_line'] = float(macd.macd_line)
                features['macd_signal'] = float(macd.signal_line)
                features['macd_histogram'] = float(macd.histogram)
                features['macd_bullish'] = 1.0 if macd.macd_line > macd.signal_line else 0.0
            
            # Bollinger Bands
            bb = TechnicalIndicators.bollinger_bands(prices)
            if bb:
                current_price = float(prices[-1])
                bb_position = (current_price - float(bb.lower_band)) / (float(bb.upper_band) - float(bb.lower_band))
                features['bb_position'] = bb_position
                features['bb_squeeze'] = (float(bb.upper_band) - float(bb.lower_band)) / float(bb.middle_band)
            
            # Additional technical indicators
            if len(self.ohlc_history.get(symbol, [])) > 20:
                ohlc_data = self.ohlc_history[symbol]
                highs = [float(ohlc.high_price) for ohlc in ohlc_data]
                lows = [float(ohlc.low_price) for ohlc in ohlc_data]
                closes = [float(ohlc.close_price) for ohlc in ohlc_data]
                
                # Stochastic
                stoch = TechnicalIndicators.stochastic_oscillator(
                    [Decimal(str(h)) for h in highs],
                    [Decimal(str(l)) for l in lows],
                    [Decimal(str(c)) for c in closes]
                )
                if stoch:
                    features['stoch_k'] = stoch[0]
                    features['stoch_d'] = stoch[1]
                
                # Williams %R
                williams_r = TechnicalIndicators.williams_r(
                    [Decimal(str(h)) for h in highs],
                    [Decimal(str(l)) for l in lows],
                    [Decimal(str(c)) for c in closes]
                )
                if williams_r:
                    features['williams_r'] = williams_r
        
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
        
        return features
    
    def _extract_volume_features(self, symbol: str) -> Dict[str, float]:
        """Extract volume-based features."""
        features = {}
        
        try:
            if symbol not in self.volume_history:
                return features
            
            volumes = self.volume_history[symbol]
            prices = self.price_history[symbol]
            
            if len(volumes) < 20:
                return features
            
            current_volume = volumes[-1]
            
            # Volume moving averages
            for period in [5, 10, 20]:
                if len(volumes) >= period:
                    vol_ma = sum(volumes[-period:]) / period
                    features[f'volume_ma_{period}'] = vol_ma
                    features[f'volume_ratio_{period}'] = current_volume / vol_ma if vol_ma > 0 else 0
            
            # Volume trend
            if len(volumes) >= 10:
                recent_vol = sum(volumes[-5:]) / 5
                past_vol = sum(volumes[-10:-5]) / 5
                features['volume_trend'] = (recent_vol - past_vol) / past_vol if past_vol > 0 else 0
            
            # Price-volume relationship
            if len(prices) == len(volumes) and len(prices) >= 20:
                # On-balance volume
                obv = 0
                for i in range(1, min(len(prices), 50)):
                    if prices[-i] > prices[-i-1]:
                        obv += volumes[-i]
                    elif prices[-i] < prices[-i-1]:
                        obv -= volumes[-i]
                features['obv'] = obv
                
                # Volume-weighted average price (simplified)
                if len(prices) >= 20:
                    vwap_num = sum(prices[-20:][i] * volumes[-20:][i] for i in range(20))
                    vwap_den = sum(volumes[-20:])
                    if vwap_den > 0:
                        vwap = vwap_num / vwap_den
                        features['vwap'] = vwap
                        features['price_vs_vwap'] = (prices[-1] - vwap) / vwap
        
        except Exception as e:
            logger.error(f"Error extracting volume features: {e}")
        
        return features
    
    def _extract_volatility_features(self, symbol: str) -> Dict[str, float]:
        """Extract volatility-based features."""
        features = {}
        
        try:
            prices = self.price_history[symbol]
            
            if len(prices) < 20:
                return features
            
            # Calculate returns
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            # Volatility over different windows
            for window in self.volatility_windows:
                if len(returns) >= window:
                    recent_returns = returns[-window:]
                    volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
                    features[f'volatility_{window}d'] = volatility
            
            # Volatility of volatility
            if len(returns) >= 50:
                vol_series = []
                for i in range(20, len(returns)):
                    window_returns = returns[i-20:i]
                    vol = np.std(window_returns)
                    vol_series.append(vol)
                
                if len(vol_series) > 1:
                    vol_of_vol = np.std(vol_series)
                    features['volatility_of_volatility'] = vol_of_vol
            
            # Realized volatility vs implied volatility (if available)
            # This would require options data - placeholder for now
            features['vol_regime'] = 1.0 if features.get('volatility_20d', 0) > 0.3 else 0.0
            
            # Volatility clustering
            if len(returns) >= 50:
                recent_vol = np.std(returns[-10:])
                past_vol = np.std(returns[-50:-10])
                features['vol_clustering'] = recent_vol / past_vol if past_vol > 0 else 1.0
        
        except Exception as e:
            logger.error(f"Error extracting volatility features: {e}")
        
        return features
    
    def _extract_microstructure_features(self, symbol: str) -> Dict[str, float]:
        """Extract market microstructure features."""
        features = {}
        
        try:
            if symbol not in self.tick_history:
                return features
            
            ticks = self.tick_history[symbol]
            
            if len(ticks) < 20:
                return features
            
            # Bid-ask spread
            recent_ticks = ticks[-20:]
            spreads = [(float(tick.ask_price) - float(tick.bid_price)) / float(tick.price) 
                      for tick in recent_ticks if tick.ask_price > 0 and tick.bid_price > 0]
            
            if spreads:
                features['avg_spread'] = np.mean(spreads)
                features['spread_volatility'] = np.std(spreads)
            
            # Order imbalance
            bid_sizes = [tick.bid_size for tick in recent_ticks]
            ask_sizes = [tick.ask_size for tick in recent_ticks]
            
            if bid_sizes and ask_sizes:
                total_bid = sum(bid_sizes)
                total_ask = sum(ask_sizes)
                if total_bid + total_ask > 0:
                    features['order_imbalance'] = (total_bid - total_ask) / (total_bid + total_ask)
            
            # Price impact
            if len(ticks) >= 10:
                price_changes = [float(ticks[i].price) - float(ticks[i-1].price) 
                               for i in range(1, len(ticks))]
                volume_changes = [ticks[i].volume for i in range(1, len(ticks))]
                
                if len(price_changes) > 0 and len(volume_changes) > 0:
                    # Simplified price impact measure
                    avg_price_change = np.mean(np.abs(price_changes))
                    avg_volume = np.mean(volume_changes)
                    if avg_volume > 0:
                        features['price_impact'] = avg_price_change / avg_volume
        
        except Exception as e:
            logger.error(f"Error extracting microstructure features: {e}")
        
        return features
    
    def _extract_temporal_features(self, symbol: str) -> Dict[str, float]:
        """Extract time-based features."""
        features = {}
        
        try:
            now = datetime.now()
            
            # Time of day features
            features['hour'] = now.hour
            features['minute'] = now.minute
            features['is_market_open'] = 1.0 if 9 <= now.hour < 16 else 0.0
            features['is_opening_hour'] = 1.0 if 9 <= now.hour < 10 else 0.0
            features['is_closing_hour'] = 1.0 if 15 <= now.hour < 16 else 0.0
            
            # Day of week
            features['day_of_week'] = now.weekday()
            features['is_monday'] = 1.0 if now.weekday() == 0 else 0.0
            features['is_friday'] = 1.0 if now.weekday() == 4 else 0.0
            
            # Month effects
            features['month'] = now.month
            features['is_month_end'] = 1.0 if now.day >= 25 else 0.0
            features['is_quarter_end'] = 1.0 if now.month in [3, 6, 9, 12] and now.day >= 25 else 0.0
            
            # Seasonal features
            features['sin_hour'] = np.sin(2 * np.pi * now.hour / 24)
            features['cos_hour'] = np.cos(2 * np.pi * now.hour / 24)
            features['sin_day'] = np.sin(2 * np.pi * now.weekday() / 7)
            features['cos_day'] = np.cos(2 * np.pi * now.weekday() / 7)
        
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
        
        return features
    
    def _extract_cross_asset_features(self, symbol: str) -> Dict[str, float]:
        """Extract cross-asset correlation features."""
        features = {}
        
        try:
            # This would require data from multiple assets
            # For now, create placeholder features
            features['market_correlation'] = 0.5  # Placeholder
            features['sector_correlation'] = 0.6  # Placeholder
            features['index_correlation'] = 0.7   # Placeholder
            
            # In a real implementation, you would:
            # 1. Calculate correlations with market indices (NIFTY, SENSEX)
            # 2. Calculate sector correlations
            # 3. Calculate correlations with related stocks
            # 4. Include currency and commodity correlations
        
        except Exception as e:
            logger.error(f"Error extracting cross-asset features: {e}")
        
        return features
    
    def _extract_regime_features(self, symbol: str) -> Dict[str, float]:
        """Extract market regime features."""
        features = {}
        
        try:
            prices = self.price_history[symbol]
            
            if len(prices) < 100:
                return features
            
            # Trend regime
            short_ma = sum(prices[-20:]) / 20
            long_ma = sum(prices[-50:]) / 50
            features['trend_regime'] = 1.0 if short_ma > long_ma else 0.0
            
            # Volatility regime
            recent_vol = np.std(prices[-20:])
            long_vol = np.std(prices[-100:])
            features['vol_regime'] = 1.0 if recent_vol > long_vol * 1.5 else 0.0
            
            # Mean reversion regime
            current_price = prices[-1]
            mean_price = sum(prices[-50:]) / 50
            std_price = np.std(prices[-50:])
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            features['mean_reversion_regime'] = 1.0 if abs(z_score) > 2 else 0.0
            
            # Momentum regime
            momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
            momentum_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0
            features['momentum_regime'] = 1.0 if momentum_5 > 0 and momentum_20 > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
        
        return features
    
    def _extract_sentiment_features(self, symbol: str) -> Dict[str, float]:
        """Extract market sentiment features."""
        features = {}
        
        try:
            # Placeholder for sentiment features
            # In a real implementation, you would include:
            # 1. News sentiment analysis
            # 2. Social media sentiment
            # 3. Options flow sentiment
            # 4. Insider trading activity
            # 5. Analyst recommendations
            
            features['news_sentiment'] = 0.0      # Neutral
            features['social_sentiment'] = 0.0    # Neutral
            features['options_sentiment'] = 0.0   # Neutral
            features['analyst_sentiment'] = 0.0   # Neutral
        
        except Exception as e:
            logger.error(f"Error extracting sentiment features: {e}")
        
        return features
    
    def get_feature_vector(self, symbol: str) -> Optional[np.ndarray]:
        """Get feature vector for ML models."""
        features = self.extract_features(symbol)
        if not features:
            return None
        
        # Combine all features into a single vector
        feature_vector = []
        
        for feature_dict in [
            features.price_features,
            features.technical_features,
            features.volume_features,
            features.volatility_features,
            features.microstructure_features,
            features.temporal_features,
            features.cross_asset_features,
            features.regime_features,
            features.sentiment_features
        ]:
            feature_vector.extend(feature_dict.values())
        
        return np.array(feature_vector, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        # This would return the names of all features in order
        # For now, return a placeholder
        return [f"feature_{i}" for i in range(100)]  # Placeholder


# Global feature engineer instance
feature_engineer = FeatureEngineer()