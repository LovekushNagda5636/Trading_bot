"""
Market Regime Detector — Real-time market regime classification.
Detects whether market is trending, sideways, volatile, etc.
and provides context for adaptive strategy selection.
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Classifies the current market regime using multiple indicators:
    - Trend detection via EMA slope
    - Volatility via intraday range
    - Volume analysis
    - Index momentum
    """

    def __init__(self):
        # Rolling data for regime detection
        self._nifty_prices: deque = deque(maxlen=100)
        self._banknifty_prices: deque = deque(maxlen=100)
        self._market_data_snapshots: deque = deque(maxlen=50)
        
        self.current_regime: str = "unknown"
        self.regime_confidence: float = 0.0
        self.regime_duration_minutes: float = 0.0
        self._last_regime_change: datetime = datetime.now()
        
        # Sector tracking
        self._sector_trends: Dict[str, str] = {}
        
        # VIX proxy (computed from market data)
        self.implied_volatility: float = 0.0

    def update(self, market_data: Dict[str, Dict]) -> Dict:
        """
        Update regime detection with new market data.
        
        Args:
            market_data: Dict of symbol -> market data from scanner
            
        Returns:
            Dict with regime info and market context
        """
        # Extract index data
        nifty = market_data.get("NIFTY", {})
        banknifty = market_data.get("BANKNIFTY", {})
        
        if nifty.get("ltp", 0) > 0:
            self._nifty_prices.append(nifty["ltp"])
        if banknifty.get("ltp", 0) > 0:
            self._banknifty_prices.append(banknifty["ltp"])
        
        # Store snapshot
        snapshot = {
            "time": datetime.now().isoformat(),
            "nifty_ltp": nifty.get("ltp", 0),
            "nifty_change": nifty.get("change_pct", 0),
            "banknifty_ltp": banknifty.get("ltp", 0),
            "banknifty_change": banknifty.get("change_pct", 0),
            "total_advancing": 0,
            "total_declining": 0,
            "avg_volume_ratio": 1.0,
        }
        
        # Count advancing/declining
        for sym, data in market_data.items():
            if data.get("instrument_type") == "INDEX":
                continue
            if data.get("change_pct", 0) > 0:
                snapshot["total_advancing"] += 1
            elif data.get("change_pct", 0) < 0:
                snapshot["total_declining"] += 1
        
        self._market_data_snapshots.append(snapshot)
        
        # Detect regime
        prev_regime = self.current_regime
        self.current_regime, self.regime_confidence = self._classify_regime(market_data)
        
        if prev_regime != self.current_regime:
            self._last_regime_change = datetime.now()
            logger.info(f"📊 Regime change: {prev_regime} → {self.current_regime} "
                       f"(confidence: {self.regime_confidence:.2f})")
        
        self.regime_duration_minutes = (datetime.now() - self._last_regime_change).total_seconds() / 60
        
        # Update sector trends
        self._update_sector_trends(market_data)
        
        # Compute implied volatility
        self.implied_volatility = self._compute_implied_volatility(market_data)
        
        return self.get_context(market_data)

    def _classify_regime(self, market_data: Dict) -> Tuple[str, float]:
        """
        Classify market regime based on multiple factors.
        Returns (regime, confidence).
        """
        nifty = market_data.get("NIFTY", {})
        
        # Factor 1: Index momentum
        nifty_change = nifty.get("change_pct", 0)
        
        # Factor 2: Breadth (advancing vs declining)
        advancing = 0
        declining = 0
        flat = 0
        total_range_pct = 0
        count = 0
        
        for sym, data in market_data.items():
            if data.get("instrument_type") == "INDEX":
                continue
            chg = data.get("change_pct", 0)
            if chg > 0.3:
                advancing += 1
            elif chg < -0.3:
                declining += 1
            else:
                flat += 1
            
            # Range analysis
            hl = data.get("high", 0) - data.get("low", 0)
            op = data.get("open", 0)
            if op > 0:
                total_range_pct += (hl / op * 100)
                count += 1
        
        total = advancing + declining + flat
        avg_range = total_range_pct / count if count > 0 else 0
        
        if total == 0:
            return "unknown", 0.0
        
        adv_ratio = advancing / total
        dec_ratio = declining / total
        breadth_diff = adv_ratio - dec_ratio
        
        # Factor 3: Index trend via prices
        trend_slope = 0
        if len(self._nifty_prices) >= 5:
            prices = list(self._nifty_prices)
            # Simple linear regression slope
            x = np.arange(len(prices))
            if len(prices) > 1:
                slope = np.polyfit(x, prices, 1)[0]
                trend_slope = slope / np.mean(prices) * 100  # Normalized
        
        # Classification logic
        votes = {"trending_up": 0, "trending_down": 0, "sideways": 0, "volatile": 0}
        
        # Momentum vote
        if nifty_change > 0.5:
            votes["trending_up"] += 2
        elif nifty_change < -0.5:
            votes["trending_down"] += 2
        else:
            votes["sideways"] += 1
        
        # Breadth vote
        if breadth_diff > 0.2:
            votes["trending_up"] += 2
        elif breadth_diff < -0.2:
            votes["trending_down"] += 2
        else:
            votes["sideways"] += 1
        
        # Range vote (volatility)
        if avg_range > 2.5:
            votes["volatile"] += 3
        elif avg_range > 1.5:
            votes["volatile"] += 1
        elif avg_range < 0.8:
            votes["sideways"] += 2
        
        # Trend slope vote
        if trend_slope > 0.01:
            votes["trending_up"] += 1
        elif trend_slope < -0.01:
            votes["trending_down"] += 1
        
        # Pick winner
        regime = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[regime] / total_votes if total_votes > 0 else 0
        
        return regime, round(confidence, 3)

    def _update_sector_trends(self, market_data: Dict):
        """Track sector-level trends."""
        sector_map = {
            "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                        "BANKBARODA", "PNB", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB"],
            "it": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM",
                   "MPHASIS", "COFORGE", "PERSISTENT"],
            "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
                       "AUROPHARMA", "BIOCON", "LUPIN", "TORNTPHARM"],
            "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT",
                     "HEROMOTOCO", "ASHOKLEY", "TVSMOTOR"],
            "metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL", "JINDALSTEL"],
            "energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "TATAPOWER"],
            "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
                     "GODREJCP", "MARICO", "COLPAL"],
        }
        
        for sector, symbols in sector_map.items():
            changes = [market_data[s]["change_pct"] for s in symbols
                      if s in market_data and "change_pct" in market_data[s]]
            if not changes:
                continue
            
            avg_change = np.mean(changes)
            if avg_change > 0.5:
                self._sector_trends[sector] = "bullish"
            elif avg_change < -0.5:
                self._sector_trends[sector] = "bearish"
            else:
                self._sector_trends[sector] = "neutral"

    def _compute_implied_volatility(self, market_data: Dict) -> float:
        """Compute a proxy for market implied volatility."""
        ranges = []
        for sym, data in market_data.items():
            if data.get("instrument_type") == "INDEX":
                continue
            high = data.get("high", 0)
            low = data.get("low", 0)
            open_p = data.get("open", 0)
            if open_p > 0 and high > low:
                ranges.append((high - low) / open_p * 100)
        
        if not ranges:
            return 0.0
        
        return round(np.mean(ranges), 2)

    def get_context(self, market_data: Dict = None) -> Dict:
        """Get full market context for decision making."""
        nifty = (market_data or {}).get("NIFTY", {})
        
        return {
            "regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "regime_duration_minutes": round(self.regime_duration_minutes, 1),
            "volatility": self.implied_volatility,
            "nifty_change_pct": nifty.get("change_pct", 0),
            "nifty_ltp": nifty.get("ltp", 0),
            "sector_trends": dict(self._sector_trends),
            "vix": self.implied_volatility * 5,  # Rough VIX proxy
            "timestamp": datetime.now().isoformat(),
        }

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Get the sector trend for a symbol."""
        sector_map = {
            "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                        "BANKBARODA", "PNB", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB"],
            "it": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
            "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
            "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO"],
            "metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL"],
            "energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL"],
            "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA"],
        }
        
        for sector, symbols in sector_map.items():
            if symbol in symbols:
                return self._sector_trends.get(sector, "neutral")
        return "neutral"
