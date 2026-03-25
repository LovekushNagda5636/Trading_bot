"""
Correlation Risk Manager — Tracks inter-stock correlations and sector
exposure to prevent over-concentration in correlated assets.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


# ── Sector Mapping for Indian F&O Stocks ──────────────────────────────────

SECTOR_MAP = {
    # Banking & Financials
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "PNB": "Banking", "IDFCFIRSTB": "Banking",
    "FEDERALBNK": "Banking", "BANDHANBNK": "Banking", "AUBANK": "Banking",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "MANAPPURAM": "NBFC", "MUTHOOTFIN": "NBFC", "SBICARD": "NBFC",
    "HDFCAMC": "AMC", "HDFCLIFE": "Insurance", "SBILIFE": "Insurance",
    "ICICIPRULI": "Insurance", "ICICIGI": "Insurance",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT",
    "PERSISTENT": "IT", "LTTS": "IT",
    # Energy / Oil & Gas
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "GAIL": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "TATAPOWER": "Energy", "ADANIGREEN": "Energy",
    "ADANIENT": "Conglomerate",
    # Auto
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    "ASHOKLEY": "Auto", "TVSMOTOR": "Auto", "BALKRISIND": "Auto",
    # Metals & Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "COALINDIA": "Metals", "NMDC": "Metals",
    "SAIL": "Metals", "NATIONALUM": "Metals",
    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma", "BIOCON": "Pharma",
    "LUPIN": "Pharma", "AUROPHARMA": "Pharma",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "GODREJCP": "FMCG", "COLPAL": "FMCG", "TATACONSUM": "FMCG",
    "PIDILITIND": "FMCG",
    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",
    # Cement & Construction
    "ULTRACEMCO": "Cement", "SHREECEM": "Cement", "AMBUJACEM": "Cement",
    "ACC": "Cement", "GRASIM": "Cement", "DALMIASUGAR": "Cement",
    "RAMCOCEM": "Cement",
    "LT": "Infrastructure", "ABB": "Infrastructure",
    # Others
    "TITAN": "Consumer", "PAGEIND": "Consumer", "HAVELLS": "Consumer",
    "VOLTAS": "Consumer", "WHIRLPOOL": "Consumer",
    "ASIANPAINT": "Consumer", "BERGEPAINT": "Consumer",
    "PIIND": "Chemicals", "UPL": "Chemicals", "SRF": "Chemicals",
    "ATUL": "Chemicals", "DEEPAKNTR": "Chemicals",
}


class CorrelationRiskManager:
    """
    Manages correlation risk by:
    1. Tracking sector exposure of open positions
    2. Maintaining rolling return correlations between symbols
    3. Adjusting position sizes based on portfolio concentration
    4. Blocking new positions that would over-weight a sector
    """

    # Max percentage of capital in any single sector
    MAX_SECTOR_EXPOSURE_PCT = 30.0
    # Max number of positions in the same sector
    MAX_SAME_SECTOR_POSITIONS = 3
    # Correlation threshold above which positions are considered correlated
    CORRELATION_THRESHOLD = 0.70
    # Max total portfolio exposure as a fraction of capital
    MAX_PORTFOLIO_EXPOSURE = 0.80

    def __init__(self):
        # Current open positions: {symbol: {"qty": int, "value": float, "side": str}}
        self._positions: Dict[str, Dict] = {}
        # Return history for correlation: {symbol: [returns]}
        self._return_history: Dict[str, List[float]] = defaultdict(list)
        # Price history
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._max_history = 100  # Rolling window

    def update_position(self, symbol: str, qty: int, current_value: float, side: str = "BUY"):
        """Update or add a tracked position."""
        if qty > 0:
            self._positions[symbol] = {
                "qty": qty,
                "value": current_value,
                "side": side,
                "sector": self.get_sector(symbol),
            }
        elif symbol in self._positions:
            del self._positions[symbol]

    def remove_position(self, symbol: str):
        """Remove a closed position."""
        self._positions.pop(symbol, None)

    def update_price(self, symbol: str, price: float):
        """Update price history for correlation tracking."""
        history = self._price_history[symbol]
        if history:
            prev = history[-1]
            if prev > 0:
                ret = (price - prev) / prev
                self._return_history[symbol].append(ret)
                if len(self._return_history[symbol]) > self._max_history:
                    self._return_history[symbol] = self._return_history[symbol][-self._max_history:]

        history.append(price)
        if len(history) > self._max_history:
            self._price_history[symbol] = history[-self._max_history:]

    @staticmethod
    def get_sector(symbol: str) -> str:
        """Look up sector for a symbol."""
        return SECTOR_MAP.get(symbol, "Unknown")

    # ── Exposure Checks ──────────────────────────────────────────────────────

    def get_sector_exposure(self) -> Dict[str, Dict]:
        """
        Get current exposure broken down by sector.

        Returns:
            Dict[sector_name, {"value": float, "count": int, "symbols": list}]
        """
        exposure = defaultdict(lambda: {"value": 0.0, "count": 0, "symbols": []})

        for sym, pos in self._positions.items():
            sector = pos.get("sector", "Unknown")
            exposure[sector]["value"] += pos["value"]
            exposure[sector]["count"] += 1
            exposure[sector]["symbols"].append(sym)

        return dict(exposure)

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure value."""
        return sum(pos["value"] for pos in self._positions.values())

    def can_enter_position(
        self,
        symbol: str,
        proposed_value: float,
        total_capital: float,
    ) -> Dict[str, Any]:
        """
        Check if a new position is allowed given correlation/sector constraints.

        Returns:
            Dict with 'allowed' (bool), 'reason' (str), 'adjustments' (dict)
        """
        sector = self.get_sector(symbol)
        sector_exposure = self.get_sector_exposure()
        total_exposure = self.get_total_exposure()

        result = {
            "allowed": True,
            "reason": "OK",
            "adjustments": {},
            "sector": sector,
        }

        # ── Check 1: Total portfolio exposure ──────────────────────────────
        if total_capital > 0:
            total_with_new = total_exposure + proposed_value
            if total_with_new / total_capital > self.MAX_PORTFOLIO_EXPOSURE:
                result["allowed"] = False
                result["reason"] = (
                    f"Total exposure would reach {total_with_new/total_capital*100:.1f}% "
                    f"(max {self.MAX_PORTFOLIO_EXPOSURE*100:.0f}%)"
                )
                return result

        # ── Check 2: Sector concentration ──────────────────────────────────
        if sector in sector_exposure:
            sec = sector_exposure[sector]
            new_sector_value = sec["value"] + proposed_value

            if total_capital > 0:
                sector_pct = new_sector_value / total_capital * 100
                if sector_pct > self.MAX_SECTOR_EXPOSURE_PCT:
                    result["allowed"] = False
                    result["reason"] = (
                        f"Sector '{sector}' exposure would reach {sector_pct:.1f}% "
                        f"(max {self.MAX_SECTOR_EXPOSURE_PCT:.0f}%)"
                    )
                    return result

            if sec["count"] >= self.MAX_SAME_SECTOR_POSITIONS:
                result["allowed"] = False
                result["reason"] = (
                    f"Already {sec['count']} positions in '{sector}' "
                    f"(max {self.MAX_SAME_SECTOR_POSITIONS})"
                )
                return result

        # ── Check 3: Correlation with existing positions ───────────────────
        correlated_positions = self._find_correlated_positions(symbol)
        if correlated_positions:
            corr_count = len(correlated_positions)
            if corr_count >= 2:
                # More than 2 highly correlated positions → reduce size
                size_factor = max(0.3, 1.0 - 0.25 * corr_count)
                result["adjustments"]["size_factor"] = size_factor
                result["adjustments"]["reason"] = (
                    f"Correlated with {corr_count} existing positions: "
                    f"{[p[0] for p in correlated_positions]}"
                )
                result["reason"] = f"Allowed (size reduced to {size_factor:.0%} due to correlation)"

        return result

    def _find_correlated_positions(self, symbol: str) -> List[Tuple[str, float]]:
        """Find existing positions that are highly correlated with the given symbol."""
        correlated = []
        sym_returns = self._return_history.get(symbol, [])

        if len(sym_returns) < 20:
            # Not enough data — fall back to sector-based correlation
            target_sector = self.get_sector(symbol)
            for pos_sym in self._positions:
                pos_sector = self.get_sector(pos_sym)
                if pos_sector == target_sector and pos_sym != symbol:
                    correlated.append((pos_sym, 0.75))  # Assume high correlation
            return correlated

        for pos_sym in self._positions:
            if pos_sym == symbol:
                continue

            pos_returns = self._return_history.get(pos_sym, [])
            if len(pos_returns) < 20:
                continue

            # Compute Pearson correlation
            min_len = min(len(sym_returns), len(pos_returns))
            a = np.array(sym_returns[-min_len:])
            b = np.array(pos_returns[-min_len:])

            if np.std(a) == 0 or np.std(b) == 0:
                continue

            corr = np.corrcoef(a, b)[0, 1]

            if abs(corr) >= self.CORRELATION_THRESHOLD:
                correlated.append((pos_sym, round(float(corr), 3)))

        return correlated

    def get_position_size_multiplier(
        self,
        symbol: str,
        proposed_value: float,
        total_capital: float,
    ) -> float:
        """
        Get a position size multiplier (0.0 to 1.0) that accounts for:
        - Sector concentration
        - Correlation with existing positions
        - Total portfolio exposure

        Use: adjusted_size = base_size * multiplier
        """
        check = self.can_enter_position(symbol, proposed_value, total_capital)

        if not check["allowed"]:
            return 0.0

        multiplier = check.get("adjustments", {}).get("size_factor", 1.0)

        # Additional gradual reduction as portfolio fills up
        total_exposure = self.get_total_exposure()
        if total_capital > 0:
            utilization = total_exposure / total_capital
            if utilization > 0.5:
                # Gradually reduce: at 80% utilization → 0.5x multiplier
                reduction = max(0.5, 1.0 - (utilization - 0.5) * 1.67)
                multiplier *= reduction

        return round(min(multiplier, 1.0), 3)

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of current portfolio risk metrics."""
        sector_exposure = self.get_sector_exposure()
        total_exposure = self.get_total_exposure()

        return {
            "total_positions": len(self._positions),
            "total_exposure": round(total_exposure, 2),
            "sector_exposure": {
                sec: {
                    "value": round(data["value"], 2),
                    "count": data["count"],
                    "symbols": data["symbols"],
                }
                for sec, data in sector_exposure.items()
            },
            "top_sector": (
                max(sector_exposure, key=lambda s: sector_exposure[s]["value"])
                if sector_exposure
                else "None"
            ),
            "position_symbols": list(self._positions.keys()),
        }
