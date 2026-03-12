"""
F&O / Options / Futures Trading Module
Scans for Options and Futures opportunities using Angel One SmartAPI.
Integrates with the adaptive learning engine.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from SmartApi import SmartConnect

logger = logging.getLogger(__name__)


# ── Option Chain & Futures Token Builder ──────────────────────────────────────

# Expiry helper — get nearest Thursday for weekly options
def _get_nearest_expiry() -> str:
    """Get nearest Thursday expiry date in DDMMMYYYY format."""
    today = datetime.now()
    days_until_thursday = (3 - today.weekday()) % 7
    if days_until_thursday == 0 and today.hour >= 15:
        days_until_thursday = 7
    expiry = today + timedelta(days=days_until_thursday)
    return expiry.strftime("%d%b%Y").upper()


def _get_monthly_expiry() -> str:
    """Get last Thursday of current month for monthly expiry."""
    today = datetime.now()
    # Go to last day of month
    if today.month == 12:
        last_day = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        last_day = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    
    # Walk back to Thursday
    while last_day.weekday() != 3:
        last_day -= timedelta(days=1)
    
    return last_day.strftime("%d%b%Y").upper()


class FnOScanner:
    """
    Scans for Futures and Options trading opportunities.
    - Index Options (NIFTY, BANKNIFTY weekly)
    - Stock Futures (momentum/breakout based)
    - Option straddles/strangles based on volatility
    """

    def __init__(self, smart_api: SmartConnect = None):
        self.smart_api = smart_api
        self.option_chain_cache: Dict[str, List] = {}
        self.futures_data_cache: Dict[str, Dict] = {}
        
        # Strategy configs (will be tuned by adaptive engine)
        self.config = {
            "index_option_budget": 10000,
            "stock_futures_budget": 15000,
            "max_option_positions": 3,
            "max_futures_positions": 2,
            "option_sl_pct": 30,  # % of premium
            "option_target_pct": 50,  # % of premium
            "futures_sl_pct": 0.8,  # % of futures price
            "futures_target_pct": 1.5,
        }

    def set_api(self, smart_api: SmartConnect):
        """Set the API connection."""
        self.smart_api = smart_api

    # ── Index Options Scanning ───────────────────────────────────────────────

    def scan_index_options(self, index_data: Dict[str, Dict],
                           market_context: Dict = None) -> List[Dict]:
        """
        Scan for index option opportunities (NIFTY/BANKNIFTY).
        Uses market regime to decide direction and strike selection.
        """
        if not self.smart_api:
            return []
        
        ctx = market_context or {}
        regime = ctx.get("regime", "unknown")
        volatility = ctx.get("volatility", 0)
        opportunities = []
        
        for index_name in ["NIFTY", "BANKNIFTY"]:
            data = index_data.get(index_name, {})
            ltp = data.get("ltp", 0)
            if ltp <= 0:
                continue
            
            change_pct = data.get("change_pct", 0)
            
            # Calculate ATM strike
            if index_name == "NIFTY":
                step = 50
                lot_size = 25
            else:
                step = 100
                lot_size = 15
            
            atm_strike = round(ltp / step) * step
            
            # Generate option opportunities based on regime
            opps = self._generate_option_signals(
                index_name, ltp, atm_strike, step, lot_size,
                change_pct, regime, volatility
            )
            opportunities.extend(opps)
        
        return opportunities

    def _generate_option_signals(self, index: str, ltp: float, atm: int,
                                  step: int, lot: int, change_pct: float,
                                  regime: str, volatility: float) -> List[Dict]:
        """Generate option trading signals based on market analysis."""
        signals = []
        expiry = _get_nearest_expiry()
        
        # ── Strategy 1: Directional Option Buying ───
        if regime == "trending_up" and change_pct > 0.3:
            # Buy Call
            strike = atm  # ATM Call
            premium_est = max(50, ltp * 0.005)  # Rough estimate
            signals.append({
                "symbol": f"{index}{expiry}{strike}CE",
                "underlying": index,
                "instrument_type": "OPT",
                "option_type": "CE",
                "strike": strike,
                "expiry": expiry,
                "direction": "BUY",
                "ltp": premium_est,
                "lot_size": lot,
                "qty": lot,
                "margin": round(premium_est * lot, 2),
                "target_1": round(premium_est * 1.5, 2),
                "stop_loss": round(premium_est * 0.7, 2),
                "max_profit": round(premium_est * 0.5 * lot, 2),
                "max_loss": round(premium_est * 0.3 * lot, 2),
                "risk_reward": round(0.5 / 0.3, 2),
                "score": 55 + min(20, abs(change_pct) * 8),
                "priority": "HIGH" if change_pct > 0.8 else "MEDIUM",
                "strategies": ["Bullish Option Buy"],
                "num_strategies": 1,
                "reasons": [f"{index} trending up {change_pct:+.1f}%, ATM CE buy"],
                "source": "FnO_Scanner",
                "ts": datetime.now().isoformat(),
            })
        
        elif regime == "trending_down" and change_pct < -0.3:
            # Buy Put
            strike = atm
            premium_est = max(50, ltp * 0.005)
            signals.append({
                "symbol": f"{index}{expiry}{strike}PE",
                "underlying": index,
                "instrument_type": "OPT",
                "option_type": "PE",
                "strike": strike,
                "expiry": expiry,
                "direction": "BUY",
                "ltp": premium_est,
                "lot_size": lot,
                "qty": lot,
                "margin": round(premium_est * lot, 2),
                "target_1": round(premium_est * 1.5, 2),
                "stop_loss": round(premium_est * 0.7, 2),
                "max_profit": round(premium_est * 0.5 * lot, 2),
                "max_loss": round(premium_est * 0.3 * lot, 2),
                "risk_reward": round(0.5 / 0.3, 2),
                "score": 55 + min(20, abs(change_pct) * 8),
                "priority": "HIGH" if abs(change_pct) > 0.8 else "MEDIUM",
                "strategies": ["Bearish Option Buy"],
                "num_strategies": 1,
                "reasons": [f"{index} trending down {change_pct:+.1f}%, ATM PE buy"],
                "source": "FnO_Scanner",
                "ts": datetime.now().isoformat(),
            })
        
        # ── Strategy 2: Volatility Play — Straddle ───
        if regime in ("volatile", "unknown") and volatility > 1.5:
            # Long Straddle when expecting big move
            ce_premium = max(50, ltp * 0.004)
            pe_premium = max(50, ltp * 0.004)
            total_premium = ce_premium + pe_premium
            
            signals.append({
                "symbol": f"{index}{expiry}{atm}STRDL",
                "underlying": index,
                "instrument_type": "OPT",
                "option_type": "STRADDLE",
                "strike": atm,
                "expiry": expiry,
                "direction": "BUY",
                "ltp": total_premium,
                "lot_size": lot,
                "qty": lot,
                "margin": round(total_premium * lot, 2),
                "target_1": round(total_premium * 1.4, 2),
                "stop_loss": round(total_premium * 0.7, 2),
                "max_profit": round(total_premium * 0.4 * lot, 2),
                "max_loss": round(total_premium * 0.3 * lot, 2),
                "risk_reward": round(0.4 / 0.3, 2),
                "score": 50 + min(15, volatility * 5),
                "priority": "MEDIUM",
                "strategies": ["Long Straddle"],
                "num_strategies": 1,
                "reasons": [f"High volatility ({volatility:.1f}%), expecting big move"],
                "source": "FnO_Scanner",
                "ts": datetime.now().isoformat(),
            })
        
        # ── Strategy 3: Short Straddle (Sideways) ───
        if regime == "sideways" and volatility < 1.2:
            # Sell Straddle when expecting range-bound
            ce_premium = max(50, ltp * 0.004)
            pe_premium = max(50, ltp * 0.004)
            total_premium = ce_premium + pe_premium
            
            signals.append({
                "symbol": f"{index}{expiry}{atm}SS",
                "underlying": index,
                "instrument_type": "OPT",
                "option_type": "SHORT_STRADDLE",
                "strike": atm,
                "expiry": expiry,
                "direction": "SELL",
                "ltp": total_premium,
                "lot_size": lot,
                "qty": lot,
                "margin": round(total_premium * lot * 3, 2),  # Higher margin for selling
                "target_1": round(total_premium * 0.6, 2),  # Keep 40% premium
                "stop_loss": round(total_premium * 1.5, 2),
                "max_profit": round(total_premium * 0.4 * lot, 2),
                "max_loss": round(total_premium * 0.5 * lot, 2),
                "risk_reward": round(0.4 / 0.5, 2),
                "score": 45 + min(15, (1.5 - volatility) * 20),
                "priority": "MEDIUM",
                "strategies": ["Short Straddle"],
                "num_strategies": 1,
                "reasons": [f"Low volatility ({volatility:.1f}%), range-bound expected"],
                "source": "FnO_Scanner",
                "ts": datetime.now().isoformat(),
            })
        
        # ── Strategy 4: OTM Option Buy (Breakout Play) ───
        if abs(change_pct) > 1.0:
            if change_pct > 0:
                otm_strike = atm + step * 2  # 2 steps OTM
                premium_est = max(20, ltp * 0.002)
                signals.append({
                    "symbol": f"{index}{expiry}{otm_strike}CE",
                    "underlying": index,
                    "instrument_type": "OPT",
                    "option_type": "CE",
                    "strike": otm_strike,
                    "expiry": expiry,
                    "direction": "BUY",
                    "ltp": premium_est,
                    "lot_size": lot,
                    "qty": lot,
                    "margin": round(premium_est * lot, 2),
                    "target_1": round(premium_est * 2.0, 2),  # Double
                    "stop_loss": round(premium_est * 0.5, 2),
                    "max_profit": round(premium_est * 1.0 * lot, 2),
                    "max_loss": round(premium_est * 0.5 * lot, 2),
                    "risk_reward": 2.0,
                    "score": 40 + min(20, abs(change_pct) * 6),
                    "priority": "MEDIUM",
                    "strategies": ["OTM Breakout Call"],
                    "num_strategies": 1,
                    "reasons": [f"{index} strong move +{change_pct:.1f}%, OTM breakout play"],
                    "source": "FnO_Scanner",
                    "ts": datetime.now().isoformat(),
                })
            else:
                otm_strike = atm - step * 2
                premium_est = max(20, ltp * 0.002)
                signals.append({
                    "symbol": f"{index}{expiry}{otm_strike}PE",
                    "underlying": index,
                    "instrument_type": "OPT",
                    "option_type": "PE",
                    "strike": otm_strike,
                    "expiry": expiry,
                    "direction": "BUY",
                    "ltp": premium_est,
                    "lot_size": lot,
                    "qty": lot,
                    "margin": round(premium_est * lot, 2),
                    "target_1": round(premium_est * 2.0, 2),
                    "stop_loss": round(premium_est * 0.5, 2),
                    "max_profit": round(premium_est * 1.0 * lot, 2),
                    "max_loss": round(premium_est * 0.5 * lot, 2),
                    "risk_reward": 2.0,
                    "score": 40 + min(20, abs(change_pct) * 6),
                    "priority": "MEDIUM",
                    "strategies": ["OTM Breakout Put"],
                    "num_strategies": 1,
                    "reasons": [f"{index} strong move {change_pct:.1f}%, OTM breakout play"],
                    "source": "FnO_Scanner",
                    "ts": datetime.now().isoformat(),
                })
        
        return signals

    # ── Stock Futures Scanning ───────────────────────────────────────────────

    def scan_stock_futures(self, equity_data: Dict[str, Dict],
                           market_context: Dict = None) -> List[Dict]:
        """
        Scan for stock futures opportunities.
        Looks for strong momentum/breakout stocks suitable for futures.
        """
        ctx = market_context or {}
        regime = ctx.get("regime", "unknown")
        budget = ctx.get("budget", 25000)
        opportunities = []
        expiry = _get_monthly_expiry()
        
        for sym, data in equity_data.items():
            if data.get("instrument_type") == "INDEX":
                continue
            
            ltp = data.get("ltp", 0)
            if ltp <= 0:
                continue
            
            change_pct = data.get("change_pct", 0)
            volume = data.get("volume", 0)
            high = data.get("high", ltp)
            low = data.get("low", ltp)
            open_p = data.get("open", ltp)
            
            # ── Exhaustion filter: don't chase extended moves ──
            move_from_open = (ltp - open_p) / open_p * 100 if open_p > 0 else 0
            day_range = (high - low) / open_p * 100 if open_p > 0 else 0
            
            # Futures need strong signals
            score = 0
            reasons = []
            strategies = []
            direction = None
            
            # ── Strong Momentum ───
            if abs(change_pct) > 2.5 and volume > 1_000_000:
                # ── NEW: Diminishing returns for very extended moves ──
                if abs(change_pct) <= 4.0:
                    score += 30
                else:
                    # Likely exhausted after 4%+ — chase signal weakens
                    score += 20
                direction = "BUY" if change_pct > 0 else "SELL"
                strategies.append("Futures Momentum")
                reasons.append(f"Strong {change_pct:+.1f}% move with volume {volume/1e6:.1f}M")
            
            # ── Breakout with Volume ───
            range_pct = (high - low) / open_p * 100 if open_p > 0 else 0
            if range_pct > 2.0 and volume > 500_000:
                if ltp >= high * 0.998:  # Near day high
                    score += 25
                    direction = "BUY"
                    strategies.append("Futures Breakout")
                    reasons.append(f"Breakout at high, range {range_pct:.1f}%")
                elif ltp <= low * 1.002:  # Near day low
                    score += 25
                    direction = "SELL"
                    strategies.append("Futures Breakdown")
                    reasons.append(f"Breakdown at low, range {range_pct:.1f}%")
            
            # ── Sector Rotation ───
            if regime in ("trending_up", "trending_down"):
                sector_trend = ctx.get("sector_trends", {})
                for sector, symbols_list in {
                    "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"],
                    "it": ["TCS", "INFY", "HCLTECH", "WIPRO"],
                    "auto": ["MARUTI", "TATAMOTORS", "M&M"],
                }.items():
                    if sym in symbols_list:
                        if sector_trend.get(sector) == "bullish" and change_pct > 1.0:
                            score += 15
                            direction = direction or "BUY"
                            strategies.append("Sector Rotation")
                            reasons.append(f"{sector.upper()} sector bullish")
                        elif sector_trend.get(sector) == "bearish" and change_pct < -1.0:
                            score += 15
                            direction = direction or "SELL"
                            strategies.append("Sector Rotation")
                            reasons.append(f"{sector.upper()} sector bearish")
            
            if score < 30 or direction is None:
                continue
            
            # ── NEW: Exhaustion filter for FnO ──
            if direction == "BUY" and move_from_open > 4.0 and high > low:
                if (ltp - low) / (high - low) > 0.85:
                    continue  # Too extended, skip
            elif direction == "SELL" and move_from_open < -4.0 and high > low:
                if (ltp - low) / (high - low) < 0.15:
                    continue  # Too extended, skip
            
            # ── FIXED: Budget-aware lot sizing ──
            # Use maximum 25% of budget for FnO margin (~15% margin requirement)
            max_fno_notional = budget * 0.25 / 0.15  # Max notional value per FnO trade
            lot_size = max(1, int(max_fno_notional / ltp))
            lot_size = min(lot_size, 50)  # Hard cap to prevent huge positions
            
            sl_pct = self.config["futures_sl_pct"] / 100
            tgt_pct = self.config["futures_target_pct"] / 100
            
            if direction == "BUY":
                sl = round(ltp * (1 - sl_pct), 2)
                t1 = round(ltp * (1 + tgt_pct), 2)
            else:
                sl = round(ltp * (1 + sl_pct), 2)
                t1 = round(ltp * (1 - tgt_pct), 2)
            
            margin = round(ltp * lot_size * 0.15, 2)  # ~15% margin
            
            # Skip if margin exceeds budget allocation
            if margin > budget * 0.30:
                lot_size = max(1, int(budget * 0.30 / (ltp * 0.15)))
                margin = round(ltp * lot_size * 0.15, 2)
            
            opportunities.append({
                "symbol": f"{sym}FUT{expiry[:5]}",
                "underlying": sym,
                "instrument_type": "FUT",
                "direction": direction,
                "ltp": ltp,
                "open": open_p,
                "high": high,
                "low": low,
                "prev_close": data.get("prev_close", ltp),
                "change_pct": change_pct,
                "volume": volume,
                "lot_size": lot_size,
                "qty": lot_size,
                "margin": margin,
                "target_1": t1,
                "target_2": round(t1 + (t1 - ltp) * 0.5, 2),
                "stop_loss": sl,
                "max_profit": round(abs(t1 - ltp) * lot_size, 2),
                "max_loss": round(abs(sl - ltp) * lot_size, 2),
                "risk_reward": round(abs(t1 - ltp) / abs(sl - ltp), 2) if abs(sl - ltp) > 0 else 0,
                "score": score,
                "confidence": 0.5,
                "priority": "HIGH" if score >= 50 else "MEDIUM",
                "strategies": strategies,
                "num_strategies": len(strategies),
                "reasons": reasons,
                "source": "FnO_Scanner",
                "ts": datetime.now().isoformat(),
            })
        
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities[:10]  # Top 10

    # ── Combined Scan ────────────────────────────────────────────────────────

    def scan_all(self, market_data: Dict[str, Dict],
                  market_context: Dict = None) -> List[Dict]:
        """
        Run all F&O scans and return combined opportunities.
        """
        all_opps = []
        
        # Index options
        index_data = {k: v for k, v in market_data.items()
                     if v.get("instrument_type") == "INDEX"}
        index_opps = self.scan_index_options(index_data, market_context)
        all_opps.extend(index_opps)
        
        # Stock futures
        futures_opps = self.scan_stock_futures(market_data, market_context)
        all_opps.extend(futures_opps)
        
        # Sort by score
        all_opps.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"📈 F&O Scanner: {len(index_opps)} option signals, "
                    f"{len(futures_opps)} futures signals")
        
        return all_opps
