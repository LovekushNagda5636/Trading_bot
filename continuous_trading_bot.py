#!/usr/bin/env python3
"""
ANGEL ONE SELF-LEARNING TRADING BOT
====================================
Full market scanner with AI self-learning capabilities:
- 8+ intraday equity strategies
- F&O / Options / Futures scanner
- Market regime detection
- Adaptive strategy weighting (learns from past trades)
- Dynamic SL/target adjustment
- Trailing stop-losses
- Trade journaling with mistake classification
- End-of-session learning cycle

The bot gets smarter every time it runs.
"""

import time
import tempfile
import threading
import schedule
import logging
import logging.handlers
import json
import sys
import os
import traceback
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from SmartApi import SmartConnect
from collections import defaultdict
from decimal import Decimal

# Load .env for credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from angel_one_auth_service import AngelOneAuth

# ── Directories ───────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
Path("data/journal").mkdir(parents=True, exist_ok=True)
Path("data/models").mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            "logs/trading_bot.log", encoding="utf-8",
            maxBytes=10_000_000, backupCount=5  # 10MB, keep 5 rotated files
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party logs (SmartAPI AB1019, urllib3 retries, etc.)
# SmartAPI uses logzero internally — silence it at the root level.
logging.getLogger("smartConnect").setLevel(logging.CRITICAL)
logging.getLogger("SmartApi").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
try:
    import logzero
    logzero.logger.setLevel(logging.CRITICAL)
except Exception:
    pass


def load_config() -> Dict:
    try:
        with open("config/angel_one_config.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Config load error: {e}")
        return {}


# ── Market hours ──────────────────────────────────────────────────────────────
def is_weekday():
    return datetime.now().weekday() < 5

def is_market_hours():
    now = datetime.now().strftime("%H:%M")
    return is_weekday() and "09:15" <= now <= "15:30"

def is_pre_market():
    now = datetime.now().strftime("%H:%M")
    return is_weekday() and "09:00" <= now < "09:15"


# ── Self-Learning Imports ─────────────────────────────────────────────────────
from trading_bot.ml.trade_journal import TradeJournal, TradeRecord
from trading_bot.ml.adaptive_strategy_engine import AdaptiveStrategyEngine
from trading_bot.ml.regime_detector import MarketRegimeDetector
from trading_bot.ml.fno_scanner import FnOScanner

# ── v3.0 Enhanced Modules ────────────────────────────────────────────────────
from trading_bot.data.candle_manager import CandleManager
from trading_bot.ml.ensemble_scorer import EnsembleScorer
from trading_bot.ml.micro_learner import MicroLearner
from trading_bot.risk.transaction_costs import TransactionCostModel
from trading_bot.risk.correlation_manager import CorrelationRiskManager
from trading_bot.analysis.multi_timeframe import MultiTimeframeAnalyzer
from trading_bot.ml.model_trainer import ModelTrainer

# ── Angel One Market Data Fetcher ─────────────────────────────────────────────

# Complete F&O token map: {symbol: NSE_token}
FNO_TOKENS = {
    # NIFTY 50
    "RELIANCE": "2885", "TCS": "11536", "HDFCBANK": "1333", "INFY": "1594",
    "ICICIBANK": "4963", "KOTAKBANK": "1922", "HINDUNILVR": "1394",
    "SBIN": "3045", "BHARTIARTL": "10604", "ITC": "1660", "LT": "11483",
    "AXISBANK": "5900", "MARUTI": "10999", "BAJFINANCE": "317",
    "HCLTECH": "7229", "WIPRO": "3787", "ULTRACEMCO": "11532",
    "NESTLEIND": "17963", "BAJAJFINSV": "16675", "TITAN": "3506",
    "ADANIPORTS": "15083", "POWERGRID": "14977", "NTPC": "11630",
    "TECHM": "13538", "COALINDIA": "20374", "JSWSTEEL": "11723",
    "GRASIM": "1232", "HINDALCO": "1363", "INDUSINDBK": "5258",
    "M&M": "2031", "HDFCLIFE": "467", "SBILIFE": "21808",
    "BPCL": "526", "BRITANNIA": "547", "DIVISLAB": "10940",
    "EICHERMOT": "910", "HEROMOTOCO": "1348", "CIPLA": "694",
    "DRREDDY": "881", "SUNPHARMA": "3351", "TATASTEEL": "3499",
    "ONGC": "2475", "APOLLOHOSP": "157", "BAJAJ-AUTO": "16669",
    "TATAMOTORS": "3456", "LTIM": "17818", "ADANIENT": "25",
    "TRENT": "1964", "ASIANPAINT": "236",
    # Banking & Finance
    "BANKBARODA": "4668", "PNB": "10666", "CANBK": "10794",
    "IDFCFIRSTB": "11184", "FEDERALBNK": "1023", "AUBANK": "21238",
    "MUTHOOTFIN": "23650", "CHOLAFIN": "685", "LICHSGFIN": "1997",
    "SBICARD": "17971", "SHRIRAMFIN": "4306", "PEL": "2923",
    "RECLTD": "15355", "PFC": "14299",
    # Auto & Ancillary
    "ASHOKLEY": "212", "TVSMOTOR": "8479", "MRF": "2277",
    "MOTHERSON": "21616", "BHARATFORG": "422", "EXIDEIND": "6994",
    # IT & Tech
    "MPHASIS": "4503", "COFORGE": "11543", "PERSISTENT": "18365",
    "LTTS": "18564", "NAUKRI": "13751",
    # Pharma
    "AUROPHARMA": "275", "BIOCON": "11373", "TORNTPHARM": "3518",
    "LUPIN": "10440", "ALKEM": "11703", "LAURUSLABS": "19234",
    "GLENMARK": "7406",
    # Metal & Mining
    "VEDL": "3063", "NATIONALUM": "6364", "NMDC": "15332",
    "SAIL": "2963", "JINDALSTEL": "6733",
    # Energy
    "GAIL": "4717", "PETRONET": "11351", "IGL": "11262",
    "MGL": "17534", "TATAPOWER": "3426", "NHPC": "13397",
    # FMCG
    "DABUR": "772", "GODREJCP": "10099", "MARICO": "4067",
    "COLPAL": "15141", "TATACONSUM": "3432", "UBL": "16713",
    "PIIND": "24797", "UPL": "11287",
    # Chemicals & Materials
    "PIDILITIND": "2664", "SRF": "3273", "DEEPAKNTR": "19943",
    "AARTIIND": "7",
    # Cement & Construction
    "SHREECEM": "3103", "AMBUJACEM": "1270", "ACC": "13880",
    "DALBHARAT": "8075", "JKCEMENT": "13270",
    # Capital Goods & Infrastructure
    "ABB": "13", "SIEMENS": "3150", "HAVELLS": "9819",
    "VOLTAS": "3718", "CUMMINSIND": "2475", "BEL": "383",
    "HAL": "2303", "BHEL": "438",
    # Real Estate
    "DLF": "14732", "GODREJPROP": "17875", "OBEROIRLTY": "20242",
    "PRESTIGE": "14413", "LODHA": "28794",
    # Financial Services  
    "ANGELONE": "15474", "CDSL": "30462", "MCX": "31181",
    "IRCTC": "13611",
    # Consumer Tech
    "DIXON": "21690", "POLYCAB": "9590", "PAGEIND": "14413",
    "ZOMATO": "5097", "DELHIVERY": "541",
    "LICI": "6400",
    # Telecom
    "IDEA": "14366",
    # Others
    "INDUSTOWER": "29135", "IOC": "1624",
    "BERGEPAINT": "404", "CONCOR": "4749",
    "ABCAPITAL": "21614", "ESCORTS": "958",
    "TATAELXSI": "3445", "TATACOMM": "3425",
    "HINDPETRO": "1406", "CHAMBLFERT": "637", "GNFC": "7287",
    "INDHOTEL": "1512", "IEX": "12639",
    "CESC": "628",
}

# Index tokens for NFO
INDEX_TOKENS = {
    "NIFTY": "99926000",
    "BANKNIFTY": "99926009",
    "FINNIFTY": "99926037",
    "MIDCPNIFTY": "99926074",
}

# ── MCX Commodity Tokens ──────────────────────────────────────────────────────
# These are exchange-level symbol tokens for MCX commodity futures.
# Angel One uses "MCX" as the exchange key for commodity data.
# Token IDs correspond to near-month (active) futures contracts.
# NOTE: These token IDs may change on contract expiry — update monthly.
MCX_COMMODITY_TOKENS = {
    # Energy
    "NATURALGAS": "504265",
    "CRUDEOIL":   "499095",
    # Precious Metals
    "GOLD":       "454818",
    "GOLDM":      "477904",       # Gold Mini
    "GOLDGUINEA": "488785",       # Gold Guinea
    "SILVER":     "464150",
    "SILVERM":    "457533",       # Silver Mini
    # Base Metals
    "COPPER":     "510480",
    "ZINC":       "510478",
    "LEAD":       "510476",
    "NICKEL":     "488796",
    "ALUMINIUM":  "510472",
    # Agri
    "COTTON":     "510483",
    "MENTHAOIL":  "488802",
}

# Commodity sector classification for dashboard display
COMMODITY_SECTORS = {
    "Energy":          ["NATURALGAS", "CRUDEOIL"],
    "Precious Metals": ["GOLD", "GOLDM", "GOLDGUINEA", "SILVER", "SILVERM"],
    "Base Metals":     ["COPPER", "ZINC", "LEAD", "NICKEL", "ALUMINIUM"],
    "Agriculture":     ["COTTON", "MENTHAOIL"],
}

def is_commodity_market_hours():
    """MCX commodity market hours: Mon-Fri 9:00-23:30 (normal) / 23:55 (agri)."""
    now = datetime.now()
    if now.weekday() >= 5:       # Sat-Sun closed
        return False
    t = now.strftime("%H:%M")
    return "09:00" <= t <= "23:30"


class AngelOneDataFetcher:
    """Fetches market data using Angel One SmartAPI — reliable, no scraping."""

    def __init__(self, auth: AngelOneAuth):
        self.auth = auth
        self.smart_api: Optional[SmartConnect] = None

    def _get_api(self) -> Optional[SmartConnect]:
        if self.auth.smart_api:
            return self.auth.smart_api
        return None

    def fetch_all(self) -> Dict[str, Dict]:
        """Fetch market data for ALL F&O stocks using Angel One API."""
        api = self._get_api()
        if not api:
            logger.error("Angel One not authenticated — cannot fetch market data")
            return {}

        all_data = {}

        # Split tokens into batches of 50 (API limit)
        token_list = list(FNO_TOKENS.items())
        batches = [token_list[i:i+50] for i in range(0, len(token_list), 50)]

        for batch_num, batch in enumerate(batches, 1):
            try:
                tokens = [tok for _, tok in batch]
                symbols = [sym for sym, _ in batch]

                mode = "FULL"
                exchange_tokens = {"NSE": tokens}

                data = api.getMarketData(mode, exchange_tokens)

                if data and data.get("status"):
                    fetched = data.get("data", {}).get("fetched", [])
                    unfetched = data.get("data", {}).get("unfetched", [])

                    for item in fetched:
                        token = str(item.get("symbolToken", ""))
                        sym = None
                        for s, t in batch:
                            if t == token:
                                sym = s
                                break
                        if not sym:
                            continue

                        ltp = float(item.get("ltp", 0))
                        if ltp <= 0:
                            continue

                        open_p = float(item.get("open", ltp))
                        high = float(item.get("high", ltp))
                        low = float(item.get("low", ltp))
                        close = float(item.get("close", ltp))
                        volume = int(item.get("tradeVolume", 0))
                        # Angel One API: 'close' field = previous day's close
                        prev_close = close

                        # Extract bid/ask info if available in FULL mode
                        bid_price = float(item.get("best_bid_price", ltp))
                        ask_price = float(item.get("best_ask_price", ltp))
                        bid_size = int(item.get("best_bid_qty", 0))
                        ask_size = int(item.get("best_ask_qty", 0))

                        chg_pct = ((ltp - open_p) / open_p * 100) if open_p > 0 else 0
                        chg_from_prev = ((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0

                        all_data[sym] = {
                            "symbol": sym,
                            "ltp": ltp,
                            "open": open_p,
                            "high": high,
                            "low": low,
                            "prev_close": prev_close,
                            "change_pct": round(chg_pct, 2),
                            "change_from_prev": round(chg_from_prev, 2),
                            "volume": volume,
                            "bid_price": bid_price,
                            "ask_price": ask_price,
                            "bid_size": bid_size,
                            "ask_size": ask_size,
                            "instrument_type": "EQ",
                            "source": "AngelOne",
                            "ts": datetime.now().isoformat(),
                        }

                    if unfetched:
                        logger.debug(f"Batch {batch_num}: {len(unfetched)} unfetched tokens")
                else:
                    error_msg = data.get("message", "Unknown error") if data else "No response"
                    logger.warning(f"Batch {batch_num} failed: {error_msg}")

                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"Batch {batch_num} error: {e}")
                continue

        # Fetch indices
        try:
            idx_tokens = list(INDEX_TOKENS.values())
            idx_data = api.getMarketData("FULL", {"NSE": idx_tokens})
            if idx_data and idx_data.get("status"):
                for item in idx_data.get("data", {}).get("fetched", []):
                    token = str(item.get("symbolToken", ""))
                    sym = None
                    for s, t in INDEX_TOKENS.items():
                        if t == token:
                            sym = s
                            break
                    if not sym:
                        continue
                    ltp = float(item.get("ltp", 0))
                    if ltp <= 0:
                        continue
                    all_data[sym] = {
                        "symbol": sym,
                        "ltp": ltp,
                        "open": float(item.get("open", ltp)),
                        "high": float(item.get("high", ltp)),
                        "low": float(item.get("low", ltp)),
                        "prev_close": float(item.get("close", ltp)),
                        "change_pct": round(((ltp - float(item.get("open", ltp))) / float(item.get("open", ltp)) * 100) if float(item.get("open", ltp)) > 0 else 0, 2),
                        "volume": int(item.get("tradeVolume", 0)),
                        "instrument_type": "INDEX",
                        "source": "AngelOne",
                        "ts": datetime.now().isoformat(),
                    }
        except Exception as e:
            logger.debug(f"Index fetch error: {e}")

        logger.info(f"Fetched {len(all_data)} stocks via Angel One API")
        return all_data

    def fetch_commodities(self) -> Dict[str, Dict]:
        """Fetch MCX commodity futures data using Angel One API."""
        api = self._get_api()
        if not api:
            return {}

        commodity_data = {}
        token_list = list(MCX_COMMODITY_TOKENS.items())

        try:
            tokens = [tok for _, tok in token_list]
            data = api.getMarketData("FULL", {"MCX": tokens})

            if data and data.get("status"):
                for item in data.get("data", {}).get("fetched", []):
                    token = str(item.get("symbolToken", ""))
                    sym = None
                    for s, t in token_list:
                        if t == token:
                            sym = s
                            break
                    if not sym:
                        continue

                    ltp = float(item.get("ltp", 0))
                    if ltp <= 0:
                        continue

                    open_p = float(item.get("open", ltp))
                    high = float(item.get("high", ltp))
                    low = float(item.get("low", ltp))
                    close = float(item.get("close", ltp))
                    volume = int(item.get("tradeVolume", 0))
                    prev_close = close

                    chg_pct = ((ltp - open_p) / open_p * 100) if open_p > 0 else 0

                    # Determine commodity sector
                    sector = "Other"
                    for sec_name, sec_syms in COMMODITY_SECTORS.items():
                        if sym in sec_syms:
                            sector = sec_name
                            break

                    commodity_data[sym] = {
                        "symbol": sym,
                        "ltp": ltp,
                        "open": open_p,
                        "high": high,
                        "low": low,
                        "prev_close": prev_close,
                        "change_pct": round(chg_pct, 2),
                        "volume": volume,
                        "instrument_type": "COMMODITY",
                        "commodity_sector": sector,
                        "source": "AngelOne_MCX",
                        "ts": datetime.now().isoformat(),
                    }
            else:
                msg = data.get("message", "Unknown") if data else "No response"
                logger.warning(f"MCX fetch failed: {msg}")
        except Exception as e:
            logger.warning(f"Commodity fetch error: {e}")

        if commodity_data:
            logger.info(f"📦 Fetched {len(commodity_data)} MCX commodities")
        return commodity_data


# ── Commodity Analyzer ────────────────────────────────────────────────────────

class CommodityAnalyzer:
    """
    Analyzes MCX commodity data and generates trading signals.
    Uses momentum, RSI estimation, volatility, and trend detection.
    """

    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        self.max_history = 200

    def update_history(self, sym: str, ltp: float):
        """Track price history for trend analysis."""
        if sym not in self.price_history:
            self.price_history[sym] = []
        self.price_history[sym].append(ltp)
        if len(self.price_history[sym]) > self.max_history:
            self.price_history[sym] = self.price_history[sym][-self.max_history:]

    @staticmethod
    def _estimate_rsi(data: Dict) -> float:
        """Estimate RSI from today's OHLC data."""
        ltp = data.get("ltp", 0)
        hi = data.get("high", ltp)
        lo = data.get("low", ltp)
        if hi <= lo:
            return 50.0
        return round((ltp - lo) / (hi - lo) * 100, 1)

    def analyze(self, commodity_data: Dict[str, Dict]) -> List[Dict]:
        """Analyze all commodities and return scored signals."""
        signals = []

        for sym, data in commodity_data.items():
            ltp = data.get("ltp", 0)
            op = data.get("open", ltp)
            hi = data.get("high", ltp)
            lo = data.get("low", ltp)
            prev = data.get("prev_close", op)
            vol = data.get("volume", 0)

            if ltp <= 0 or op <= 0:
                continue

            self.update_history(sym, ltp)

            chg_pct = data.get("change_pct", 0)
            rsi = self._estimate_rsi(data)
            day_range = ((hi - lo) / op * 100) if op > 0 else 0

            # Calculate trend from history
            history = self.price_history.get(sym, [])
            trend = "neutral"
            trend_strength = 0.0
            if len(history) >= 5:
                recent_avg = sum(history[-3:]) / 3
                past_avg = sum(history[-5:-2]) / 3 if len(history) >= 5 else recent_avg
                if past_avg > 0:
                    trend_strength = (recent_avg - past_avg) / past_avg * 100
                    if trend_strength > 0.3:
                        trend = "bullish"
                    elif trend_strength < -0.3:
                        trend = "bearish"

            # ── Scoring ──
            score = 0
            strategies = []
            reasons = []
            direction = "BUY"

            # 1. Momentum signal
            if abs(chg_pct) > 1.0:
                mom_score = min(25, abs(chg_pct) * 5)
                score += mom_score
                strategies.append("Momentum")
                direction = "BUY" if chg_pct > 0 else "SELL"
                reasons.append(f"Strong move {chg_pct:+.1f}%")

            # 2. RSI signal
            if rsi < 25:
                score += 15
                strategies.append("Oversold")
                direction = "BUY"
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 75:
                score += 15
                strategies.append("Overbought")
                direction = "SELL"
                reasons.append(f"RSI overbought ({rsi:.0f})")

            # 3. Trend alignment
            if (trend == "bullish" and direction == "BUY") or \
               (trend == "bearish" and direction == "SELL"):
                score += 10
                strategies.append("Trend Aligned")
                reasons.append(f"Trend: {trend} ({trend_strength:+.2f}%)")

            # 4. Volatility play
            if day_range > 2.0:
                score += 8
                strategies.append("High Volatility")
                reasons.append(f"Day range {day_range:.1f}%")

            # 5. Gap signal
            if prev > 0:
                gap = (op - prev) / prev * 100
                if abs(gap) > 1.0:
                    score += 10
                    strategies.append("Gap" if gap > 0 else "Gap Down")
                    reasons.append(f"Gap {gap:+.1f}%")

            if score < 15:
                continue

            # Targets
            atr_pct = max(0.005, day_range / 100)
            if direction == "BUY":
                t1 = round(ltp * (1 + atr_pct * 1.5), 2)
                sl = round(ltp * (1 - atr_pct), 2)
            else:
                t1 = round(ltp * (1 - atr_pct * 1.5), 2)
                sl = round(ltp * (1 + atr_pct), 2)

            rr = abs(t1 - ltp) / abs(sl - ltp) if abs(sl - ltp) > 0 else 0

            signals.append({
                "symbol": sym,
                "direction": direction,
                "score": round(score, 1),
                "ltp": ltp,
                "open": op,
                "high": hi,
                "low": lo,
                "change_pct": chg_pct,
                "volume": vol,
                "rsi": rsi,
                "trend": trend,
                "trend_strength": round(trend_strength, 2),
                "day_range_pct": round(day_range, 2),
                "target_1": t1,
                "stop_loss": sl,
                "risk_reward": round(rr, 2),
                "strategies": strategies,
                "num_strategies": len(strategies),
                "reasons": reasons,
                "instrument_type": "COMMODITY",
                "commodity_sector": data.get("commodity_sector", "Other"),
                "qty": 1,  # Lot-based for MCX
                "confidence": min(1.0, score / 80),
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals


# ── Intraday Strategies ───────────────────────────────────────────────────────

class StrategyEngine:
    """Smart intraday strategies with exhaustion detection and signal dedup."""

    # Strategy groups — correlated strategies get discounted when combining
    CORRELATED_GROUPS = {
        "momentum": {"Momentum", "Volume Surge", "Prev Day Move"},
        "position": {"Breakout", "Breakdown", "VWAP Signal"},
        "reversal": {"Reversal Buy", "Reversal Sell", "Gap Opening"},
    }

    def __init__(self, budget: float = 25000, adaptive_engine: AdaptiveStrategyEngine = None):
        self.budget = budget
        self.adaptive_engine = adaptive_engine

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rsi(q: Dict) -> float:
        """Estimate RSI from OHLC: measures where close is in the day range."""
        ltp = q.get("ltp", 0)
        hi = q.get("high", ltp)
        lo = q.get("low", ltp)
        op = q.get("open", ltp)
        prev = q.get("prev_close", op)
        if hi <= lo:
            return 50.0
        # Combine two views: position in today's range + vs prev close
        pos_in_range = (ltp - lo) / (hi - lo)  # 0 = at low, 1 = at high
        chg = (ltp - prev) / prev if prev > 0 else 0
        # Map to 0-100 RSI-like scale
        rsi = pos_in_range * 60 + 20  # base 20-80
        if chg > 0.02:
            rsi += 10
        elif chg < -0.02:
            rsi -= 10
        return max(5.0, min(95.0, rsi))

    @staticmethod
    def _move_from_open(q: Dict) -> float:
        """Pct move from open."""
        ltp, op = q.get("ltp", 0), q.get("open", 0)
        return ((ltp - op) / op * 100) if op > 0 else 0

    @staticmethod
    def _is_exhausted(q: Dict, direction: str, threshold: float = 3.0) -> bool:
        """Check if the move is already exhausted (too extended to chase)."""
        ltp = q.get("ltp", 0)
        op = q.get("open", 0)
        hi = q.get("high", ltp)
        lo = q.get("low", ltp)
        if op <= 0:
            return False
        mv = (ltp - op) / op * 100
        day_range = (hi - lo) / op * 100

        if direction == "BUY":
            # Exhausted if already up > threshold% AND near day high (>85% of range)
            if mv > threshold and hi > lo and (ltp - lo) / (hi - lo) > 0.85:
                return True
        elif direction == "SELL":
            # Exhausted if already down > threshold% AND near day low (<15% of range)
            if mv < -threshold and hi > lo and (ltp - lo) / (hi - lo) < 0.15:
                return True

        # Also exhausted if day range is very extended already
        if day_range > 5.0 and abs(mv) > 4.0:
            return True
        return False

    def _deduplicated_score(self, agreeing: List[Dict]) -> float:
        """Compute combined score with discount for correlated strategies."""
        if not agreeing:
            return 0
        # Find which groups are represented
        seen_groups: Set[str] = set()
        adjusted_scores = []

        for sig in agreeing:
            strat = sig.get("strategy", "")
            score_key = "adjusted_score" if "adjusted_score" in sig else "score"
            base = sig.get(score_key, sig.get("score", 0))

            # Check if this strategy correlates with one already counted
            in_group = None
            for grp, members in self.CORRELATED_GROUPS.items():
                if strat in members:
                    in_group = grp
                    break

            if in_group and in_group in seen_groups:
                # Discount correlated signals — they add only 40% of their score
                adjusted_scores.append(base * 0.40)
            else:
                adjusted_scores.append(base)
                if in_group:
                    seen_groups.add(in_group)

        return min(100, sum(adjusted_scores))

    # ── Main Analyzer ────────────────────────────────────────────────────────

    def analyze(self, q: Dict, market_context: Dict = None) -> List[Dict]:
        """Analyze stock data and return scored signals with exhaustion checks."""
        raw_signals = []
        rsi = self._compute_rsi(q)

        for fn in [
            self._gap, self._momentum, self._reversal, self._volume_surge,
            self._breakout, self._vwap, self._prev_day, self._narrow_range,
        ]:
            try:
                s = fn(q)
                if s:
                    # ── Exhaustion filter: reject signals chasing extended moves ──
                    d = s["direction"]
                    if self._is_exhausted(q, d):
                        logger.debug(f"⛔ {q.get('symbol','?')}: {s['strategy']} rejected — move exhausted")
                        continue

                    # ── RSI filter: don't BUY when overbought or SELL when oversold ──
                    if d == "BUY" and rsi > 80:
                        s["score"] *= 0.4
                        s["reason"] += " [WARN: overbought RSI]" 
                    elif d == "SELL" and rsi < 20:
                        s["score"] *= 0.4
                        s["reason"] += " [WARN: oversold RSI]"

                    # ── Time-of-day decay: afternoon signals are weaker ──
                    hour = datetime.now().hour
                    if hour >= 14:
                        s["score"] *= 0.7
                    elif hour >= 13:
                        s["score"] *= 0.85

                    s["rsi"] = round(rsi, 1)
                    s["score"] = round(s["score"], 1)
                    raw_signals.append(s)
            except Exception as e:
                logger.warning(f"Strategy evaluation failed for {q.get('symbol', '?')}: {e}")

        if self.adaptive_engine:
            try:
                ai_sigs = self.adaptive_engine.get_ai_signals(q.get("symbol", ""), q, market_context)
                if ai_sigs:
                    raw_signals.extend(ai_sigs)
            except Exception as e:
                logger.debug(f"AI Strategy execution error: {e}")

        # Apply learned adaptive weights if available
        if self.adaptive_engine and raw_signals:
            raw_signals = self.adaptive_engine.enhance_signal(
                raw_signals, q, market_context
            )

        return raw_signals

    # ── Individual Strategies (with improved filters) ──

    def _gap(self, q):
        ltp, prev, op = q.get("ltp", 0), q.get("prev_close", 0), q.get("open", 0)
        if prev <= 0 or op <= 0:
            return None
        gap = (op - prev) / prev * 100
        if abs(gap) < 1.0:
            return None
        d = "BUY" if gap > 0 else "SELL"
        sc = min(25, abs(gap) * 5)  # Reduced cap from 35 to 25

        # Gap fill check: if price is moving AGAINST the gap, reduce score a lot
        if d == "BUY" and ltp < op * 0.995:
            sc *= 0.3  # Gap is filling! Strong penalty
        elif d == "SELL" and ltp > op * 1.005:
            sc *= 0.3
        # Partial fill penalty
        elif d == "BUY" and ltp < op * 0.999:
            sc *= 0.6
        elif d == "SELL" and ltp > op * 1.001:
            sc *= 0.6

        return {"strategy": "Gap Opening", "direction": d, "score": round(sc, 1),
                "reason": f"Gap {'up' if gap > 0 else 'down'} {abs(gap):.1f}%"}

    def _momentum(self, q):
        ltp, op, vol = q.get("ltp", 0), q.get("open", 0), q.get("volume", 0)
        if op <= 0:
            return None
        mv = (ltp - op) / op * 100
        if abs(mv) < 1.5:
            return None

        # ── NEW: Cap momentum score for very extended moves (diminishing returns)
        d = "BUY" if mv > 0 else "SELL"
        if abs(mv) <= 3.0:
            sc = min(25, abs(mv) * 5)
        else:
            # Diminishing returns after 3% — likely chasing
            sc = 15 + min(10, (abs(mv) - 3.0) * 2)

        if vol > 2_000_000:
            sc += 8
        elif vol > 500_000:
            sc += 4

        return {"strategy": "Momentum", "direction": d, "score": round(sc, 1),
                "reason": f"{mv:+.1f}% from open, vol {vol:,}"}

    def _reversal(self, q):
        ltp, hi, lo, op = q.get("ltp", 0), q.get("high", 0), q.get("low", 0), q.get("open", 0)
        if hi <= lo or op <= 0:
            return None
        rng = hi - lo
        rng_pct = rng / op * 100
        if rng_pct < 2.0:
            return None

        # Improved: only signal reversals if price has bounced from extreme
        pos = (ltp - lo) / rng  # 0=low, 1=high

        if pos < 0.15:
            # Near low — but only BUY if there's a bounce wick (low is not LTP)
            wick_ratio = (ltp - lo) / rng if rng > 0 else 0
            sc = min(25, rng_pct * 4)  # Reduced max
            if wick_ratio < 0.05:  # Literally at the bottom, no bounce yet
                sc *= 0.5  # Don't catch falling knife
            return {"strategy": "Reversal Buy", "direction": "BUY",
                    "score": round(sc, 1),
                    "reason": f"Near day low ({pos*100:.0f}%), range {rng_pct:.1f}%"}
        if pos > 0.85:
            sc = min(25, rng_pct * 4)
            return {"strategy": "Reversal Sell", "direction": "SELL",
                    "score": round(sc, 1),
                    "reason": f"Near day high ({pos*100:.0f}%), range {rng_pct:.1f}%"}
        return None

    def _volume_surge(self, q):
        vol, ltp, op = q.get("volume", 0), q.get("ltp", 0), q.get("open", 0)
        if op <= 0 or vol < 500_000:
            return None
        mv = (ltp - op) / op * 100
        if abs(mv) < 0.5:
            return None
        d = "BUY" if mv > 0 else "SELL"

        # ── Volume as independent confirmation only ──
        if vol > 5_000_000:
            sc = 18
        elif vol > 2_000_000:
            sc = 12
        else:
            sc = 6

        # Small directional boost, but volume alone isn't a direction indicator
        sc += min(6, abs(mv) * 2)

        return {"strategy": "Volume Surge", "direction": d, "score": round(sc, 1),
                "reason": f"Vol {vol/1e6:.1f}M, {mv:+.1f}%"}

    def _breakout(self, q):
        ltp, hi, lo, op = q.get("ltp", 0), q.get("high", 0), q.get("low", 0), q.get("open", 0)
        if hi <= lo or op <= 0:
            return None
        rng_pct = (hi - lo) / op * 100

        # ── Tighter filter: need meaningful range but not overcrowded ──
        if rng_pct < 1.0 or rng_pct > 6.0:
            return None  # Too tight or too extended

        # Only trigger if volume supports the move
        vol = q.get("volume", 0)
        if vol < 200_000:
            return None

        sc = round(min(22, rng_pct * 5), 1)  # Reduced max

        if ltp >= hi * 0.999:
            return {"strategy": "Breakout", "direction": "BUY",
                    "score": sc,
                    "reason": f"At day high, range {rng_pct:.1f}%, vol {vol:,}"}
        if ltp <= lo * 1.001:
            return {"strategy": "Breakdown", "direction": "SELL",
                    "score": sc,
                    "reason": f"At day low, range {rng_pct:.1f}%, vol {vol:,}"}
        return None

    def _vwap(self, q):
        ltp, hi, lo, vol = q.get("ltp", 0), q.get("high", 0), q.get("low", 0), q.get("volume", 0)
        if hi <= 0:
            return None
        op = q.get("open", ltp)
        # Better VWAP proxy: volume-weighted typical price approximation
        vwap = (hi + lo + ltp + op) / 4  # 4-point average (better approx)
        dev = (ltp - vwap) / vwap * 100
        if abs(dev) < 1.0:  # Increased threshold from 0.8%
            return None

        d = "BUY" if dev > 0 else "SELL"
        sc = min(18, abs(dev) * 4)  # Reduced cap from 25

        # VWAP is meaningful only with volume
        if vol < 300_000:
            sc *= 0.6

        return {"strategy": "VWAP Signal", "direction": d, "score": round(sc, 1),
                "reason": f"LTP {dev:+.1f}% from VWAP (approx)"}

    def _prev_day(self, q):
        ltp, prev = q.get("ltp", 0), q.get("prev_close", 0)
        if prev <= 0:
            return None
        chg = (ltp - prev) / prev * 100
        if abs(chg) < 2.0:  # Raised threshold from 1.5% — need stronger signal
            return None
        d = "BUY" if chg > 0 else "SELL"

        # ── Diminishing returns for huge prev day moves (likely priced in) ──
        if abs(chg) <= 4.0:
            sc = min(20, abs(chg) * 4)
        else:
            sc = 16 + min(6, (abs(chg) - 4.0) * 1.5)

        return {"strategy": "Prev Day Move", "direction": d,
                "score": round(sc, 1),
                "reason": f"{chg:+.1f}% from prev close"}

    def _narrow_range(self, q):
        ltp, hi, lo, op, vol = q.get("ltp", 0), q.get("high", 0), q.get("low", 0), q.get("open", 0), q.get("volume", 0)
        if hi <= lo or op <= 0:
            return None
        rng_pct = (hi - lo) / op * 100
        if rng_pct > 0.8 or rng_pct < 0.05 or vol < 100_000:
            return None
        d = "BUY" if ltp >= op else "SELL"
        hour = datetime.now().hour
        bonus = 12 if hour < 11 else 6 if hour < 13 else 2
        return {"strategy": "Narrow Range", "direction": d, "score": round(12 + bonus, 1),
                "reason": f"Tight {rng_pct:.2f}% range, vol {vol:,}"}


# ── Market Scanner ────────────────────────────────────────────────────────────

class MarketScanner:
    """Full market scanner using Angel One SmartAPI + adaptive strategies."""

    def __init__(self, config: Dict, adaptive_engine: AdaptiveStrategyEngine = None):
        self.cfg = config
        self.budget = config.get("trading", {}).get("budget", 25000)
        self.auth = None
        self.fetcher: Optional[AngelOneDataFetcher] = None
        self.strategy = StrategyEngine(self.budget, adaptive_engine)
        self.adaptive_engine = adaptive_engine
        self.fno_stocks: List[str] = list(FNO_TOKENS.keys())
        self.last_raw: Dict[str, Dict] = {}

    def set_auth(self, auth: AngelOneAuth):
        self.auth = auth
        self.fetcher = AngelOneDataFetcher(auth)

    def scan(self) -> Dict[str, Dict]:
        if not self.fetcher:
            logger.error("No data fetcher configured")
            return {}
        all_data = self.fetcher.fetch_all()
        self.last_raw = all_data
        return all_data

    def opportunities(self, min_score: float = 30,
                      market_context: Dict = None) -> List[Dict]:
        """Scan for equity intraday opportunities with smart scoring."""
        raw = self.scan()
        ctx = market_context or {}
        all_opps = []

        # Use learned min_score if available
        if self.adaptive_engine:
            min_score = self.adaptive_engine.params.min_score

        for sym, data in raw.items():
            if data.get("instrument_type") == "INDEX":
                continue
            
            # Skip if basic prices are zero or missing (e.g. illiquid stocks)
            ltp = data.get("ltp", 0)
            op = data.get("open", 0)
            hi = data.get("high", 0)
            lo = data.get("low", 0)
            
            if ltp <= 0 or op <= 0 or hi <= 0 or lo <= 0:
                continue

            # Update Feature Engineer for AI strategies
            if self.adaptive_engine and self.adaptive_engine.fe:
                from trading_bot.core.models import MarketTick, OHLC, Exchange
                tick = MarketTick(
                    symbol=sym,
                    exchange=Exchange.NSE,
                    timestamp=datetime.now(),
                    price=Decimal(str(data["ltp"])),
                    volume=data.get("volume", 0),
                    bid_price=Decimal(str(data.get("bid_price", data["ltp"]))),
                    ask_price=Decimal(str(data.get("ask_price", data["ltp"]))),
                    bid_size=data.get("bid_size", 0),
                    ask_size=data.get("ask_size", 0)
                )
                self.adaptive_engine.fe.update_market_data(sym, tick)
                
                # Sanitize high/low for OHLC validation
                s_ltp = Decimal(str(ltp))
                s_op = Decimal(str(op))
                s_hi = Decimal(str(max(hi, ltp, op)))
                s_lo = Decimal(str(min(lo, ltp, op)))
                
                # Update OHLC
                ohlc = OHLC(
                    timestamp=datetime.now(),
                    open_price=s_op,
                    high_price=s_hi,
                    low_price=s_lo,
                    close_price=s_ltp,
                    volume=data.get("volume", 0)
                )
                self.adaptive_engine.fe.update_ohlc_data(sym, ohlc)

            signals = self.strategy.analyze(data, ctx)
            if not signals:
                continue

            # Use adjusted_score if available (from adaptive engine)
            score_key = "adjusted_score" if "adjusted_score" in (signals[0] if signals else {}) else "score"
            best = max(signals, key=lambda s: s.get(score_key, s.get("score", 0)))
            agreeing = [s for s in signals if s["direction"] == best["direction"]]

            # ── NEW: Use deduplicated scoring to prevent correlated inflation ──
            combined = self.strategy._deduplicated_score(agreeing)

            if combined < min_score:
                continue

            ltp = data["ltp"]
            direction = best["direction"]

            # ── NEW: Compute actual RSI for the opportunity record ──
            rsi = StrategyEngine._compute_rsi(data)

            # Adaptive position sizing
            if self.adaptive_engine:
                targets = self.adaptive_engine.get_dynamic_targets(
                    ltp, direction, data, ctx
                )
                sl = targets["stop_loss"]
                t1 = targets["target_1"]
                t2 = targets["target_2"]
                
                confidence = self.adaptive_engine._compute_confidence(
                    {"score": combined, "num_strategies": len(agreeing),
                     "risk_reward": abs(t1 - ltp) / abs(sl - ltp) if abs(sl - ltp) > 0 else 1,
                     "volume": data.get("volume", 0), "strategies": [s["strategy"] for s in agreeing],
                     "rsi": rsi},
                    ctx
                )
                qty = self.adaptive_engine.get_position_size(ltp, sl, self.budget, confidence)
            else:
                qty = max(1, int(self.budget * 0.15 / ltp))
                if direction == "BUY":
                    t1, t2, sl = ltp * 1.015, ltp * 1.030, ltp * 0.990
                else:
                    t1, t2, sl = ltp * 0.985, ltp * 0.970, ltp * 1.010
                confidence = 0.5

            margin = ltp * qty
            if margin > self.budget * 0.5:
                qty = max(1, int(self.budget * 0.5 / ltp))
                margin = ltp * qty

            profit = abs(t2 - ltp) * qty
            loss = abs(ltp - sl) * qty
            rr = round(profit / loss, 2) if loss > 0 else 0

            # ── NEW: Minimum risk:reward filter ──
            if rr < 1.2:
                continue  # Not worth the risk

            all_opps.append({
                "symbol": sym,
                "instrument_type": data.get("instrument_type", "EQ"),
                "ltp": round(ltp, 2),
                "open": round(data.get("open", ltp), 2),
                "high": round(data.get("high", ltp), 2),
                "low": round(data.get("low", ltp), 2),
                "prev_close": round(data.get("prev_close", ltp), 2),
                "direction": direction,
                "change_pct": data.get("change_pct", 0),
                "volume": data.get("volume", 0),
                "qty": qty,
                "margin": round(margin, 2),
                "target_1": round(t1, 2),
                "target_2": round(t2, 2),
                "stop_loss": round(sl, 2),
                "max_profit": round(profit, 2),
                "max_loss": round(loss, 2),
                "risk_reward": rr,
                "score": round(combined, 1),
                "confidence": round(confidence, 3),
                "rsi": round(rsi, 1),
                "priority": "HIGH" if combined >= 55 else "MEDIUM" if combined >= 40 else "LOW",
                "strategies": [s["strategy"] for s in agreeing],
                "num_strategies": len(agreeing),
                "reasons": [s["reason"] for s in agreeing],
                "source": "AngelOne",
                "ts": data.get("ts", ""),
            })

        all_opps.sort(key=lambda x: x["score"], reverse=True)
        return all_opps


# ── Trade & Profit Tracker (with Learning) ────────────────────────────────────

class TradeManager:
    """Tracks trades, calculates P&L, and feeds the learning system."""

    MAX_TRADES_PER_DAY = 15  # Hard limit on entries per day
    MAX_CONSECUTIVE_LOSSES = 5  # Disable trading after N straight losses

    def __init__(self, budget: float, journal: TradeJournal = None,
                 adaptive_engine: AdaptiveStrategyEngine = None):
        self.filename = "trades.json"
        self._lock = threading.Lock()  # Protect file reads/writes
        self.trades: List[Dict] = self._load_trades()
        self.budget = budget
        self.journal = journal
        self.adaptive_engine = adaptive_engine
        
        # Tracking
        self._price_highs: Dict[str, float] = {}
        self._price_lows: Dict[str, float] = {}
        self._trading_disabled = False  # Set by circuit breakers
        
        # On restart: check if circuit breakers should be active
        self._restore_circuit_breaker_state()

    def _load_trades(self) -> List[Dict]:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    logger.error(f"trades.json is not a list, got {type(data)}")
                    return []
                return data
            except json.JSONDecodeError as e:
                logger.critical(f"CORRUPT trades.json — cannot parse: {e}")
                # Preserve the corrupt file for investigation
                backup = f"{self.filename}.corrupt.{int(time.time())}"
                try:
                    os.rename(self.filename, backup)
                    logger.critical(f"Corrupt file backed up to {backup}")
                except OSError:
                    pass
                return []
            except (IOError, OSError) as e:
                logger.error(f"Cannot read trades.json: {e}")
                return []
        return []

    def _save_trades(self):
        """Atomic save: write to temp file then rename to prevent corruption."""
        with self._lock:
            try:
                dir_name = os.path.dirname(os.path.abspath(self.filename)) or "."
                fd, tmp_path = tempfile.mkstemp(
                    suffix=".tmp", prefix="trades_", dir=dir_name
                )
                with os.fdopen(fd, 'w') as f:
                    json.dump(self.trades, f, indent=2)
                # Atomic rename (on Windows, need to remove target first)
                if os.path.exists(self.filename):
                    os.replace(tmp_path, self.filename)
                else:
                    os.rename(tmp_path, self.filename)
            except Exception as e:
                logger.critical(f"FAILED to save trades: {e}")
                # Try to clean up temp file
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass

    def _restore_circuit_breaker_state(self):
        """On restart, check if circuit breakers should still be active today."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_closed = [
            t for t in self.trades
            if (t.get("exit_time") or "").startswith(today) and t["status"] != "OPEN"
        ]
        # Check consecutive losses
        if len(today_closed) >= self.MAX_CONSECUTIVE_LOSSES:
            recent = today_closed[-self.MAX_CONSECUTIVE_LOSSES:]
            if all(t.get("pnl", 0) < 0 for t in recent):
                self._trading_disabled = True
                logger.warning("🛡️ Restarted with consecutive loss breaker ACTIVE")
        # Check daily loss limit
        daily_pnl = sum(t.get("pnl", 0) for t in self.trades
                        if (t.get("entry_time") or "").startswith(today))
        max_daily_loss = self.budget * 0.05
        if daily_pnl < -max_daily_loss:
            self._trading_disabled = True
            logger.warning(f"🛡️ Restarted with daily loss lock ACTIVE (P&L: ₹{daily_pnl:.2f})")
        
        open_count = sum(1 for t in self.trades if t["status"] == "OPEN")
        if open_count > 0:
            logger.info(f"🔄 Recovered {open_count} OPEN positions from previous session")

    def track_opportunity(self, opp: Dict, market_context: Dict = None):
        """Record an opportunity for P&L tracking (Paper Trading) + Journal."""
        # ── Circuit breaker: trading disabled by loss streak ──
        if self._trading_disabled:
            logger.warning(f"Trading disabled by circuit breaker — rejecting {opp.get('symbol', '?')}")
            return None
        
        # ── Max trades per day ──
        today = datetime.now().strftime("%Y-%m-%d")
        today_entries = sum(
            1 for t in self.trades
            if (t.get("entry_time") or "").startswith(today)
        )
        if today_entries >= self.MAX_TRADES_PER_DAY:
            logger.info(f"Max trades per day ({self.MAX_TRADES_PER_DAY}) reached — rejecting {opp.get('symbol', '?')}")
            return None
        
        # ── Volatile regime hard block ──
        ctx = market_context or {}
        if ctx.get("regime") == "volatile" and ctx.get("volatility", 0) > 3.0:
            logger.info(f"High volatility ({ctx.get('volatility', 0):.1f}%) — rejecting {opp.get('symbol', '?')}")
            return None
        
        # Avoid duplicate: reject if symbol has ANY open trade
        for t in self.trades:
            if t["symbol"] == opp["symbol"] and t["status"] == "OPEN":
                logger.debug(f"Skipping {opp['symbol']} — already has open trade")
                return None
        
        # Check if adaptive engine says NO
        if self.adaptive_engine:
            should_trade, reason, confidence = self.adaptive_engine.should_take_trade(
                opp, market_context
            )
            if not should_trade:
                logger.info(f"🚫 Trade REJECTED: {opp['symbol']} — {reason}")
                return None

        trade_id = f"T{int(time.time())}_{opp['symbol']}"
        trade = {
            "id": trade_id,
            "symbol": opp["symbol"],
            "direction": opp["direction"],
            "instrument_type": opp.get("instrument_type", "EQ"),
            "entry_price": opp["ltp"],
            "entry_time": datetime.now().isoformat(),
            "qty": opp["qty"],
            "target": opp["target_1"],
            "target_2": opp.get("target_2", opp["target_1"]),
            "stop_loss": opp["stop_loss"],
            "status": "OPEN",
            "pnl": 0,
            "exit_price": None,
            "exit_time": None,
            "reason": opp["reasons"][0] if opp["reasons"] else "Signal",
            "confidence": opp.get("confidence", 0.5),
            "strategies": opp.get("strategies", []),
            "score": opp.get("score", 0),
            # Tracking fields
            "peak_price": opp["ltp"],    # For trailing SL
            "trough_price": opp["ltp"],  # For trailing SL
            "partial_exit_done": False,
        }
        
        self.trades.append(trade)
        self._save_trades()
        
        # Record in journal with full context
        if self.journal:
            self.journal.record_entry(opp, market_context)
        
        instr = opp.get("instrument_type", "EQ")
        logger.info(f"🚀 [PAPER {instr}] Entered {trade['direction']} {trade['symbol']} "
                    f"@ ₹{trade['entry_price']} | Score: {opp.get('score', 0)} | "
                    f"Confidence: {opp.get('confidence', 0):.2f}")
        return trade

    def update_pnl(self, current_data: Dict[str, Dict], market_context: Dict = None):
        """Update P&L for all open trades with trailing stops."""
        changed = False
        summary = {"open_trades": 0, "total_pnl": 0, "closed_today": 0,
                    "wins": 0, "losses": 0}

        now_time = datetime.now().strftime("%H:%M")
        is_eod = now_time >= "15:25"
        
        trailing_sl_pct = 0.005  # Default 0.5%
        if self.adaptive_engine:
            trailing_sl_pct = self.adaptive_engine.params.trailing_sl_pct / 100

        for t in self.trades:
            if t["status"] != "OPEN":
                exit_time = t.get("exit_time")
                if exit_time and exit_time.startswith(datetime.now().strftime("%Y-%m-%d")):
                    summary["closed_today"] += 1
                    if t.get("pnl", 0) > 0:
                        summary["wins"] += 1
                    elif t.get("pnl", 0) < 0:
                        summary["losses"] += 1
                continue

            summary["open_trades"] += 1
            sym = t["symbol"]
            
            # For F&O paper trades, use underlying symbol
            underlying = t.get("underlying", sym)
            lookup_sym = underlying if underlying in current_data else sym
            
            if lookup_sym not in current_data:
                continue

            ltp = current_data[lookup_sym]["ltp"]
            
            # Track peak/trough for trailing stop
            if t["direction"] == "BUY":
                t["peak_price"] = max(t.get("peak_price", ltp), ltp)
            else:
                t["trough_price"] = min(t.get("trough_price", ltp), ltp)
            
            # Calculate live P&L
            if t["direction"] == "BUY":
                pnl = (ltp - t["entry_price"]) * t["qty"]
                
                # ── Exit Logic (Priority Order) ──
                
                # 1. Target 2 hit (full exit)
                if ltp >= t.get("target_2", t["target"]):
                    t["status"], t["exit_price"] = "TARGET2_HIT", ltp
                
                # 2. Target 1 hit (partial exit if not done)
                elif ltp >= t["target"] and not t.get("partial_exit_done"):
                    # Book partial profit, trail rest
                    t["partial_exit_done"] = True
                    # Move SL to breakeven
                    t["stop_loss"] = t["entry_price"]
                    logger.info(f"📊 Target 1 hit for {sym}! SL moved to breakeven ₹{t['entry_price']}")
                
                # 3. Trailing stop loss
                elif t["peak_price"] > t["entry_price"]:
                    trailing_sl = t["peak_price"] * (1 - trailing_sl_pct)
                    if trailing_sl > t["stop_loss"]:
                        t["stop_loss"] = round(trailing_sl, 2)
                    if ltp <= t["stop_loss"]:
                        t["status"], t["exit_price"] = "TRAIL_EXIT", ltp
                
                # 4. Original SL
                elif ltp <= t["stop_loss"]:
                    t["status"], t["exit_price"] = "SL_HIT", ltp
                
                # 5. EOD square-off
                elif is_eod:
                    t["status"], t["exit_price"] = "EOD_SQUARE_OFF", ltp
            else:
                pnl = (t["entry_price"] - ltp) * t["qty"]
                
                if ltp <= t.get("target_2", t["target"]):
                    t["status"], t["exit_price"] = "TARGET2_HIT", ltp
                elif ltp <= t["target"] and not t.get("partial_exit_done"):
                    t["partial_exit_done"] = True
                    t["stop_loss"] = t["entry_price"]
                    logger.info(f"📊 Target 1 hit for {sym}! SL moved to breakeven ₹{t['entry_price']}")
                elif t.get("trough_price", ltp) < t["entry_price"]:
                    trailing_sl = t["trough_price"] * (1 + trailing_sl_pct)
                    if trailing_sl < t["stop_loss"]:
                        t["stop_loss"] = round(trailing_sl, 2)
                    if ltp >= t["stop_loss"]:
                        t["status"], t["exit_price"] = "TRAIL_EXIT", ltp
                elif ltp >= t["stop_loss"]:
                    t["status"], t["exit_price"] = "SL_HIT", ltp
                elif is_eod:
                    t["status"], t["exit_price"] = "EOD_SQUARE_OFF", ltp

            t["pnl"] = round(pnl, 2)
            summary["total_pnl"] += pnl
            
            if t["status"] != "OPEN":
                t["exit_time"] = datetime.now().isoformat()
                changed = True
                
                # Record in journal
                if self.journal:
                    max_fav = abs(t.get("peak_price", ltp) - t["entry_price"]) if t["direction"] == "BUY" \
                              else abs(t["entry_price"] - t.get("trough_price", ltp))
                    max_adv = abs(t["entry_price"] - t.get("trough_price", ltp)) if t["direction"] == "BUY" \
                              else abs(t.get("peak_price", ltp) - t["entry_price"])
                    
                    self.journal.record_exit(
                        t["id"], ltp, t["status"],
                        max_favorable=max_fav, max_adverse=max_adv
                    )
                
                if t["pnl"] > 0:
                    summary["wins"] += 1
                    res = "✅ PROFIT"
                else:
                    summary["losses"] += 1
                    res = "❌ LOSS"
                logger.info(f"{res} | {t['status']} | {t['symbol']} @ ₹{ltp} | "
                           f"P&L: ₹{t['pnl']} | Strategies: {', '.join(t.get('strategies', []))}")

        if changed:
            self._save_trades()
        
        # ── DAILY LOSS CIRCUIT BREAKER ──────────────────────────────────
        # If cumulative daily P&L exceeds loss limit, force-close ALL open
        daily_pnl = self.get_daily_pnl()
        max_daily_loss_pct = 5.0  # default 5% of budget
        max_daily_loss = self.budget * (max_daily_loss_pct / 100)
        
        if daily_pnl < -max_daily_loss:
            open_trades = [t for t in self.trades if t["status"] == "OPEN"]
            if open_trades:
                logger.critical(
                    f"🚨 DAILY LOSS LIMIT BREACHED: ₹{daily_pnl:.2f} "
                    f"(limit: -₹{max_daily_loss:.2f}). Force-closing {len(open_trades)} positions!"
                )
                for t in open_trades:
                    sym = t["symbol"]
                    underlying = t.get("underlying", sym)
                    lookup_sym = underlying if underlying in current_data else sym
                    ltp = current_data[lookup_sym]["ltp"] if lookup_sym in current_data else t["entry_price"]
                    
                    if t["direction"] == "BUY":
                        pnl = (ltp - t["entry_price"]) * t["qty"]
                    else:
                        pnl = (t["entry_price"] - ltp) * t["qty"]
                    
                    t["status"] = "DAILY_LOSS_LIMIT"
                    t["exit_price"] = ltp
                    t["pnl"] = round(pnl, 2)
                    t["exit_time"] = datetime.now().isoformat()
                    
                    if self.journal:
                        self.journal.record_exit(t["id"], ltp, "DAILY_LOSS_LIMIT")
                    
                    logger.critical(f"🚨 FORCE CLOSED {sym} @ ₹{ltp} | P&L: ₹{t['pnl']}")
                
                self._save_trades()
                summary["daily_loss_limit_triggered"] = True
        
        return summary

    def get_open_count(self) -> int:
        """Get number of open positions."""
        return sum(1 for t in self.trades if t["status"] == "OPEN")

    def get_daily_pnl(self) -> float:
        """Get total P&L for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return sum(
            t.get("pnl", 0) for t in self.trades
            if (t.get("exit_time") and t["exit_time"].startswith(today)) or
            (t["status"] == "OPEN" and t.get("entry_time") and t["entry_time"].startswith(today))
        )

    def check_consecutive_losses(self):
        """Disable trading if too many consecutive losses."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_closed = [
            t for t in self.trades
            if (t.get("exit_time") or "").startswith(today) and t["status"] != "OPEN"
        ]
        # Check last N trades
        if len(today_closed) >= self.MAX_CONSECUTIVE_LOSSES:
            recent = today_closed[-self.MAX_CONSECUTIVE_LOSSES:]
            if all(t.get("pnl", 0) < 0 for t in recent):
                if not self._trading_disabled:
                    self._trading_disabled = True
                    total_loss = sum(t.get("pnl", 0) for t in recent)
                    logger.critical(
                        f"🚨 CONSECUTIVE LOSS BREAKER: {self.MAX_CONSECUTIVE_LOSSES} straight losses "
                        f"(total: ₹{total_loss:.2f}). Trading DISABLED for today."
                    )

    def get_daily_summary(self) -> Dict:
        """Generate end-of-day performance summary."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_trades = [
            t for t in self.trades
            if (t.get("entry_time") or "").startswith(today)
        ]
        
        closed = [t for t in today_trades if t["status"] != "OPEN"]
        still_open = [t for t in today_trades if t["status"] == "OPEN"]
        
        wins = [t for t in closed if t.get("pnl", 0) > 0]
        losses = [t for t in closed if t.get("pnl", 0) < 0]
        
        total_pnl = sum(t.get("pnl", 0) for t in today_trades)
        realized_pnl = sum(t.get("pnl", 0) for t in closed)
        unrealized_pnl = sum(t.get("pnl", 0) for t in still_open)
        
        biggest_win = max((t.get("pnl", 0) for t in closed), default=0)
        biggest_loss = min((t.get("pnl", 0) for t in closed), default=0)
        
        win_rate = len(wins) / len(closed) * 100 if closed else 0
        
        # Strategies used today
        strategies_used = set()
        for t in today_trades:
            for s in t.get("strategies", []):
                strategies_used.add(s)
        
        summary = {
            "date": today,
            "total_trades": len(today_trades),
            "closed": len(closed),
            "still_open": len(still_open),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "biggest_win": round(biggest_win, 2),
            "biggest_loss": round(biggest_loss, 2),
            "strategies_used": list(strategies_used),
            "trading_disabled": self._trading_disabled,
        }
        return summary


# ── Main Self-Learning Bot ───────────────────────────────────────────────────

class TradingBot:
    """
    Self-learning trading bot that:
    1. Scans equity, futures, and options markets
    2. Uses adaptive strategy weights based on past performance
    3. Records every trade with full context
    4. Learns from mistakes and adjusts parameters
    5. Creates new composite strategies from winning patterns
    """

    def __init__(self):
        self.cfg = load_config()
        self.auth = AngelOneAuth()
        self.budget = self.cfg.get("trading", {}).get("budget", 25000)
        
        # ── Self-learning components ──
        self.journal = TradeJournal()
        self.adaptive_engine = AdaptiveStrategyEngine(journal=self.journal)
        self.regime_detector = MarketRegimeDetector()
        self.fno_scanner = FnOScanner()
        self.commodity_analyzer = CommodityAnalyzer()  # ← ADD THIS
        
        # ── v3.0 Enhanced Components ──
        self.candle_manager = CandleManager(max_candles=200)
        self.ensemble_scorer = EnsembleScorer()
        self.micro_learner = MicroLearner(persistence_dir=".")
        self.cost_model = TransactionCostModel(trade_type="equity_intraday")
        self.correlation_mgr = CorrelationRiskManager()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.mtf_analyzer.set_candle_manager(self.candle_manager)
        
        # ── Core components ──
        self.scanner = MarketScanner(self.cfg, self.adaptive_engine)
        self.tm = TradeManager(self.budget, self.journal, self.adaptive_engine)
        
        self.is_running = False
        self.scan_count = 0
        self._learning_done_today = False
        
        # ── Persistent state recovery ──
        self._state_file = Path("data/bot_state.json")
        self._restore_state()

    def _restore_state(self):
        """Restore bot state from previous run (if same day)."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                # Only restore if state is from today
                if state.get("date") == datetime.now().strftime("%Y-%m-%d"):
                    self.scan_count = state.get("scan_count", 0)
                    self._learning_done_today = state.get("learning_done", False)
                    logger.info(
                        f"🔄 Restored bot state: scan_count={self.scan_count}, "
                        f"learning_done={self._learning_done_today}"
                    )
                else:
                    logger.info("📅 New trading day — fresh start")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not restore bot state: {e}")

    def _save_state(self):
        """Persist bot state for crash recovery."""
        try:
            state = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "scan_count": self.scan_count,
                "learning_done": self._learning_done_today,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Could not save bot state: {e}")

    def start(self):
        logger.info("=" * 80)
        logger.info("🤖  ANGEL ONE SELF-LEARNING TRADING BOT v3.0")
        logger.info("   ⚡ Enhanced: Real Indicators | Ensemble Scoring | MTF Confluence")
        logger.info("   ⚡ Enhanced: Correlation Risk | Micro-Learning | Transaction Costs")
        logger.info("=" * 80)
        logger.info(f"💰 Budget: ₹{self.budget:,}")
        logger.info(f"📊 Stocks: {len(FNO_TOKENS)} F&O + {len(INDEX_TOKENS)} indices")
        logger.info(f"🧠 Learning: ON | Journal: {len(self.journal.trades)} past trades")
        logger.info(f"🔬 Micro-Learner: {self.micro_learner._total_trades_processed} trades processed")
        
        # Print what the bot has learned
        summary = self.journal.get_learning_summary()
        if summary.get("total_trades", 0) > 0:
            logger.info(f"📈 Past performance: {summary.get('total_trades', 0)} trades, "
                       f"Win Rate: {summary.get('win_rate', 0):.1f}%, "
                       f"Total P&L: ₹{summary.get('total_pnl', 0):,.2f}")
            logger.info(f"🎯 Learned min score: {self.adaptive_engine.params.min_score}")
            logger.info(f"🛡️ SL multiplier: {self.adaptive_engine.params.sl_multiplier}x")
            logger.info(f"🎯 Target multiplier: {self.adaptive_engine.params.target_multiplier}x")
            if summary.get("best_strategies"):
                logger.info(f"⭐ Best strategies: {', '.join(summary['best_strategies'][:3])}")
            if summary.get("worst_strategies"):
                logger.info(f"⚠️ Worst strategies: {', '.join(summary['worst_strategies'][:3])}")
            if summary.get("common_mistakes"):
                top_mistakes = sorted(summary["common_mistakes"].items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"🔍 Common mistakes: {', '.join(f'{k}({v})' for k, v in top_mistakes)}")
            # v3.0: Print micro-learner insights
            micro_summary = self.micro_learner.get_learning_summary()
            bad_hours = micro_summary.get("bad_hours", [])
            if bad_hours:
                logger.info(f"⏰ Bad hours (learned): {bad_hours}")
            opt_score = micro_summary.get("optimal_score_threshold", 0)
            if opt_score > 0:
                logger.info(f"📊 Optimal score threshold (micro-learned): {opt_score:.0f}")
        else:
            logger.info("🆕 First run — bot will start learning from this session")
        
        logger.info("-" * 80)

        if self.auth.password:
            logger.info("🔐 Logging in to Angel One ...")
            if self.auth.login():
                logger.info("✅ Authenticated")
                self.scanner.set_auth(self.auth)
                if self.auth.smart_api:
                    self.fno_scanner.set_api(self.auth.smart_api)
                    # v3.0: Connect CandleManager to API
                    self.candle_manager.set_api(self.auth.smart_api)
                    logger.info("📊 CandleManager connected for real indicators")

        if is_market_hours():
            self._do_scan()

        schedule.every(5).minutes.do(self._scan_if_open)
        self.is_running = True
        logger.info("✅ Bot running. Press Ctrl+C to stop.")

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)
                
                # End-of-day learning
                now_time = datetime.now().strftime("%H:%M")
                if now_time >= "15:35" and not self._learning_done_today:
                    self._end_of_day_learning()
        except KeyboardInterrupt:
            self._end_of_day_learning()
            logger.info("🛑 Bot stopped")

    def _scan_if_open(self):
        if is_market_hours():
            # Re-authenticate before scanning if session is stale
            if not self.auth.ensure_authenticated():
                logger.warning("⚠️ Session expired and re-login failed — skipping scan")
                return
            self._do_scan()

    def _do_scan(self):
        try:
            self.scan_count += 1
            logger.info(f"{'='*60}")
            logger.info(f"📊 Scan #{self.scan_count} | {datetime.now().strftime('%H:%M:%S')}")
            
            # 1. Fetch market data
            raw_data = self.scanner.scan()
            if not raw_data:
                logger.warning("No market data received")
                return
            
            # 2. Detect market regime
            market_context = self.regime_detector.update(raw_data)
            market_context["budget"] = self.budget
            market_context["open_positions"] = self.tm.get_open_count()
            market_context["daily_pnl"] = self.tm.get_daily_pnl()
            
            logger.info(f"🌡️ Regime: {market_context['regime']} "
                        f"(confidence: {market_context['regime_confidence']:.2f}) | "
                        f"Volatility: {market_context['volatility']:.2f}%")
            
            # v3.0: Fetch candle data for top movers (limit API calls)
            self._fetch_candle_data_for_movers(raw_data)
            
            # v3.0: Update correlation manager with live prices
            for sym, data in raw_data.items():
                if data.get("instrument_type") != "INDEX":
                    self.correlation_mgr.update_price(sym, data.get("ltp", 0))
            
            # 3. Update P&L for existing trades with trailing stops
            summary = self.tm.update_pnl(raw_data, market_context)
            logger.info(f"💰 P&L: Open={summary['open_trades']} | "
                       f"Live P&L=₹{summary['total_pnl']:.2f} | "
                       f"Closed today: W{summary.get('wins', 0)}/L{summary.get('losses', 0)}")
            
            # v3.0: Feed micro-learner with any trades that just closed
            self._feed_micro_learner(market_context)
            
            # Check for consecutive loss circuit breaker
            self.tm.check_consecutive_losses()
            if summary.get("daily_loss_limit_triggered"):
                logger.critical("🚨 Daily loss limit triggered — monitoring only")
                return

            # 4. Don't enter new trades close to EOD
            now_time = datetime.now().strftime("%H:%M")
            max_entry_hour = self.adaptive_engine.params.avoid_after_hour
            if now_time >= f"{max_entry_hour}:00":
                logger.info(f"⏰ Past entry cutoff ({max_entry_hour}:00), monitoring only")
                return
            
            # v3.0: Check micro-learned bad hours
            current_hour = datetime.now().hour
            if self.micro_learner.is_bad_hour(current_hour):
                logger.info(f"⏰ Hour {current_hour} is historically bad — reducing entry aggressiveness")
                market_context["bad_hour"] = True

            # 5. Scan equity opportunities
            opps = self.scanner.opportunities(
                min_score=self.adaptive_engine.params.min_score,
                market_context=market_context
            )
            
            # v3.0: Enhance opportunities with ensemble scoring, MTF, and correlation checks
            opps = self._enhance_opportunities_v3(opps, market_context)
            
            # 6. Scan F&O opportunities
            fno_opps = self.fno_scanner.scan_all(raw_data, market_context)
            
            # 7. Scan MCX Commodity opportunities (if market hours)
            commodity_opps = []
            if is_commodity_market_hours():
                try:
                    commodity_raw = self.scanner.fetcher.fetch_commodities()
                    if commodity_raw:
                        commodity_opps = self.commodity_analyzer.analyze(commodity_raw)
                        logger.info(f"📦 Commodity scan: {len(commodity_raw)} prices, {len(commodity_opps)} signals")
                except Exception as e:
                    logger.warning(f"Commodity scan error: {e}")
            
            # 8. Combine and display
            all_opps = opps + fno_opps + commodity_opps
            all_opps.sort(key=lambda x: x.get("ensemble_score", x.get("score", 0)), reverse=True)
            
            if all_opps:
                # Auto-track top opportunities with GLOBAL position limit
                max_total = self.cfg.get("trading", {}).get("max_positions", 5)
                
                # ── MARKET REGIME FILTER: reduce capacity in adverse regimes ──
                regime = market_context.get("regime", "unknown")
                REGIME_CAPACITY = {
                    "trending": 1.0,     # Full capacity
                    "ranging": 0.75,     # 75% capacity
                    "volatile": 0.5,     # 50% capacity
                    "crash": 0.0,        # NO new entries
                }
                regime_factor = REGIME_CAPACITY.get(regime, 0.6)
                regime_adjusted_max = max(1, int(max_total * regime_factor))
                
                if regime_factor < 1.0:
                    logger.info(f"🌡️ Regime '{regime}' → max positions reduced: {max_total} → {regime_adjusted_max}")
                
                current_open = self.tm.get_open_count()
                remaining_capacity = max(0, regime_adjusted_max - current_open)
                
                if remaining_capacity == 0:
                    logger.info(f"📊 Max positions ({max_total}) reached — no new entries")
                
                eq_entered = 0
                fno_entered = 0
                max_eq = min(3, remaining_capacity)
                
                for o in all_opps:
                    # Hard stop: never exceed global position limit
                    total_entered = eq_entered + fno_entered
                    if total_entered >= remaining_capacity:
                        break
                    
                    # v3.0: Use ensemble score if available, otherwise fall back to raw score
                    entry_score = o.get("ensemble_score", o.get("score", 0))
                    if entry_score < self.adaptive_engine.params.high_confidence_score:
                        continue
                    
                    # v3.0: Skip opportunities with SKIP/WATCH ensemble recommendation
                    ens_rec = o.get("ensemble_recommendation", "ENTER")
                    if ens_rec in ("SKIP", "WATCH"):
                        continue
                    
                    # v3.0: Check correlation risk before entering
                    sym = o.get("symbol", "")
                    proposed_value = o.get("ltp", 0) * o.get("qty", 1)
                    corr_check = self.correlation_mgr.can_enter_position(
                        sym, proposed_value, self.budget
                    )
                    if not corr_check["allowed"]:
                        logger.info(f"🔗 Correlation block: {sym} — {corr_check['reason']}")
                        continue
                    
                    # v3.0: Apply correlation-based size adjustment
                    size_mult = corr_check.get("adjustments", {}).get("size_factor", 1.0)
                    if size_mult < 1.0:
                        o["qty"] = max(1, int(o["qty"] * size_mult))
                        logger.info(f"🔗 Size adjusted for {sym}: ×{size_mult:.0%} (correlation)")
                    
                    instr = o.get("instrument_type", "EQ")
                    
                    if instr == "EQ" and eq_entered < max_eq:
                        result = self.tm.track_opportunity(o, market_context)
                        if result:
                            eq_entered += 1
                            # v3.0: Update correlation manager with new position
                            self.correlation_mgr.update_position(
                                sym, o.get("qty", 1), proposed_value, o.get("direction", "BUY")
                            )
                    elif instr in ("OPT", "FUT") and fno_entered < min(2, remaining_capacity - eq_entered):
                        result = self.tm.track_opportunity(o, market_context)
                        if result:
                            fno_entered += 1

                # Display opportunities
                print(f"\n{'='*110}")
                print(f"🎯 {len(all_opps)} OPPORTUNITIES | "
                      f"Regime: {market_context['regime']} | "
                      f"Min Score: {self.adaptive_engine.params.min_score:.0f} | "
                      f"Sectors: {len(self.correlation_mgr.get_sector_exposure())} | "
                      f"{datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*110}")
                
                # Equity — with v3.0 ensemble info
                eq_opps = [o for o in all_opps if o.get("instrument_type") == "EQ"]
                if eq_opps:
                    print(f"\n  📈 EQUITY ({len(eq_opps)}):")
                    for i, o in enumerate(eq_opps[:10], 1):
                        st = ", ".join(o["strategies"][:2])
                        conf = o.get("confidence", 0)
                        ens = o.get("ensemble_score", o.get("score", 0))
                        mtf = o.get("mtf_recommendation", "-")
                        print(f"    {i:>2}. {o['symbol']:<14} {o['direction']:<4} "
                              f"₹{o['ltp']:<10.2f} Ens={ens:<5.0f} "
                              f"Conf={conf:<4.2f} MTF={mtf:<10} "
                              f"T1=₹{o['target_1']:<9.2f} SL=₹{o['stop_loss']:<9.2f} [{st}]")
                
                # F&O
                fno_display = [o for o in all_opps if o.get("instrument_type") in ("OPT", "FUT")]
                if fno_display:
                    print(f"\n  📊 F&O ({len(fno_display)}):")
                    for i, o in enumerate(fno_display[:5], 1):
                        st = ", ".join(o.get("strategies", [])[:2])
                        print(f"    {i:>2}. {o['symbol']:<24} {o['direction']:<4} "
                              f"Score={o.get('score', 0):<5.0f} "
                              f"T=₹{o['target_1']:<9.2f} SL=₹{o['stop_loss']:<9.2f} [{st}]")
                
                # Commodities
                commodity_display = [o for o in all_opps if o.get("instrument_type") == "COMMODITY"]
                if commodity_display:
                    print(f"\n  📦 COMMODITIES ({len(commodity_display)}):")
                    for i, o in enumerate(commodity_display[:5], 1):
                        st = ", ".join(o.get("strategies", [])[:2])
                        sector = o.get("commodity_sector", "Other")
                        print(f"    {i:>2}. {o['symbol']:<14} {o['direction']:<4} "
                              f"₹{o['ltp']:<10.2f} Score={o.get('score', 0):<5.0f} "
                              f"T=₹{o['target_1']:<9.2f} SL=₹{o['stop_loss']:<9.2f} "
                              f"[{sector}] [{st}]")
                
                # v3.0: Correlation risk summary
                risk_summary = self.correlation_mgr.get_risk_summary()
                if risk_summary["total_positions"] > 0:
                    print(f"\n  🔗 Portfolio Risk: {risk_summary['total_positions']} positions | "
                          f"Exposure: ₹{risk_summary['total_exposure']:,.0f} | "
                          f"Top Sector: {risk_summary['top_sector']}")
                
                print(f"{'='*110}\n")
            else:
                # Always print a scan summary even when no signals pass the threshold
                print(f"\n{'='*90}")
                print(f"📊 SCAN #{self.scan_count} | {datetime.now().strftime('%H:%M:%S')} "
                      f"| Regime: {market_context.get('regime','?')} "
                      f"| Volatility: {market_context.get('volatility',0):.2f}% "
                      f"| Min Score: {self.adaptive_engine.params.min_score:.0f}")
                print(f"   Fetched: {len(raw_data)} stocks | "
                      f"F&O signals: {len(fno_opps)} | "
                      f"Commodity signals: {len(commodity_opps)}")
                print(f"   ⚠️  No equity signals above score threshold ({self.adaptive_engine.params.min_score:.0f})")
                # Show top 5 movers anyway
                movers = sorted(
                    [d for d in raw_data.values() if d.get("instrument_type") != "INDEX"],
                    key=lambda x: abs(x.get("change_pct", 0)), reverse=True
                )[:5]
                if movers:
                    print(f"   Top movers:")
                    for m in movers:
                        arrow = "▲" if m.get("change_pct", 0) >= 0 else "▼"
                        print(f"     {arrow} {m['symbol']:<14} ₹{m['ltp']:<10.2f} {m.get('change_pct', 0):+.2f}%")
                # Show indices
                indices = {s: d for s, d in raw_data.items() if d.get("instrument_type") == "INDEX"}
                if indices:
                    idx_str = "  |  ".join(
                        f"{s}: {d['ltp']:.0f} ({d.get('change_pct', 0):+.2f}%)"
                        for s, d in indices.items()
                    )
                    print(f"   Indices: {idx_str}")
                print(f"{'='*90}\n")
                
        except Exception as e:
            logger.error(f"Scan error: {e}")
            traceback.print_exc()
        finally:
            self._save_state()  # Persist state after every scan

    # ── v3.0: New helper methods ────────────────────────────────────────────────

    def _fetch_candle_data_for_movers(self, raw_data: Dict):
        """Fetch candle data for top movers to compute real indicators."""
        try:
            # Sort by absolute change to prioritize active stocks
            sorted_syms = sorted(
                [(s, d) for s, d in raw_data.items() if d.get("instrument_type") != "INDEX"],
                key=lambda x: abs(x[1].get("change_pct", 0)),
                reverse=True,
            )
            
            # Fetch candles for top 10 movers only (Angel One rate limit: ~1 req/sec)
            fetched = 0
            for sym, data in sorted_syms[:10]:
                token = FNO_TOKENS.get(sym)
                if not token:
                    continue
                candles = self.candle_manager.fetch_candles(
                    sym, token, exchange="NSE", interval="5m", lookback_days=3
                )
                if candles:
                    fetched += 1
                time.sleep(0.8)  # Respect Angel One rate limit (AB1019 prevention)
            
            if fetched > 0:
                logger.info(f"📊 Fetched 5m candles for {fetched} stocks")
                
        except Exception as e:
            logger.debug(f"Candle fetch batch error: {e}")

    def _enhance_opportunities_v3(self, opps: List[Dict], market_context: Dict) -> List[Dict]:
        """
        v3.0: Enhance each opportunity with:
        1. Real candle-based indicators (RSI, ATR, VWAP, etc.)
        2. Ensemble scoring (rule + ML + indicator confluence + regime + journal)
        3. Multi-timeframe confluence check
        """
        enhanced = []
        regime = market_context.get("regime", "unknown")
        
        for opp in opps:
            sym = opp.get("symbol", "")
            ltp = opp.get("ltp", 0)
            direction = opp.get("direction", "BUY")
            
            # 1. Get real candle indicators
            candle_indicators = self.candle_manager.compute_all_indicators(sym, "5m")
            
            # Override proxy RSI with real RSI if available
            real_rsi = candle_indicators.get("rsi_14")
            if real_rsi is not None:
                opp["rsi"] = round(real_rsi, 1)
                opp["rsi_source"] = "candle_14p"  # Mark as real RSI
            
            # Add ATR-based targets if available
            atr = candle_indicators.get("atr_14")
            if atr and atr > 0:
                opp["atr"] = round(atr, 2)
                opp["atr_pct"] = candle_indicators.get("atr_pct", 0)
            
            # Add VWAP info
            vwap = candle_indicators.get("vwap")
            if vwap:
                opp["vwap"] = round(vwap, 2)
                opp["vwap_deviation"] = candle_indicators.get("vwap_deviation", 0)
            
            # 2. Ensemble scoring
            strategy_name = opp.get("strategies", [""])[0] if opp.get("strategies") else ""
            journal_adj = self.micro_learner.get_strategy_weight(strategy_name) - 1.0  # Convert to adjustment
            
            ensemble_result = self.ensemble_scorer.score_opportunity(
                rule_score=opp.get("score", 0),
                ml_probability=None,  # TODO: connect ModelTrainer predictions
                candle_indicators=candle_indicators,
                regime=regime,
                strategy_name=strategy_name,
                journal_adjustment=journal_adj,
                ltp=ltp,
            )
            
            opp["ensemble_score"] = ensemble_result["ensemble_score"]
            opp["ensemble_confidence"] = ensemble_result["confidence"]
            opp["ensemble_recommendation"] = ensemble_result["recommendation"]
            opp["ensemble_components"] = ensemble_result["components"]
            
            # 3. Multi-timeframe confluence
            mtf_result = self.mtf_analyzer.analyze(sym, direction)
            opp["mtf_confluence"] = mtf_result["confluence_score"]
            opp["mtf_recommendation"] = mtf_result["recommendation"]
            
            # Downgrade ensemble score if MTF is divergent
            if mtf_result["recommendation"] in ("DIVERGENT", "COUNTER_TREND"):
                opp["ensemble_score"] *= 0.7
                opp["ensemble_recommendation"] = "WATCH"
            elif mtf_result["recommendation"] == "STRONG_CONFLUENCE":
                opp["ensemble_score"] = min(100, opp["ensemble_score"] * 1.1)
            
            # 4. Transaction cost check — is the expected profit worth the costs?
            costs = self.cost_model.calculate_costs(
                buy_price=ltp, sell_price=opp.get("target_1", ltp),
                quantity=opp.get("qty", 1)
            )
            opp["est_costs"] = round(costs.total_cost, 2)
            opp["breakeven_pct"] = costs.breakeven_move_pct
            
            # Skip if costs eat more than 40% of expected profit
            expected_profit = opp.get("max_profit", 0)
            if expected_profit > 0 and costs.total_cost > expected_profit * 0.4:
                opp["ensemble_recommendation"] = "SKIP"
                opp["skip_reason"] = "transaction_costs_too_high"
            
            enhanced.append(opp)
        
        # Re-sort by ensemble score
        enhanced.sort(key=lambda x: x.get("ensemble_score", 0), reverse=True)
        return enhanced

    def _feed_micro_learner(self, market_context: Dict):
        """Feed the micro-learner with recently closed trades."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            for t in self.trades if hasattr(self, 'trades') else self.tm.trades:
                if t.get("status") == "OPEN":
                    continue
                # Check if this trade has already been fed to micro-learner
                if t.get("_micro_learned"):
                    continue
                # Only process today's closures
                exit_time = t.get("exit_time", "")
                if not exit_time or not exit_time.startswith(today):
                    continue
                
                self.micro_learner.learn_from_trade({
                    "symbol": t.get("symbol", ""),
                    "strategy": t.get("strategies", [""])[0] if t.get("strategies") else "unknown",
                    "pnl": t.get("pnl", 0),
                    "entry_price": t.get("entry_price", 0),
                    "exit_price": t.get("exit_price", 0),
                    "entry_time": t.get("entry_time", ""),
                    "exit_time": exit_time,
                    "regime": market_context.get("regime", "unknown"),
                    "signal_score": t.get("score", 0),
                    "sl_hit": t.get("status") == "SL_HIT",
                    "target_hit": "TARGET" in t.get("status", ""),
                    "qty": t.get("qty", 0),
                })
                
                # v3.0: Also record in ensemble scorer
                if t.get("_ensemble_signal"):
                    self.ensemble_scorer.record_outcome(
                        t["_ensemble_signal"], t.get("pnl", 0)
                    )
                
                # v3.0: Update correlation manager
                self.correlation_mgr.remove_position(t.get("symbol", ""))
                
                # Mark as processed
                t["_micro_learned"] = True
                
        except Exception as e:
            logger.debug(f"Micro-learner feed error: {e}")

    def _end_of_day_learning(self):
        """End of day: run the learning cycle."""
        if self._learning_done_today:
            return
        
        self._learning_done_today = True
        self._save_state()  # Persist learning flag before long operation
        logger.info("=" * 80)
        logger.info("🧠 END-OF-DAY LEARNING CYCLE (v3.0)")
        logger.info("=" * 80)
        
        try:
            # Run the adaptive engine's learning
            report = self.adaptive_engine.learn_from_session()
            
            # v3.0: Adapt ensemble weights based on accumulated outcomes
            self.ensemble_scorer.adapt_weights(min_samples=10)
            
            # Log the learning report
            if report:
                params = report.get("adaptive_params", {})
                logger.info(f"📊 Updated parameters:")
                logger.info(f"   Min score: {params.get('min_score', '?')}")
                logger.info(f"   SL multiplier: {params.get('sl_multiplier', '?')}x")
                logger.info(f"   Target multiplier: {params.get('target_multiplier', '?')}x")
                logger.info(f"   Entry cutoff: {params.get('avoid_after_hour', '?')}:00")
                
                weights = report.get("strategy_weights", {})
                if weights:
                    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                    logger.info(f"   Top strategies: {', '.join(f'{k}({v:.2f})' for k, v in sorted_w[:5])}")
                
                composites = report.get("composite_strategies", 0)
                if composites:
                    logger.info(f"   Composite strategies discovered: {composites}")
                
                journal_summary = report.get("journal_summary", {})
                if journal_summary.get("common_mistakes"):
                    logger.info(f"   Common mistakes: {json.dumps(journal_summary['common_mistakes'])}")
                if journal_summary.get("lessons"):
                    logger.info(f"   Recent lessons:")
                    for lesson in journal_summary["lessons"][-3:]:
                        logger.info(f"     - {lesson}")
            
            # v3.0: Log micro-learner and ensemble insights
            micro_summary = self.micro_learner.get_learning_summary()
            logger.info(f"\n  🔬 MICRO-LEARNER INSIGHTS:")
            logger.info(f"   Trades processed: {micro_summary['total_trades_processed']}")
            logger.info(f"   Optimal score threshold: {micro_summary['optimal_score_threshold']:.0f}")
            logger.info(f"   SL multiplier rec: {micro_summary['sl_multiplier_rec']:.2f}x")
            logger.info(f"   Target multiplier rec: {micro_summary['target_multiplier_rec']:.2f}x")
            if micro_summary.get('bad_hours'):
                logger.info(f"   Bad hours: {micro_summary['bad_hours']}")
            
            ens_summary = self.ensemble_scorer.get_summary()
            logger.info(f"\n  🎯 ENSEMBLE SCORER:")
            logger.info(f"   Weights: {json.dumps({k: round(v, 3) for k, v in ens_summary['weights'].items()})}")
            logger.info(f"   Total scored: {ens_summary['total_scored']}")
            
            # v3.0: Log strategy × regime heatmap
            heatmap = self.micro_learner.get_strategy_regime_heatmap()
            if heatmap:
                logger.info(f"\n  📊 STRATEGY × REGIME WIN RATES:")
                for strat, regimes in heatmap.items():
                    regime_str = ", ".join(f"{r}: {wr:.0%}" for r, wr in regimes.items())
                    logger.info(f"   {strat}: {regime_str}")
            
            # Save learning report to file
            report_data = report or {}
            report_data["v3_micro_learner"] = micro_summary
            report_data["v3_ensemble"] = ens_summary
            report_data["v3_strategy_regime_heatmap"] = heatmap
            report_data["v3_hourly_heatmap"] = self.micro_learner.get_hourly_heatmap()
            report_data["v3_correlation_risk"] = self.correlation_mgr.get_risk_summary()
            
            report_file = Path("data/journal") / f"learning_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"📝 Learning report saved to {report_file}")
            
            # ── DAILY SUMMARY ──
            daily = self.tm.get_daily_summary()
            
            # v3.0: Add net P&L after transaction costs
            today_str = datetime.now().strftime("%Y-%m-%d")
            total_costs = 0.0
            for t in self.tm.trades:
                if (t.get("exit_time") or "").startswith(today_str) and t.get("status") != "OPEN":
                    entry_p = t.get("entry_price", 0)
                    exit_p = t.get("exit_price", entry_p)
                    qty = t.get("qty", 1)
                    if entry_p > 0 and exit_p > 0:
                        costs = self.cost_model.calculate_costs(entry_p, exit_p, qty)
                        total_costs += costs.total_cost
            
            net_pnl = daily["total_pnl"] - total_costs
            
            logger.info("=" * 80)
            logger.info("📋 DAILY SUMMARY (v3.0)")
            logger.info("=" * 80)
            logger.info(f"   Date: {daily['date']}")
            logger.info(f"   Trades: {daily['total_trades']} ({daily['closed']} closed, {daily['still_open']} open)")
            logger.info(f"   Wins/Losses: {daily['wins']}/{daily['losses']} (Win Rate: {daily['win_rate']}%)")
            logger.info(f"   Gross P&L: ₹{daily['total_pnl']:.2f}")
            logger.info(f"   Transaction Costs: ₹{total_costs:.2f}")
            logger.info(f"   Net P&L: ₹{net_pnl:.2f}")
            logger.info(f"   Realized: ₹{daily['realized_pnl']} | Unrealized: ₹{daily['unrealized_pnl']}")
            logger.info(f"   Biggest Win: ₹{daily['biggest_win']} | Biggest Loss: ₹{daily['biggest_loss']}")
            logger.info(f"   Strategies: {', '.join(daily['strategies_used']) if daily['strategies_used'] else 'None'}")
            if daily['trading_disabled']:
                logger.info(f"   ⚠️ Trading was DISABLED by circuit breaker")
            logger.info("=" * 80)
            
            # Save daily summary with v3.0 enhancements
            daily["transaction_costs"] = round(total_costs, 2)
            daily["net_pnl"] = round(net_pnl, 2)
            summary_file = Path("data/journal") / f"daily_summary_{daily['date']}.json"
            with open(summary_file, 'w') as f:
                json.dump(daily, f, indent=2)
            
        except Exception as e:
            logger.error(f"Learning cycle error: {e}")
            traceback.print_exc()


def main():
    bot = TradingBot()
    bot.start()


if __name__ == "__main__":
    main()