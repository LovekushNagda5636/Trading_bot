#!/usr/bin/env python3
"""
Self-Learning Dashboard Server v2.0
====================================
Serves the trading dashboard with:
- Live / historical market data from Angel One
- Self-learning analytics (regime, journal, strategy weights)
- F&O opportunities
- Trade history and P&L tracking

Usage:
  python dashboard.py              # Live data (market hours) or last close data
  python dashboard.py --replay     # Fetch & display last trading day's data
"""

import json
import os
import sys
import logging
import time
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from functools import wraps

sys.path.insert(0, ".")
from continuous_trading_bot import (
    MarketScanner, StrategyEngine, load_config,
    is_market_hours, is_pre_market, is_commodity_market_hours,
    FNO_TOKENS, INDEX_TOKENS, MCX_COMMODITY_TOKENS, COMMODITY_SECTORS,
    CommodityAnalyzer,
)
from angel_one_auth_service import AngelOneAuth
from trading_bot.ml.trade_journal import TradeJournal
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party library logs
logging.getLogger("smartConnect").setLevel(logging.CRITICAL)
logging.getLogger("SmartApi").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)
# Restrict CORS to localhost only
CORS(app, origins=["http://127.0.0.1:*", "http://localhost:*"])

# ── Dashboard Authentication ─────────────────────────────────────────────────
DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")

def require_auth(f):
    """Decorator to require authentication on API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not DASHBOARD_TOKEN:
            # No token configured — allow access (dev mode)
            return f(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {DASHBOARD_TOKEN}":
            return f(*args, **kwargs)
        # Also accept token as query parameter for browser convenience
        if request.args.get("token") == DASHBOARD_TOKEN:
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized"}), 401
    return decorated

# ── Health Check (no auth required) ──────────────────────────────────────────
_start_time = datetime.now()

@app.route("/health")
def health_check():
    """Health check for systemd, load balancer, and monitoring."""
    import psutil
    process = psutil.Process()
    return jsonify({
        "status": "healthy",
        "uptime_seconds": int((datetime.now() - _start_time).total_seconds()),
        "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
        "market_open": is_market_hours(),
        "timestamp": datetime.now().isoformat(),
    }), 200

# ── Global State ──────────────────────────────────────────────────────────────
cfg = load_config()
auth = AngelOneAuth()

# Self-learning components
journal = TradeJournal()
adaptive_engine = AdaptiveStrategyEngine(journal=journal)
regime_detector = MarketRegimeDetector()
fno_scanner = FnOScanner()
commodity_analyzer = CommodityAnalyzer()

# v3.0 Enhanced Components
candle_manager = CandleManager(max_candles=200)
ensemble_scorer = EnsembleScorer()
micro_learner = MicroLearner(persistence_dir=".")
cost_model = TransactionCostModel(trade_type="equity_intraday")
correlation_mgr = CorrelationRiskManager()
mtf_analyzer = MultiTimeframeAnalyzer()
mtf_analyzer.set_candle_manager(candle_manager)

# Scanner with adaptive engine
scanner = MarketScanner(cfg, adaptive_engine)

# Cache
last_scan = {"opportunities": [], "market_summary": {}, "timestamp": None}
cached_market_data = {}
cached_fno_opps = []
cached_context = {}
cached_commodity_data = {}
cached_commodity_signals = []


def init():
    """Initialize auth and data feeds."""
    if auth.password:
        logger.info("Logging in to Angel One...")
        if auth.login():
            logger.info("Authenticated - live market data enabled")
            scanner.set_auth(auth)
            if auth.smart_api:
                fno_scanner.set_api(auth.smart_api)
                # v3.0: Connect CandleManager to API
                candle_manager.set_api(auth.smart_api)
                logger.info("📊 CandleManager connected for real indicators")
        else:
            logger.error("Auth failed - dashboard will show cached data only")


def fetch_market_data():
    """Fetch current market data from Angel One."""
    global cached_market_data, cached_context, cached_fno_opps
    global cached_commodity_data, cached_commodity_signals

    raw = scanner.scan()
    if not raw:
        logger.warning("No market data received")

    if raw:
        cached_market_data = raw

        # Update regime detector
        cached_context = regime_detector.update(raw)
        cached_context["budget"] = cfg.get("trading", {}).get("budget", 25000)
        
        # v3.0: Update correlation manager with live prices
        for sym, data in raw.items():
            if data.get("instrument_type") != "INDEX":
                correlation_mgr.update_price(sym, data.get("ltp", 0))

        # Scan F&O opportunities
        cached_fno_opps = fno_scanner.scan_all(raw, cached_context)

    # Fetch & analyze MCX commodities (runs even if equity scan fails)
    try:
        if scanner.fetcher:
            logger.info("📦 Fetching MCX commodity data...")
            commodity_raw = scanner.fetcher.fetch_commodities()
            if commodity_raw:
                cached_commodity_data = commodity_raw
                cached_commodity_signals = commodity_analyzer.analyze(commodity_raw)
                logger.info(f"📦 Commodity scan: {len(commodity_raw)} prices, {len(cached_commodity_signals)} signals")
            else:
                logger.warning("📦 Commodity fetch returned empty — MCX segment may not be active")
        else:
            logger.warning("📦 No fetcher available for commodities")
    except Exception as e:
        logger.warning(f"Commodity scan error: {e}")

    return raw


def _get_commodity_response():
    """Build commodity portion of API response."""
    return {
        "commodity_data": list(cached_commodity_data.values()),
        "commodity_signals": cached_commodity_signals[:20],
        "commodity_market_open": is_commodity_market_hours(),
        "total_commodities": len(MCX_COMMODITY_TOKENS),
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("dashboard.html")


@app.route("/api/scan")
@require_auth
def api_scan():
    """Full market scan — equity + F&O + learning analytics."""
    global last_scan
    try:
        raw = fetch_market_data()
        if not raw:
            if last_scan.get("timestamp"):
                # Inject fresh commodity data into cached response
                last_scan.update(_get_commodity_response())
                return jsonify(last_scan)
            return jsonify({"error": "No market data available"}), 503

        # Equity opportunities with adaptive scoring
        opps = scanner.opportunities(
            min_score=0,
            market_context=cached_context
        )
        
        # v3.0: Enhance with real indicators, ensemble scoring, and MTF
        opps = _enhance_dashboard_opps_v3(opps, cached_context)

        # Combine equity + F&O opportunities
        all_opps = opps + cached_fno_opps
        all_opps.sort(key=lambda x: x.get("ensemble_score", x.get("score", 0)), reverse=True)

        total = len(raw)
        gainers = sum(1 for d in raw.values()
                      if d.get("change_pct", 0) > 0 and d.get("instrument_type") != "INDEX")
        losers = sum(1 for d in raw.values()
                     if d.get("change_pct", 0) < 0 and d.get("instrument_type") != "INDEX")

        top_g = max(
            [d for d in raw.values() if d.get("instrument_type") != "INDEX"],
            key=lambda x: x.get("change_pct", 0), default={}
        )
        top_l = min(
            [d for d in raw.values() if d.get("instrument_type") != "INDEX"],
            key=lambda x: x.get("change_pct", 0), default={}
        )

        indices = {sym: data for sym, data in raw.items()
                   if data.get("instrument_type") == "INDEX"}

        market_summary = {
            "total_stocks": total,
            "gainers": gainers,
            "losers": losers,
            "unchanged": total - gainers - losers,
            "top_gainer": {
                "symbol": top_g.get("symbol", "-"),
                "change_pct": round(top_g.get("change_pct", 0), 2),
                "ltp": top_g.get("ltp", 0),
            },
            "top_loser": {
                "symbol": top_l.get("symbol", "-"),
                "change_pct": round(top_l.get("change_pct", 0), 2),
                "ltp": top_l.get("ltp", 0),
            },
            "indices": {
                s: {"ltp": round(d["ltp"], 2), "change_pct": round(d.get("change_pct", 0), 2)}
                for s, d in indices.items()
            },
            "is_market_open": is_market_hours(),
            "is_pre_market": is_pre_market(),
            # Self-learning context
            "regime": cached_context.get("regime", "unknown"),
            "regime_confidence": cached_context.get("regime_confidence", 0),
            "volatility": cached_context.get("volatility", 0),
            "sector_trends": cached_context.get("sector_trends", {}),
            # v3.0 Risk Context
            "portfolio_risk": correlation_mgr.get_risk_summary(),
        }

        all_stocks = []
        for d in sorted(
            [v for v in raw.values() if v.get("instrument_type") != "INDEX"],
            key=lambda x: abs(x.get("change_pct", 0)),
            reverse=True
        ):
            all_stocks.append({
                "symbol": d.get("symbol", ""),
                "ltp": round(d.get("ltp", 0), 2),
                "open": round(d.get("open", 0), 2),
                "high": round(d.get("high", 0), 2),
                "low": round(d.get("low", 0), 2),
                "prev_close": round(d.get("prev_close", 0), 2),
                "change_pct": round(d.get("change_pct", 0), 2),
                "volume": d.get("volume", 0),
            })

        # Trade tracking data
        trades = list(scanner.strategy.trades.values()) if hasattr(scanner.strategy, "trades") else []
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        closed_trades = [t for t in trades if t.get("status") != "OPEN"]
        current_pnl = sum(t.get("pnl", 0) for t in open_trades)
        total_realized = sum(t.get("pnl", 0) for t in closed_trades)

        wins = len([t for t in closed_trades
                    if t.get("status", "").startswith("TARGET") or
                    (t.get("status") in ("TRAIL_EXIT", "EOD_SQUARE_OFF") and t.get("pnl", 0) > 0)])

        # Self-learning analytics
        learning_summary = journal.get_learning_summary()
        engine_report = adaptive_engine.get_learning_report()

        last_scan = {
            "opportunities": all_opps[:50],
            "market_summary": market_summary,
            "all_stocks": all_stocks,
            "trades": {
                "active": open_trades[::-1],
                "history": closed_trades[::-1][:30],
                "live_pnl": round(current_pnl, 2),
                "realized_pnl": round(total_realized, 2),
                "win_rate": round(wins / len(closed_trades) * 100, 1) if closed_trades else 0,
            },
            # F&O section
            "fno_opportunities": cached_fno_opps[:10],
            # Self-learning section
            "learning": {
                "regime": cached_context.get("regime", "unknown"),
                "regime_confidence": cached_context.get("regime_confidence", 0),
                "volatility": round(cached_context.get("volatility", 0), 2),
                "min_score": adaptive_engine.params.min_score,
                "sl_multiplier": adaptive_engine.params.sl_multiplier,
                "target_multiplier": adaptive_engine.params.target_multiplier,
                "journal_trades": learning_summary.get("total_trades", 0),
                "journal_win_rate": learning_summary.get("win_rate", 0),
                "journal_pnl": learning_summary.get("total_pnl", 0),
                "best_strategies": learning_summary.get("best_strategies", [])[:5],
                "worst_strategies": learning_summary.get("worst_strategies", [])[:3],
                "common_mistakes": learning_summary.get("common_mistakes", {}),
                "composite_strategies": engine_report.get("composite_strategies", 0),
                "strategy_weights": engine_report.get("strategy_weights", {}),
                "sector_trends": cached_context.get("sector_trends", {}),
                # v3.0 Micro-Learner insights
                "micro_learned": micro_learner.get_learning_summary(),
                "ensemble_stats": ensemble_scorer.get_summary(),
            },
            "timestamp": datetime.now().isoformat(),
            "auth_status": auth.is_authenticated,
            "budget": cfg.get("trading", {}).get("budget", 25000),
            "total_fno_stocks": len(scanner.fno_stocks),
        }

        # Always inject fresh commodity data
        last_scan.update(_get_commodity_response())

        return jsonify(last_scan)

    except Exception as e:
        logger.error(f"Scan error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
@require_auth
def api_status():
    return jsonify({
        "bot_running": True,
        "auth_status": auth.is_authenticated,
        "market_open": is_market_hours(),
        "last_scan": last_scan.get("timestamp"),
        "fno_stocks": len(scanner.fno_stocks),
        "regime": cached_context.get("regime", "unknown"),
        "learning_enabled": True,
    })


@app.route("/api/learning")
@require_auth
def api_learning():
    """Get detailed learning analytics."""
    summary = journal.get_learning_summary()
    report = adaptive_engine.get_learning_report()
    return jsonify({
        "summary": summary,
        "report": report,
        "journal_trades": len(journal.trades),
        "strategy_stats": {
            k: {
                "total_trades": v.total_trades,
                "win_rate": round(v.win_rate * 100, 1),
                "total_pnl": round(v.total_pnl, 2),
                "confidence": round(v.confidence_score, 3),
            }
            for k, v in journal.strategy_stats.items()
        },
    })


@app.route("/api/journal")
@require_auth
def api_journal():
    """Get trade journal data."""
    trades = []
    for t in journal.trades[-50:]:
        trades.append({
            "trade_id": t.trade_id,
            "symbol": t.symbol,
            "direction": t.direction,
            "instrument_type": t.instrument_type,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "status": t.status,
            "strategies": t.strategies_used,
            "score": t.signal_score,
            "regime": t.market_regime,
            "mistake_type": t.mistake_type,
            "lesson": t.lesson_learned,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
        })
    return jsonify({"trades": trades[::-1]})


@app.route("/api/commodities")
@app.route("/api/risk")
@require_auth
def api_risk():
    """Get detailed risk management analytics."""
    return jsonify({
        "sector_exposure": correlation_mgr.get_sector_exposure(),
        "position_correlations": correlation_mgr.get_full_correlation_matrix(),
        "risk_summary": correlation_mgr.get_risk_summary(),
        "budget": cfg.get("trading", {}).get("budget", 25000),
    })


# ── v3.0 Helper for Dashboard ────────────────────────────────────────────────

def _enhance_dashboard_opps_v3(opps, market_context):
    """v3.0 enhancement pipeline replicated for dashboard UI."""
    enhanced = []
    regime = market_context.get("regime", "unknown")
    
    # Sort and take top 15 for deep analysis (candle fetch)
    top_opps = sorted(opps, key=lambda x: x.get("score", 0), reverse=True)[:15]
    
    for opp in top_opps:
        sym = opp.get("symbol", "")
        ltp = opp.get("ltp", 0)
        direction = opp.get("direction", "BUY")
        
        # 1. Fetch candles for indicators (if not cached)
        token = FNO_TOKENS.get(sym)
        if token and auth.smart_api:
            # Short lookback for dashboard responsiveness
            candle_manager.fetch_candles(sym, token, interval="5m", lookback_days=2)
            
        # 2. Get real indicators
        indicators = candle_manager.compute_all_indicators(sym, "5m")
        if indicators:
            opp["rsi_real"] = indicators.get("rsi_14")
            opp["atr"] = indicators.get("atr_14")
            opp["vwap"] = indicators.get("vwap")
            
        # 3. Ensemble score
        strategy_name = opp.get("strategies", [""])[0] if opp.get("strategies") else ""
        journal_adj = micro_learner.get_strategy_weight(strategy_name) - 1.0
        
        ens = ensemble_scorer.score_opportunity(
            rule_score=opp.get("score", 0),
            ml_probability=None,
            candle_indicators=indicators,
            regime=regime,
            strategy_name=strategy_name,
            journal_adjustment=journal_adj,
            ltp=ltp
        )
        opp["ensemble_score"] = ens["ensemble_score"]
        opp["ensemble_components"] = ens["components"]
        opp["ensemble_recommendation"] = ens["recommendation"]
        
        # 4. MTF
        mtf = mtf_analyzer.analyze(sym, direction)
        opp["mtf_confluence"] = mtf["confluence_score"]
        opp["mtf_recommendation"] = mtf["recommendation"]
        
        # 5. Costs
        costs = cost_model.calculate_costs(ltp, opp.get("target_1", ltp), opp.get("qty", 1))
        opp["est_costs"] = round(costs.total_cost, 2)
        opp["breakeven_pct"] = costs.breakeven_move_pct
        
        enhanced.append(opp)
        
    return enhanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--replay", action="store_true",
                        help="Fetch last trading day data")
    args = parser.parse_args()

    init()

    print()
    print("=" * 60)
    print("  SELF-LEARNING TRADING DASHBOARD v2.0")
    print("=" * 60)
    print(f"  Dashboard: http://127.0.0.1:{args.port}")
    print(f"  Auth: {'Connected' if auth.is_authenticated else 'Not connected'}")
    print(f"  Stocks: {len(FNO_TOKENS)} F&O + {len(INDEX_TOKENS)} indices")
    print(f"  Commodities: {len(MCX_COMMODITY_TOKENS)} MCX contracts")
    print(f"  Learning: {len(journal.trades)} trades in journal")
    print(f"  Min Score: {adaptive_engine.params.min_score}")
    print("=" * 60)
    print()

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
