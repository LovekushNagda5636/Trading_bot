"""
Transaction Cost Model — Accurately models Angel One brokerage,
statutory charges, and taxes for P&L calculations.

India Equity Intraday (MIS):
    - Brokerage: ₹20 per executed order (or 0.03%, whichever is lower)
    - STT: 0.025% on sell side
    - Transaction charges: NSE 0.00345%, BSE 0.00375%
    - GST: 18% on (brokerage + transaction charges)
    - SEBI charges: 0.0001%
    - Stamp duty: 0.003% on buy side

India F&O:
    - Brokerage: ₹20 per executed order
    - STT: 0.0125% on sell side (Futures), 0.0625% on sell side (Options)
    - Transaction charges: 0.05% (Options), 0.0019% (Futures)
    - GST: 18% on (brokerage + transaction charges)
    - SEBI charges: 0.0001%
    - Stamp duty: 0.003% on buy side (Futures), 0.003% (Options)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostResult:
    """Breakdown of costs for a trade."""
    brokerage: float = 0.0
    stt: float = 0.0
    transaction_charges: float = 0.0
    gst: float = 0.0
    sebi_charges: float = 0.0
    stamp_duty: float = 0.0
    total_cost: float = 0.0
    cost_per_unit: float = 0.0
    breakeven_move_pct: float = 0.0


class TransactionCostModel:
    """
    Calculate accurate transaction costs for Angel One trades.
    Supports Equity Intraday, Equity Delivery, Futures, and Options.
    """

    # ── Angel One charge schedule ─────────────────────────────────────────────

    BROKERAGE_FLAT = 20.0  # ₹20 per order
    BROKERAGE_PCT_CAP = 0.0003  # 0.03% cap

    CHARGES = {
        "equity_intraday": {
            "stt_sell": 0.00025,         # 0.025% sell side
            "txn_nse": 0.0000345,        # NSE transaction charges
            "txn_bse": 0.0000375,        # BSE transaction charges
            "sebi": 0.000001,            # SEBI turnover charge
            "stamp_buy": 0.00003,        # Stamp duty on buy side
            "gst_rate": 0.18,            # GST on brokerage + txn charges
        },
        "equity_delivery": {
            "stt_both": 0.001,           # 0.1% both sides
            "txn_nse": 0.0000345,
            "txn_bse": 0.0000375,
            "sebi": 0.000001,
            "stamp_buy": 0.00015,
            "gst_rate": 0.18,
        },
        "futures": {
            "stt_sell": 0.000125,        # 0.0125% sell side
            "txn_nse": 0.0000190,        # NSE F&O charges
            "sebi": 0.000001,
            "stamp_buy": 0.00002,
            "gst_rate": 0.18,
        },
        "options": {
            "stt_sell": 0.000625,        # 0.0625% sell side (on premium)
            "txn_nse": 0.0005,           # NSE Options charges (on premium)
            "sebi": 0.000001,
            "stamp_buy": 0.00003,
            "gst_rate": 0.18,
        },
    }

    def __init__(self, trade_type: str = "equity_intraday", exchange: str = "NSE"):
        """
        Args:
            trade_type: 'equity_intraday', 'equity_delivery', 'futures', 'options'
            exchange: 'NSE' or 'BSE'
        """
        self.trade_type = trade_type
        self.exchange = exchange

    def calculate_costs(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int,
        lot_size: int = 1,
    ) -> TransactionCostResult:
        """
        Calculate total transaction costs for a complete round-trip trade.

        Args:
            buy_price: Entry price per unit
            sell_price: Exit price per unit
            quantity: Number of units (or lots for F&O)
            lot_size: Lot size (1 for equity, actual lot size for F&O)

        Returns:
            TransactionCostResult with full breakdown
        """
        charges = self.CHARGES.get(self.trade_type, self.CHARGES["equity_intraday"])
        actual_qty = quantity * lot_size

        buy_value = buy_price * actual_qty
        sell_value = sell_price * actual_qty
        total_turnover = buy_value + sell_value

        result = TransactionCostResult()

        # ── Brokerage ─────────────────────────────────────────────────────
        # ₹20 per order or 0.03% of turnover, whichever is lower — per side
        buy_brokerage = min(self.BROKERAGE_FLAT, buy_value * self.BROKERAGE_PCT_CAP)
        sell_brokerage = min(self.BROKERAGE_FLAT, sell_value * self.BROKERAGE_PCT_CAP)
        result.brokerage = buy_brokerage + sell_brokerage

        # ── STT ───────────────────────────────────────────────────────────
        if "stt_sell" in charges:
            result.stt = sell_value * charges["stt_sell"]
        elif "stt_both" in charges:
            result.stt = total_turnover * charges["stt_both"]

        # ── Transaction charges ───────────────────────────────────────────
        txn_key = "txn_nse" if self.exchange == "NSE" else "txn_bse"
        txn_rate = charges.get(txn_key, charges.get("txn_nse", 0))
        result.transaction_charges = total_turnover * txn_rate

        # ── GST (on brokerage + transaction charges) ──────────────────────
        gst_rate = charges.get("gst_rate", 0.18)
        result.gst = (result.brokerage + result.transaction_charges) * gst_rate

        # ── SEBI charges ──────────────────────────────────────────────────
        sebi_rate = charges.get("sebi", 0)
        result.sebi_charges = total_turnover * sebi_rate

        # ── Stamp duty (buy side only) ────────────────────────────────────
        stamp_rate = charges.get("stamp_buy", 0)
        result.stamp_duty = buy_value * stamp_rate

        # ── Totals ────────────────────────────────────────────────────────
        result.total_cost = (
            result.brokerage
            + result.stt
            + result.transaction_charges
            + result.gst
            + result.sebi_charges
            + result.stamp_duty
        )

        if actual_qty > 0:
            result.cost_per_unit = result.total_cost / actual_qty

        # Breakeven: minimum price move needed to cover costs
        if buy_value > 0:
            result.breakeven_move_pct = round(
                (result.total_cost / buy_value) * 100, 4
            )

        return result

    def adjust_pnl(
        self,
        gross_pnl: float,
        buy_price: float,
        sell_price: float,
        quantity: int,
        lot_size: int = 1,
    ) -> Dict[str, float]:
        """
        Compute net P&L after all transaction costs.

        Returns:
            Dict with gross_pnl, total_costs, net_pnl, cost_breakdown
        """
        costs = self.calculate_costs(buy_price, sell_price, quantity, lot_size)

        return {
            "gross_pnl": round(gross_pnl, 2),
            "total_costs": round(costs.total_cost, 2),
            "net_pnl": round(gross_pnl - costs.total_cost, 2),
            "breakeven_move_pct": costs.breakeven_move_pct,
            "cost_breakdown": {
                "brokerage": round(costs.brokerage, 2),
                "stt": round(costs.stt, 2),
                "transaction_charges": round(costs.transaction_charges, 2),
                "gst": round(costs.gst, 2),
                "sebi_charges": round(costs.sebi_charges, 2),
                "stamp_duty": round(costs.stamp_duty, 2),
            },
        }

    @staticmethod
    def estimate_min_target_pct(
        trade_type: str = "equity_intraday", exchange: str = "NSE"
    ) -> float:
        """
        Estimate minimum target percentage needed to cover round-trip costs.
        Useful for setting minimum take-profit levels.
        """
        model = TransactionCostModel(trade_type, exchange)
        # Use ₹500 stock, 20 shares as a representative trade
        costs = model.calculate_costs(
            buy_price=500, sell_price=500, quantity=20
        )
        return costs.breakeven_move_pct
