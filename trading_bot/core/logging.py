"""
Enhanced logging system for the trading bot.
- Separate log files for trading, errors, signals, risk, and audit
- Color-coded console output
- Daily log rotation with date-stamped filenames
- Structured JSON audit logs with all extra fields captured
- Performance logging with timing context manager
- Trade-specific logging helpers
- Thread-safe throughout
"""

import logging
import logging.handlers
import sys
import os
import time
import threading
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
import json


# ═══════════════════════════════════════════════════════════════
# COLOR CODES FOR CONSOLE
# ═══════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Disable colors on Windows without ANSI support or non-TTY
    @classmethod
    def supported(cls) -> bool:
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        if os.name == "nt":
            try:
                os.system("")  # Enable ANSI on Windows 10+
                return True
            except Exception:
                return False
        return True


# ═══════════════════════════════════════════════════════════════
# CUSTOM FORMATTERS
# ═══════════════════════════════════════════════════════════════

class ColorConsoleFormatter(logging.Formatter):
    """Color-coded console formatter with compact, readable output."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.WHITE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    LEVEL_ICONS = {
        logging.DEBUG: "🔍",
        logging.INFO: "✅",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and Colors.supported()

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        name = record.name

        # Shorten logger name for readability
        if "." in name:
            parts = name.split(".")
            if len(parts) > 2:
                name = f"{parts[0]}.{parts[-1]}"

        message = record.getMessage()

        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            icon = self.LEVEL_ICONS.get(record.levelno, "")
            formatted = (
                f"{Colors.DIM}{timestamp}{Colors.RESET} "
                f"{color}{level:<8}{Colors.RESET} "
                f"{Colors.CYAN}{name:<25}{Colors.RESET} "
                f"{icon} {message}"
            )
        else:
            formatted = f"{timestamp} {level:<8} {name:<25} {message}"

        # Append exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            exc_text = self.formatException(record.exc_info)
            formatted += f"\n{exc_text}"

        return formatted


class DetailedFileFormatter(logging.Formatter):
    """Detailed file formatter with full context."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]  # Millisecond precision
        level = record.levelname
        name = record.name
        message = record.getMessage()
        thread = record.thread
        func = record.funcName
        lineno = record.lineno
        filename = record.filename

        line = (
            f"{timestamp} | {level:<8} | {name} | "
            f"{filename}:{func}:{lineno} | [T:{thread}] | {message}"
        )

        # Add extra fields if they exist (for trade/order context)
        extras = {}
        skip_keys = {
            "message", "args", "created", "relativeCreated", "msecs",
            "levelname", "levelno", "pathname", "filename", "module",
            "funcName", "lineno", "exc_info", "exc_text", "stack_info",
            "name", "thread", "threadName", "process", "processName",
            "msg", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip_keys and not key.startswith("_"):
                extras[key] = value

        if extras:
            line += f" | extra={json.dumps(extras, default=str)}"

        if record.exc_info and record.exc_info[0] is not None:
            line += f"\n{self.formatException(record.exc_info)}"

        return line


class AuditFormatter(logging.Formatter):
    """
    Structured JSON formatter for audit logs.
    Captures ALL extra fields passed via the `extra` kwarg to logger methods.
    Ensures regulatory compliance for SEBI requirements.
    """

    def format(self, record: logging.LogRecord) -> str:
        audit_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "thread_id": record.thread,
            "process_id": record.process,
        }

        # Capture every extra field (trade_id, order_id, symbol, action, etc.)
        skip_keys = {
            "message", "args", "created", "relativeCreated", "msecs",
            "levelname", "levelno", "pathname", "filename", "module",
            "funcName", "lineno", "exc_info", "exc_text", "stack_info",
            "name", "thread", "threadName", "process", "processName",
            "msg", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip_keys and not key.startswith("_"):
                # Serialize non-primitive types
                if isinstance(value, (dict, list)):
                    audit_data[key] = value
                elif isinstance(value, datetime):
                    audit_data[key] = value.isoformat()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    audit_data[key] = value
                else:
                    audit_data[key] = str(value)

        if record.exc_info and record.exc_info[0] is not None:
            audit_data["exception"] = traceback.format_exception(*record.exc_info)

        return json.dumps(audit_data, default=str, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# DAILY ROTATING FILE HANDLER
# ═══════════════════════════════════════════════════════════════

class DailyRotatingFileHandler(logging.Handler):
    """
    File handler that creates a new log file each day.
    Format: {prefix}_{YYYY-MM-DD}.log
    Automatically cleans up old log files beyond retention days.
    """

    def __init__(
        self,
        log_dir: str,
        prefix: str = "trading",
        retention_days: int = 30,
        encoding: str = "utf-8",
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.retention_days = retention_days
        self.encoding = encoding
        self._current_date: Optional[date] = None
        self._current_handler: Optional[logging.FileHandler] = None
        self._lock_obj = threading.Lock()

    def _get_log_path(self, for_date: date) -> Path:
        return self.log_dir / f"{self.prefix}_{for_date.isoformat()}.log"

    def _ensure_handler(self) -> logging.FileHandler:
        today = date.today()
        if self._current_date != today or self._current_handler is None:
            # Close old handler
            if self._current_handler is not None:
                try:
                    self._current_handler.close()
                except Exception:
                    pass

            log_path = self._get_log_path(today)
            self._current_handler = logging.FileHandler(
                log_path, mode="a", encoding=self.encoding
            )
            if self.formatter:
                self._current_handler.setFormatter(self.formatter)
            self._current_date = today

            # Cleanup old files
            self._cleanup_old_files()

        return self._current_handler

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock_obj:
            try:
                handler = self._ensure_handler()
                if self.formatter:
                    handler.setFormatter(self.formatter)
                handler.emit(record)
            except Exception:
                self.handleError(record)

    def _cleanup_old_files(self) -> None:
        """Remove log files older than retention_days."""
        try:
            cutoff = datetime.now().timestamp() - (self.retention_days * 86400)
            for f in self.log_dir.glob(f"{self.prefix}_*.log"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def close(self) -> None:
        with self._lock_obj:
            if self._current_handler is not None:
                try:
                    self._current_handler.close()
                except Exception:
                    pass
                self._current_handler = None
        super().close()


# ═══════════════════════════════════════════════════════════════
# MAIN SETUP FUNCTION
# ═══════════════════════════════════════════════════════════════

_logging_initialized = False
_logging_lock = threading.Lock()


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_audit: bool = True,
    enable_error_file: bool = True,
    enable_trade_file: bool = True,
    enable_signal_file: bool = True,
    retention_days: int = 30,
) -> None:
    """
    Set up the complete logging system for the trading bot.

    Creates separate log streams:
    - Console: color-coded, compact format
    - trading_{date}.log: all bot activity (daily rotation)
    - errors_{date}.log: ERROR and CRITICAL only
    - trades_{date}.log: trade decisions, orders, fills, position updates
    - signals_{date}.log: signal generation and scoring
    - audit_{date}.log: structured JSON for compliance

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_console: Enable color console logging
        enable_file: Enable general log file
        enable_audit: Enable structured JSON audit log
        enable_error_file: Enable separate error log
        enable_trade_file: Enable separate trade activity log
        enable_signal_file: Enable separate signal log
        retention_days: Days to retain old log files
    """
    global _logging_initialized

    with _logging_lock:
        if _logging_initialized:
            return
        _logging_initialized = True

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # ── Root logger ──────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, filter at handler level
    root_logger.handlers.clear()

    # ── Console handler ──────────────────────────────────────
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(ColorConsoleFormatter())
        root_logger.addHandler(console_handler)

    # ── General file handler (daily rotation) ────────────────
    if enable_file:
        try:
            general_handler = DailyRotatingFileHandler(
                log_dir=log_dir,
                prefix="trading",
                retention_days=retention_days,
            )
            general_handler.setLevel(numeric_level)
            general_handler.setFormatter(DetailedFileFormatter())
            root_logger.addHandler(general_handler)
        except Exception as e:
            print(f"Warning: Could not set up general file logging: {e}")

    # ── Error-only file handler ──────────────────────────────
    if enable_error_file:
        try:
            error_handler = DailyRotatingFileHandler(
                log_dir=log_dir,
                prefix="errors",
                retention_days=retention_days,
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(DetailedFileFormatter())
            root_logger.addHandler(error_handler)
        except Exception as e:
            print(f"Warning: Could not set up error file logging: {e}")

    # ── Trade activity logger ────────────────────────────────
    if enable_trade_file:
        try:
            trade_logger = logging.getLogger("trades")
            trade_logger.handlers.clear()
            trade_logger.setLevel(logging.DEBUG)
            trade_logger.propagate = True  # Also goes to root

            trade_handler = DailyRotatingFileHandler(
                log_dir=log_dir,
                prefix="trades",
                retention_days=retention_days,
            )
            trade_handler.setLevel(logging.DEBUG)
            trade_handler.setFormatter(DetailedFileFormatter())
            trade_logger.addHandler(trade_handler)
        except Exception as e:
            print(f"Warning: Could not set up trade file logging: {e}")

    # ── Signal logger ────────────────────────────────────────
    if enable_signal_file:
        try:
            signal_logger = logging.getLogger("signals")
            signal_logger.handlers.clear()
            signal_logger.setLevel(logging.DEBUG)
            signal_logger.propagate = True

            signal_handler = DailyRotatingFileHandler(
                log_dir=log_dir,
                prefix="signals",
                retention_days=retention_days,
            )
            signal_handler.setLevel(logging.DEBUG)
            signal_handler.setFormatter(DetailedFileFormatter())
            signal_logger.addHandler(signal_handler)
        except Exception as e:
            print(f"Warning: Could not set up signal file logging: {e}")

    # ── Audit logger (structured JSON) ───────────────────────
    if enable_audit:
        try:
            audit_log = logging.getLogger("audit")
            audit_log.handlers.clear()
            audit_log.setLevel(logging.INFO)
            audit_log.propagate = False  # Audit stays separate

            audit_handler = DailyRotatingFileHandler(
                log_dir=log_dir,
                prefix="audit",
                retention_days=90,  # Keep audit logs longer
            )
            audit_handler.setLevel(logging.INFO)
            audit_handler.setFormatter(AuditFormatter())
            audit_log.addHandler(audit_handler)
        except Exception as e:
            print(f"Warning: Could not set up audit logging: {e}")

    # ── Suppress noisy third-party loggers ───────────────────
    for noisy in [
        "urllib3", "requests", "websocket", "SmartApi",
        "smartapi", "asyncio", "charset_normalizer",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging initialized | level={log_level} | dir={log_dir} | "
        f"console={enable_console} | file={enable_file} | audit={enable_audit}"
    )


# ═══════════════════════════════════════════════════════════════
# AUDIT LOGGER
# ═══════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Structured audit logger for trade lifecycle events and compliance.
    All methods pass context via the `extra` dict so the AuditFormatter
    captures every field into the JSON output.
    Thread-safe — uses the underlying logging module locks.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("audit")

    # ── Trade lifecycle ──────────────────────────────────────

    def log_trade_decision(
        self,
        symbol: str,
        action: str,
        quantity: int,
        strategy_name: str,
        signal_confidence: float,
        entry_price: float,
        stop_loss: float,
        target: float,
        market_regime: str = "",
        reasons: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        """Log a trade decision with full context."""
        self._logger.info(
            f"TRADE_DECISION | {action} {quantity} {symbol} @ {entry_price:.2f} "
            f"| SL={stop_loss:.2f} TGT={target:.2f} | {strategy_name} ({signal_confidence:.0f}%)",
            extra={
                "audit_action": "TRADE_DECISION",
                "symbol": symbol,
                "side": action,
                "quantity": quantity,
                "strategy_name": strategy_name,
                "signal_confidence": signal_confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "market_regime": market_regime,
                "reasons": reasons or [],
                **kwargs,
            },
        )

    def log_order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        exchange: str = "",
        product_type: str = "",
        **kwargs: Any,
    ) -> None:
        """Log order placement."""
        price_str = f"@ {price:.2f}" if price else "@ MARKET"
        self._logger.info(
            f"ORDER_PLACED | {side} {quantity} {symbol} {order_type} {price_str} | id={order_id}",
            extra={
                "audit_action": "ORDER_PLACED",
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "trigger_price": trigger_price,
                "exchange": exchange,
                "product_type": product_type,
                **kwargs,
            },
        )

    def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log order execution/fill."""
        self._logger.info(
            f"ORDER_FILLED | {side} {quantity} {symbol} @ {fill_price:.2f} "
            f"| commission={commission:.2f} slippage={slippage:.4f} | id={order_id}",
            extra={
                "audit_action": "ORDER_FILLED",
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "commission": commission,
                "slippage": slippage,
                **kwargs,
            },
        )

    def log_order_cancelled(
        self,
        order_id: str,
        symbol: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        """Log order cancellation."""
        self._logger.info(
            f"ORDER_CANCELLED | {symbol} | reason={reason} | id={order_id}",
            extra={
                "audit_action": "ORDER_CANCELLED",
                "order_id": order_id,
                "symbol": symbol,
                "reason": reason,
                **kwargs,
            },
        )

    def log_order_rejected(
        self,
        order_id: str,
        symbol: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        """Log order rejection."""
        self._logger.warning(
            f"ORDER_REJECTED | {symbol} | reason={reason} | id={order_id}",
            extra={
                "audit_action": "ORDER_REJECTED",
                "order_id": order_id,
                "symbol": symbol,
                "reason": reason,
                **kwargs,
            },
        )

    # ── Position lifecycle ───────────────────────────────────

    def log_position_opened(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        strategy_name: str = "",
        exchange: str = "",
        segment: str = "",
        **kwargs: Any,
    ) -> None:
        """Log new position opened."""
        self._logger.info(
            f"POSITION_OPENED | {side} {quantity} {symbol} @ {entry_price:.2f} "
            f"| SL={stop_loss:.2f} TGT={target:.2f} | {strategy_name}",
            extra={
                "audit_action": "POSITION_OPENED",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "strategy_name": strategy_name,
                "exchange": exchange,
                "segment": segment,
                **kwargs,
            },
        )

    def log_position_closed(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str = "",
        holding_minutes: float = 0,
        strategy_name: str = "",
        **kwargs: Any,
    ) -> None:
        """Log position closed."""
        pnl_emoji = "💰" if pnl >= 0 else "💸"
        self._logger.info(
            f"POSITION_CLOSED | {side} {quantity} {symbol} | "
            f"entry={entry_price:.2f} exit={exit_price:.2f} | "
            f"{pnl_emoji} PnL=₹{pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"reason={exit_reason} | held={holding_minutes:.0f}m",
            extra={
                "audit_action": "POSITION_CLOSED",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
                "holding_minutes": holding_minutes,
                "strategy_name": strategy_name,
                **kwargs,
            },
        )

    def log_stop_loss_updated(
        self,
        symbol: str,
        old_sl: float,
        new_sl: float,
        current_price: float,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        """Log stop-loss modification."""
        self._logger.info(
            f"SL_UPDATED | {symbol} | old={old_sl:.2f} new={new_sl:.2f} "
            f"| price={current_price:.2f} | {reason}",
            extra={
                "audit_action": "SL_UPDATED",
                "symbol": symbol,
                "old_stop_loss": old_sl,
                "new_stop_loss": new_sl,
                "current_price": current_price,
                "reason": reason,
                **kwargs,
            },
        )

    # ── Risk management ──────────────────────────────────────

    def log_risk_check(
        self,
        symbol: str,
        action: str,
        quantity: int,
        result: str,
        reason: str = "",
        risk_score: float = 0.0,
        position_size_pct: float = 0.0,
        kelly_fraction: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log risk management check."""
        level = logging.INFO if result == "APPROVED" else logging.WARNING
        self._logger.log(
            level,
            f"RISK_CHECK | {action} {quantity} {symbol} | "
            f"result={result} | risk_score={risk_score:.1f} "
            f"| kelly={kelly_fraction:.2%} | {reason}",
            extra={
                "audit_action": "RISK_CHECK",
                "symbol": symbol,
                "side": action,
                "quantity": quantity,
                "result": result,
                "reason": reason,
                "risk_score": risk_score,
                "position_size_pct": position_size_pct,
                "kelly_fraction": kelly_fraction,
                **kwargs,
            },
        )

    def log_daily_limit_hit(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        **kwargs: Any,
    ) -> None:
        """Log daily limit breach."""
        self._logger.warning(
            f"DAILY_LIMIT | {limit_type} | "
            f"current={current_value:.2f} limit={limit_value:.2f}",
            extra={
                "audit_action": "DAILY_LIMIT_HIT",
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value,
                **kwargs,
            },
        )

    def log_circuit_breaker(
        self,
        trigger: str,
        action_taken: str,
        **kwargs: Any,
    ) -> None:
        """Log circuit breaker activation."""
        self._logger.critical(
            f"CIRCUIT_BREAKER | trigger={trigger} | action={action_taken}",
            extra={
                "audit_action": "CIRCUIT_BREAKER",
                "trigger": trigger,
                "action_taken": action_taken,
                **kwargs,
            },
        )

    # ── Scan & signal events ─────────────────────────────────

    def log_scan_completed(
        self,
        segment: str,
        symbols_scanned: int,
        signals_found: int,
        duration_seconds: float,
        **kwargs: Any,
    ) -> None:
        """Log market scan completion."""
        self._logger.info(
            f"SCAN_COMPLETE | {segment} | scanned={symbols_scanned} "
            f"signals={signals_found} | {duration_seconds:.1f}s",
            extra={
                "audit_action": "SCAN_COMPLETE",
                "segment": segment,
                "symbols_scanned": symbols_scanned,
                "signals_found": signals_found,
                "duration_seconds": duration_seconds,
                **kwargs,
            },
        )

    def log_signal_generated(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        strategy_name: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        **kwargs: Any,
    ) -> None:
        """Log signal generation."""
        signal_logger = logging.getLogger("signals")
        signal_logger.info(
            f"SIGNAL | {signal_type} {symbol} | conf={confidence:.0f}% "
            f"| entry={entry_price:.2f} SL={stop_loss:.2f} TGT={target:.2f} "
            f"| {strategy_name}",
            extra={
                "audit_action": "SIGNAL_GENERATED",
                "symbol": symbol,
                "signal_type": signal_type,
                "confidence": confidence,
                "strategy_name": strategy_name,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                **kwargs,
            },
        )

    # ── System events ────────────────────────────────────────

    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        **kwargs: Any,
    ) -> None:
        """Log system-level events (startup, shutdown, connection, etc.)."""
        level = getattr(logging, severity.upper(), logging.INFO)
        self._logger.log(
            level,
            f"SYSTEM | {event_type} | {description}",
            extra={
                "audit_action": "SYSTEM_EVENT",
                "event_type": event_type,
                "severity": severity,
                **kwargs,
            },
        )

    def log_connection_event(
        self,
        service: str,
        status: str,
        details: str = "",
        **kwargs: Any,
    ) -> None:
        """Log connection status changes (broker API, WebSocket, etc.)."""
        level = logging.INFO if status in ("connected", "restored") else logging.WARNING
        self._logger.log(
            level,
            f"CONNECTION | {service} | {status} | {details}",
            extra={
                "audit_action": "CONNECTION_EVENT",
                "service": service,
                "status": status,
                "details": details,
                **kwargs,
            },
        )

    def log_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log market regime change."""
        self._logger.info(
            f"REGIME_CHANGE | {old_regime} → {new_regime} | conf={confidence:.0f}%",
            extra={
                "audit_action": "REGIME_CHANGE",
                "old_regime": old_regime,
                "new_regime": new_regime,
                "confidence": confidence,
                **kwargs,
            },
        )

    def log_learning_update(
        self,
        update_type: str,
        parameters_changed: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Log self-learning parameter updates."""
        self._logger.info(
            f"LEARNING | {update_type} | params={json.dumps(parameters_changed, default=str)}",
            extra={
                "audit_action": "LEARNING_UPDATE",
                "update_type": update_type,
                "parameters_changed": parameters_changed,
                **kwargs,
            },
        )

    def log_compliance_check(
        self,
        check_type: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log SEBI compliance checks."""
        self._logger.info(
            f"COMPLIANCE | {check_type} | result={result}",
            extra={
                "audit_action": "COMPLIANCE_CHECK",
                "check_type": check_type,
                "result": result,
                "details": details or {},
                **kwargs,
            },
        )

    # ── Daily summary ────────────────────────────────────────

    def log_daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        strategies_used: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> None:
        """Log end-of-day summary."""
        self._logger.info(
            f"DAILY_SUMMARY | trades={total_trades} W={winning_trades} L={losing_trades} "
            f"| PnL=₹{total_pnl:.2f} | WR={win_rate:.1f}% PF={profit_factor:.2f} "
            f"| MaxDD={max_drawdown:.2f}%",
            extra={
                "audit_action": "DAILY_SUMMARY",
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "strategies_used": strategies_used or {},
                **kwargs,
            },
        )


# ═══════════════════════════════════════════════════════════════
# TRADE-SPECIFIC LOGGERS
# ═══════════════════════════════════════════════════════════════

class TradeLogger:
    """
    Convenience logger for trade-related components.
    Writes to both the component's own logger and the dedicated trades log.
    """

    def __init__(self, component_name: str) -> None:
        self._logger = logging.getLogger(component_name)
        self._trade_logger = logging.getLogger("trades")

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, **kwargs)

    def trade(self, msg: str, **extra: Any) -> None:
        """Log to both the component logger and the trades log."""
        self._logger.info(msg)
        self._trade_logger.info(msg, extra=extra)

    def signal(self, msg: str, **extra: Any) -> None:
        """Log to both the component logger and the signals log."""
        self._logger.info(msg)
        logging.getLogger("signals").info(msg, extra=extra)


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE TIMING
# ═══════════════════════════════════════════════════════════════

@contextmanager
def log_performance(
    operation: str,
    logger_name: str = "performance",
    warn_threshold_ms: float = 1000.0,
):
    """
    Context manager to measure and log execution time.

    Usage:
        with log_performance("scan_nifty_50"):
            scanner.scan()

    Logs a warning if execution exceeds warn_threshold_ms.
    """
    perf_logger = logging.getLogger(logger_name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        msg = f"{operation} completed in {elapsed_ms:.1f}ms"
        if elapsed_ms > warn_threshold_ms:
            perf_logger.warning(f"SLOW | {msg}")
        else:
            perf_logger.debug(msg)


# ═══════════════════════════════════════════════════════════════
# GLOBAL INSTANCES & HELPERS
# ═══════════════════════════════════════════════════════════════

# Global audit logger — importable from anywhere
audit_logger = AuditLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a standard logger instance.

    Usage:
        from trading_bot.core.logging import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def get_trade_logger(component_name: str) -> TradeLogger:
    """
    Get a trade-aware logger that writes to both component and trade logs.

    Usage:
        from trading_bot.core.logging import get_trade_logger
        logger = get_trade_logger("risk_manager")
        logger.trade("Position sized at 5% of capital", symbol="RELIANCE")
    """
    return TradeLogger(component_name)