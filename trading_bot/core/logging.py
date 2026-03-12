"""
Simple logging system for the trading bot.
Provides basic logging without external dependencies.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_audit: bool = True
) -> None:
    """
    Set up logging for the trading bot.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_audit: Enable audit logging for compliance
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, log_level.upper())
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler for general logs
    if enable_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_path / "trading_bot.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
    
    # Audit handler for compliance logging
    if enable_audit:
        try:
            audit_handler = logging.handlers.RotatingFileHandler(
                log_path / "audit.log",
                maxBytes=50*1024*1024,  # 50MB
                backupCount=10
            )
            audit_handler.setLevel(logging.INFO)
            audit_formatter = AuditFormatter()
            audit_handler.setFormatter(audit_formatter)
            
            # Create audit logger
            audit_logger = logging.getLogger("audit")
            audit_logger.addHandler(audit_handler)
            audit_logger.setLevel(logging.INFO)
            audit_logger.propagate = False
        except Exception as e:
            print(f"Warning: Could not set up audit logging: {e}")


class AuditFormatter(logging.Formatter):
    """Custom formatter for audit logs to ensure regulatory compliance."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format audit log record."""
        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "thread": record.thread,
            "process": record.process
        }
        
        # Add extra fields if present
        if hasattr(record, 'trade_id'):
            audit_data['trade_id'] = record.trade_id
        if hasattr(record, 'order_id'):
            audit_data['order_id'] = record.order_id
        if hasattr(record, 'symbol'):
            audit_data['symbol'] = record.symbol
        if hasattr(record, 'user_id'):
            audit_data['user_id'] = record.user_id
        if hasattr(record, 'action'):
            audit_data['action'] = record.action
        
        return json.dumps(audit_data)


class AuditLogger:
    """
    Specialized logger for audit trails and regulatory compliance.
    Records all trading-related activities with structured data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
    
    def log_trade_decision(
        self,
        symbol: str,
        action: str,
        quantity: int,
        strategy_id: str,
        decision_id: str,
        **kwargs
    ) -> None:
        """Log a trading decision for audit trail."""
        self.logger.info(
            "Trade decision made",
            extra={
                'action': 'TRADE_DECISION',
                'symbol': symbol,
                'trade_action': action,
                'quantity': quantity,
                'strategy_id': strategy_id,
                'decision_id': decision_id,
                **kwargs
            }
        )
    
    def log_order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log order placement for audit trail."""
        self.logger.info(
            "Order placed",
            extra={
                'action': 'ORDER_PLACED',
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                **kwargs
            }
        )
    
    def log_order_filled(
        self,
        order_id: str,
        trade_id: str,
        symbol: str,
        quantity: int,
        price: float,
        commission: float,
        **kwargs
    ) -> None:
        """Log order execution for audit trail."""
        self.logger.info(
            "Order filled",
            extra={
                'action': 'ORDER_FILLED',
                'order_id': order_id,
                'trade_id': trade_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                **kwargs
            }
        )
    
    def log_risk_check(
        self,
        symbol: str,
        action: str,
        quantity: int,
        risk_result: str,
        risk_score: float,
        **kwargs
    ) -> None:
        """Log risk management decisions."""
        self.logger.info(
            "Risk check performed",
            extra={
                'action': 'RISK_CHECK',
                'symbol': symbol,
                'trade_action': action,
                'quantity': quantity,
                'risk_result': risk_result,
                'risk_score': risk_score,
                **kwargs
            }
        )
    
    def log_position_update(
        self,
        symbol: str,
        old_quantity: int,
        new_quantity: int,
        price: float,
        pnl: float,
        **kwargs
    ) -> None:
        """Log position updates."""
        self.logger.info(
            "Position updated",
            extra={
                'action': 'POSITION_UPDATE',
                'symbol': symbol,
                'old_quantity': old_quantity,
                'new_quantity': new_quantity,
                'price': price,
                'pnl': pnl,
                **kwargs
            }
        )
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        **kwargs
    ) -> None:
        """Log system events."""
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(
            description,
            extra={
                'action': 'SYSTEM_EVENT',
                'event_type': event_type,
                'severity': severity,
                **kwargs
            }
        )
    
    def log_compliance_check(
        self,
        check_type: str,
        result: str,
        details: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log SEBI compliance checks."""
        self.logger.info(
            "Compliance check performed",
            extra={
                'action': 'COMPLIANCE_CHECK',
                'check_type': check_type,
                'result': result,
                'details': details,
                **kwargs
            }
        )


# Global audit logger instance
audit_logger = AuditLogger()


def get_logger(name: str):
    """Get a logger instance."""
    return logging.getLogger(name)