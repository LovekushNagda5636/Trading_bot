"""
Backtesting Engine - Strategy backtesting and validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..core.models import Trade, Position, Signal
from ..strategies.base import BaseStrategy
from ..risk.risk_manager import RiskManager
from .performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005   # 0.05% slippage
    market_impact: float = 0.0001  # 0.01% market impact
    interest_rate: float = 0.06  # 6% annual interest rate
    
    # Risk management
    max_positions: int = 5
    position_size_method: str = "fixed_risk"  # fixed_risk, fixed_amount, kelly
    risk_per_trade: float = 0.01  # 1% risk per trade
    
    # Execution settings
    fill_method: str = "next_bar"  # next_bar, same_bar, realistic
    benchmark: str = "NIFTY50"


@dataclass
class BacktestResult:
    """Backtesting results."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: List[Position] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    
    # Summary statistics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0


class BacktestEngine:
    """
    Comprehensive backtesting engine for strategy validation.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Backtesting state
        self.current_capital = config.initial_capital
        self.current_positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Risk management
        risk_config = {
            'risk_limits': {
                'max_positions': config.max_positions,
                'max_position_risk': config.risk_per_trade,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.20
            }
        }
        self.risk_manager = RiskManager(risk_config)
        
        logger.info(f"Backtest engine initialized: {config.start_date} to {config.end_date}")
    
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame) -> BacktestResult:
        """
        Run complete backtesting for a strategy.
        
        Args:
            strategy: Strategy to backtest
            data: Historical OHLCV data
            
        Returns:
            BacktestResult with comprehensive results
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Reset state
        self._reset_state()
        
        # Prepare data
        data = self._prepare_data(data)
        
        # Calculate indicators
        data_with_indicators = strategy.indicators(data)
        
        # Main backtesting loop
        for i in range(len(data_with_indicators)):
            current_bar = data_with_indicators.iloc[i]
            
            # Update positions with current prices
            self._update_positions(current_bar)
            
            # Check exit conditions for existing positions
            self._check_exits(strategy, data_with_indicators, i)
            
            # Generate new signals
            if strategy.should_enter(data_with_indicators, i)[0]:
                self._process_entry_signal(strategy, data_with_indicators, i)
            
            # Update equity curve
            self._update_equity_curve(current_bar)
            
            # Update risk metrics
            self._update_risk_metrics()
        
        # Close remaining positions
        self._close_all_positions(data_with_indicators.iloc[-1])
        
        # Generate results
        result = self._generate_results(data_with_indicators)
        
        logger.info(f"Backtest completed: {len(self.trades)} trades, "
                   f"{result.total_return:.2%} return")
        
        return result
    
    def _reset_state(self) -> None:
        """Reset backtesting state."""
        self.current_capital = self.config.initial_capital
        self.current_positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Prepared data
        """
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Sort by date
        data = data.sort_index()
        
        # Filter date range
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        data = data.loc[start_date:end_date]
        
        # Add returns
        data['returns'] = data['close'].pct_change()
        
        return data
    
    def _update_positions(self, current_bar: pd.Series) -> None:
        """Update position values with current prices."""
        for symbol, position in self.current_positions.items():
            if symbol in current_bar.name or 'close' in current_bar:
                # Update position with current price
                current_price = current_bar['close']
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.average_price) * position.quantity
    
    def _check_exits(self, strategy: BaseStrategy, data: pd.DataFrame, current_idx: int) -> None:
        """Check exit conditions for existing positions."""
        positions_to_close = []
        
        for symbol, position in self.current_positions.items():
            should_exit, reason = strategy.should_exit(
                data, current_idx, position.average_price, position.entry_time
            )
            
            if should_exit:
                positions_to_close.append((symbol, reason))
        
        # Close positions
        for symbol, reason in positions_to_close:
            self._close_position(symbol, data.iloc[current_idx], reason)
    
    def _process_entry_signal(self, strategy: BaseStrategy, data: pd.DataFrame, 
                            current_idx: int) -> None:
        """Process entry signal and create position."""
        current_bar = data.iloc[current_idx]
        
        # Generate signal
        signals = strategy.generate_signals(data.iloc[current_idx-10:current_idx+1])
        
        if not signals:
            return
        
        signal = signals[-1]  # Use latest signal
        
        # Check risk management
        current_positions_list = list(self.current_positions.values())
        can_place, reason, quantity = self.risk_manager.can_place_order(
            signal, self.current_capital, current_positions_list
        )
        
        if not can_place or quantity <= 0:
            logger.debug(f"Order rejected: {reason}")
            return
        
        # Calculate entry price (with slippage)
        entry_price = self._calculate_entry_price(signal, current_bar)
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            quantity=quantity if signal.action == "BUY" else -quantity,
            average_price=entry_price,
            entry_time=current_bar.name,
            current_price=entry_price,
            unrealized_pnl=0.0
        )
        
        self.current_positions[signal.symbol] = position
        
        # Update capital (subtract position value)
        position_value = abs(quantity * entry_price)
        commission = position_value * self.config.commission
        self.current_capital -= commission
        
        logger.debug(f"Position opened: {signal.symbol} {quantity} @ {entry_price}")
    
    def _close_position(self, symbol: str, current_bar: pd.Series, reason: str) -> None:
        """Close a position and record trade."""
        if symbol not in self.current_positions:
            return
        
        position = self.current_positions[symbol]
        
        # Calculate exit price (with slippage)
        exit_price = self._calculate_exit_price(position, current_bar)
        
        # Calculate P&L
        if position.quantity > 0:  # Long position
            pnl = (exit_price - position.average_price) * position.quantity
        else:  # Short position
            pnl = (position.average_price - exit_price) * abs(position.quantity)
        
        # Subtract commission
        position_value = abs(position.quantity * exit_price)
        commission = position_value * self.config.commission
        pnl -= commission
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=current_bar.name,
            entry_price=position.average_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            commission=commission,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # Update capital
        self.current_capital += pnl
        
        # Remove position
        del self.current_positions[symbol]
        
        logger.debug(f"Position closed: {symbol} P&L: {pnl:.2f} ({reason})")
    
    def _calculate_entry_price(self, signal: Signal, current_bar: pd.Series) -> float:
        """Calculate realistic entry price with slippage."""
        base_price = signal.price
        
        if self.config.fill_method == "next_bar":
            # Use next bar's open (more realistic)
            base_price = current_bar['open']
        elif self.config.fill_method == "same_bar":
            # Use signal price (less realistic)
            base_price = signal.price
        
        # Apply slippage
        if signal.action == "BUY":
            return base_price * (1 + self.config.slippage)
        else:
            return base_price * (1 - self.config.slippage)
    
    def _calculate_exit_price(self, position: Position, current_bar: pd.Series) -> float:
        """Calculate realistic exit price with slippage."""
        base_price = current_bar['close']
        
        # Apply slippage
        if position.quantity > 0:  # Long position (selling)
            return base_price * (1 - self.config.slippage)
        else:  # Short position (buying)
            return base_price * (1 + self.config.slippage)
    
    def _update_equity_curve(self, current_bar: pd.Series) -> None:
        """Update equity curve with current portfolio value."""
        # Calculate total position value
        position_value = 0.0
        for position in self.current_positions.values():
            position_value += position.unrealized_pnl
        
        total_equity = self.current_capital + position_value
        
        self.equity_curve.append({
            'timestamp': current_bar.name,
            'equity': total_equity,
            'cash': self.current_capital,
            'positions_value': position_value,
            'num_positions': len(self.current_positions)
        })
    
    def _update_risk_metrics(self) -> None:
        """Update risk management metrics."""
        if self.equity_curve:
            current_equity = self.equity_curve[-1]['equity']
            positions_list = list(self.current_positions.values())
            daily_pnl = current_equity - self.config.initial_capital
            
            self.risk_manager.update_metrics(current_equity, positions_list, daily_pnl)
    
    def _close_all_positions(self, final_bar: pd.Series) -> None:
        """Close all remaining positions at the end of backtest."""
        symbols_to_close = list(self.current_positions.keys())
        
        for symbol in symbols_to_close:
            self._close_position(symbol, final_bar, "End of backtest")
    
    def _generate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Generate comprehensive backtest results."""
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        # Calculate performance metrics
        performance_metrics = self.performance_analyzer.calculate_metrics(
            self.trades, equity_df, self.config.initial_capital
        )
        
        # Calculate drawdown series
        if not equity_df.empty:
            equity_series = equity_df['equity']
            peak_series = equity_series.expanding().max()
            drawdown_series = (equity_series - peak_series) / peak_series
        else:
            drawdown_series = pd.Series()
        
        # Create result object
        result = BacktestResult(
            trades=self.trades,
            equity_curve=equity_df,
            positions=list(self.current_positions.values()),
            performance_metrics=performance_metrics,
            drawdown_series=drawdown_series
        )
        
        # Fill summary statistics
        if performance_metrics:
            result.total_return = performance_metrics.get('total_return', 0.0)
            result.annual_return = performance_metrics.get('annual_return', 0.0)
            result.sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            result.max_drawdown = performance_metrics.get('max_drawdown', 0.0)
            result.win_rate = performance_metrics.get('win_rate', 0.0)
            result.profit_factor = performance_metrics.get('profit_factor', 0.0)
            result.total_trades = len(self.trades)
        
        return result
    
    def run_walk_forward_analysis(self, strategy: BaseStrategy, data: pd.DataFrame,
                                 train_period: int = 252, test_period: int = 63) -> Dict:
        """
        Run walk-forward analysis for strategy validation.
        
        Args:
            strategy: Strategy to test
            data: Historical data
            train_period: Training period in days
            test_period: Testing period in days
            
        Returns:
            Walk-forward analysis results
        """
        logger.info("Starting walk-forward analysis")
        
        results = []
        start_idx = train_period
        
        while start_idx + test_period < len(data):
            # Training data
            train_data = data.iloc[start_idx-train_period:start_idx]
            
            # Test data
            test_data = data.iloc[start_idx:start_idx+test_period]
            
            # Run backtest on test period
            test_config = BacktestConfig(
                start_date=test_data.index[0].strftime('%Y-%m-%d'),
                end_date=test_data.index[-1].strftime('%Y-%m-%d'),
                initial_capital=self.config.initial_capital
            )
            
            test_engine = BacktestEngine(test_config)
            result = test_engine.run_backtest(strategy, test_data)
            
            results.append({
                'period_start': test_data.index[0],
                'period_end': test_data.index[-1],
                'return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown,
                'trades': result.total_trades
            })
            
            start_idx += test_period
        
        # Aggregate results
        if results:
            avg_return = np.mean([r['return'] for r in results])
            avg_sharpe = np.mean([r['sharpe'] for r in results])
            avg_dd = np.mean([r['max_dd'] for r in results])
            consistency = len([r for r in results if r['return'] > 0]) / len(results)
        else:
            avg_return = avg_sharpe = avg_dd = consistency = 0.0
        
        return {
            'periods': results,
            'summary': {
                'average_return': avg_return,
                'average_sharpe': avg_sharpe,
                'average_drawdown': avg_dd,
                'consistency': consistency,
                'total_periods': len(results)
            }
        }