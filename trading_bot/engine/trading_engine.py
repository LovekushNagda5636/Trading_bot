"""
Trading Engine - Main orchestrator for live trading operations.
"""

import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, time
import pandas as pd

from ..core.events import EventBus, MarketDataEvent, SignalEvent, OrderEvent
from ..core.models import Signal, Position, Trade
from ..data.live_data_feed import LiveDataFeed, LiveDataConfig
from ..data.angel_one_data_feed import AngelOneDataFeed, AngelOneDataConfig
from ..execution.broker_interface import BrokerInterface
from ..execution.angel_one_broker import AngelOneBroker
from ..risk.risk_manager import RiskManager

# Conditional import for Zerodha (only if kiteconnect is available)
try:
    from ..execution.zerodha_broker import ZerodhaBroker
    ZERODHA_AVAILABLE = True
except ImportError:
    ZERODHA_AVAILABLE = False
    ZerodhaBroker = None

# Conditional import for strategy registry
try:
    import sys
    import os
    # Add the root directory to path to import strategies
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    from strategies.base import BaseStrategy
    from strategies.registry import strategy_registry
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    STRATEGIES_AVAILABLE = False
    strategy_registry = None
    BaseStrategy = None

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine that orchestrates all components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.event_bus = EventBus()
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.active_strategies = {}
        self.current_positions = {}
        self.daily_trades = []
        self.account_balance = 0.0
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info("Trading engine initialized")
    
    def _initialize_components(self):
        """Initialize all trading components."""
        try:
            # Initialize data feed
            data_config_dict = self.config.get('data_feed', {})
            data_provider = data_config_dict.get('data_provider', 'zerodha')
            
            if data_provider == 'angel_one':
                data_config = AngelOneDataConfig(**data_config_dict)
                self.data_feed = AngelOneDataFeed(data_config, self.event_bus)
            else:
                data_config = LiveDataConfig(**data_config_dict)
                self.data_feed = LiveDataFeed(data_config, self.event_bus)
            
            # Initialize broker
            broker_config = self.config.get('broker', {})
            broker_provider = broker_config.get('provider', 'zerodha')
            
            if broker_provider == 'angel_one':
                self.broker = AngelOneBroker(broker_config)
            elif broker_provider == 'zerodha':
                if not ZERODHA_AVAILABLE:
                    raise ValueError("Zerodha support requires 'kiteconnect' package. Install with: pip install kiteconnect")
                self.broker = ZerodhaBroker(broker_config)
            else:
                raise ValueError(f"Unsupported broker provider: {broker_provider}")
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config.get('risk', {}))
            
            # Set up event handlers
            self._setup_event_handlers()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_event_handlers(self):
        """Set up event handlers for different event types."""
        self.event_bus.subscribe('market_data', self._handle_market_data)
        self.event_bus.subscribe('signal', self._handle_signal)
        self.event_bus.subscribe('order', self._handle_order)
        self.event_bus.subscribe('trade', self._handle_trade)
    
    def start(self) -> bool:
        """
        Start the trading engine.
        
        Returns:
            bool: True if started successfully
        """
        try:
            logger.info("Starting trading engine...")
            
            # Connect to broker
            if not self.broker.connect():
                logger.error("Failed to connect to broker")
                return False
            
            # Connect to data feed
            if not self.data_feed.connect():
                logger.error("Failed to connect to data feed")
                return False
            
            # Load account information
            self._load_account_info()
            
            # Load active strategies
            self._load_strategies()
            
            # Subscribe to market data
            self._subscribe_to_data()
            
            # Start main trading loop
            self.is_running = True
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            logger.info("Trading engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            return False
    
    def stop(self):
        """Stop the trading engine."""
        try:
            logger.info("Stopping trading engine...")
            
            self.is_running = False
            self.stop_event.set()
            
            # Close all positions (optional)
            if self.config.get('close_positions_on_stop', False):
                self._close_all_positions()
            
            # Disconnect components
            self.data_feed.disconnect()
            self.broker.disconnect()
            
            # Wait for trading thread to finish
            if hasattr(self, 'trading_thread') and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            logger.info("Trading engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
    
    def _load_account_info(self):
        """Load account information from broker."""
        try:
            funds = self.broker.get_funds()
            self.account_balance = funds.get('available_cash', 0.0)
            
            positions = self.broker.get_positions()
            self.current_positions = {pos.symbol: pos for pos in positions}
            
            logger.info(f"Account balance: ₹{self.account_balance:,.2f}")
            logger.info(f"Current positions: {len(self.current_positions)}")
            
        except Exception as e:
            logger.error(f"Failed to load account info: {e}")
    
    def _load_strategies(self):
        """Load and initialize active strategies."""
        try:
            strategy_configs = self.config.get('strategies', [])
            
            for strategy_config in strategy_configs:
                strategy_name = strategy_config.get('name')
                strategy_params = strategy_config.get('params', {})
                enabled = strategy_config.get('enabled', True)
                
                if not enabled:
                    continue
                
                # Create strategy instance
                if STRATEGIES_AVAILABLE and strategy_registry:
                    strategy = strategy_registry.create_strategy(strategy_name, strategy_params)
                else:
                    logger.warning(f"Strategy registry not available, skipping strategy: {strategy_name}")
                    continue
                
                if strategy:
                    self.active_strategies[strategy_name] = {
                        'instance': strategy,
                        'config': strategy_config,
                        'last_signal_time': None,
                        'positions': [],
                        'daily_pnl': 0.0
                    }
                    logger.info(f"Loaded strategy: {strategy_name}")
                else:
                    logger.error(f"Failed to load strategy: {strategy_name}")
            
            logger.info(f"Loaded {len(self.active_strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
    
    def _subscribe_to_data(self):
        """Subscribe to market data for all required instruments."""
        try:
            # Collect all instruments from strategies
            instruments = set()
            
            for strategy_info in self.active_strategies.values():
                strategy_config = strategy_info['config']
                strategy_instruments = strategy_config.get('instruments', [])
                instruments.update(strategy_instruments)
            
            if instruments:
                success = self.data_feed.subscribe(list(instruments), mode="full")
                if success:
                    logger.info(f"Subscribed to {len(instruments)} instruments")
                else:
                    logger.error("Failed to subscribe to market data")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to data: {e}")
    
    def _trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check market hours
                if not self._is_market_hours():
                    self.stop_event.wait(60)  # Check every minute
                    continue
                
                # Update account information
                self._update_account_info()
                
                # Process strategies
                self._process_strategies()
                
                # Update risk metrics
                self._update_risk_metrics()
                
                # Check for emergency conditions
                self._check_emergency_conditions()
                
                # Sleep for a short interval
                self.stop_event.wait(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.stop_event.wait(5)  # Wait longer on error
        
        logger.info("Trading loop stopped")
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now().time()
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
        
        return market_open <= now <= market_close
    
    def _update_account_info(self):
        """Update account information periodically."""
        try:
            # Update every 30 seconds
            if not hasattr(self, '_last_account_update'):
                self._last_account_update = datetime.now()
            
            if (datetime.now() - self._last_account_update).seconds >= 30:
                funds = self.broker.get_funds()
                self.account_balance = funds.get('available_cash', 0.0)
                
                positions = self.broker.get_positions()
                self.current_positions = {pos.symbol: pos for pos in positions}
                
                self._last_account_update = datetime.now()
        
        except Exception as e:
            logger.error(f"Failed to update account info: {e}")
    
    def _process_strategies(self):
        """Process all active strategies."""
        for strategy_name, strategy_info in self.active_strategies.items():
            try:
                strategy = strategy_info['instance']
                
                # Get latest data for strategy instruments
                strategy_config = strategy_info['config']
                instruments = strategy_config.get('instruments', [])
                
                for instrument in instruments:
                    # Get OHLC data
                    ohlc_data = self.data_feed.get_ohlc_data(instrument, count=100)
                    
                    if len(ohlc_data) < 20:  # Need sufficient data
                        continue
                    
                    # Convert to DataFrame
                    df = self._ohlc_to_dataframe(ohlc_data)
                    
                    # Check for entry signals
                    current_idx = len(df) - 1
                    should_enter, reason = strategy.should_enter(df, current_idx)
                    
                    if should_enter:
                        # Generate signal
                        signals = strategy.generate_signals(df.tail(10))
                        
                        if signals:
                            latest_signal = signals[-1]
                            self._emit_signal(latest_signal, strategy_name)
                    
                    # Check exit conditions for existing positions
                    self._check_strategy_exits(strategy, df, strategy_name)
            
            except Exception as e:
                logger.error(f"Error processing strategy {strategy_name}: {e}")
    
    def _ohlc_to_dataframe(self, ohlc_data: List) -> pd.DataFrame:
        """Convert OHLC data to DataFrame."""
        data = []
        for ohlc in ohlc_data:
            data.append({
                'timestamp': ohlc.timestamp,
                'open': ohlc.open_price,
                'high': ohlc.high_price,
                'low': ohlc.low_price,
                'close': ohlc.close_price,
                'volume': ohlc.volume
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def _check_strategy_exits(self, strategy: BaseStrategy, df: pd.DataFrame, 
                            strategy_name: str):
        """Check exit conditions for strategy positions."""
        strategy_positions = [
            pos for pos in self.current_positions.values()
            if pos.strategy_name == strategy_name
        ]
        
        for position in strategy_positions:
            current_idx = len(df) - 1
            should_exit, reason = strategy.should_exit(
                df, current_idx, position.average_price, position.entry_time
            )
            
            if should_exit:
                self._close_position(position, reason)
    
    def _emit_signal(self, signal: Signal, strategy_name: str):
        """Emit trading signal."""
        signal.strategy_name = strategy_name
        
        self.event_bus.emit(SignalEvent(
            signal=signal,
            timestamp=datetime.now()
        ))
    
    def _handle_market_data(self, event: MarketDataEvent):
        """Handle market data events."""
        # Market data is automatically processed by strategies
        # Additional processing can be added here if needed
        pass
    
    def _handle_signal(self, event: SignalEvent):
        """Handle trading signals."""
        try:
            signal = event.signal
            
            # Risk management check
            positions_list = list(self.current_positions.values())
            can_place, reason, quantity = self.risk_manager.can_place_order(
                signal, self.account_balance, positions_list
            )
            
            if not can_place:
                logger.info(f"Signal rejected: {reason}")
                return
            
            # Place order
            self._place_order(signal, quantity)
            
        except Exception as e:
            logger.error(f"Error handling signal: {e}")
    
    def _place_order(self, signal: Signal, quantity: int):
        """Place order based on signal."""
        try:
            from ..execution.broker_interface import BrokerOrder, OrderType
            
            # Create broker order
            order = BrokerOrder(
                order_id="",  # Will be assigned by broker
                symbol=signal.symbol,
                exchange="NSE",  # Default to NSE
                transaction_type=signal.action,
                order_type=OrderType.MARKET,  # Use market orders for now
                quantity=quantity,
                price=signal.price,
                trigger_price=signal.stop_loss,
                tag=f"strategy_{signal.strategy_name}"
            )
            
            # Place order with broker
            result = self.broker.place_order(order)
            
            if result.get('status') == 'SUCCESS':
                logger.info(f"Order placed: {signal.symbol} {signal.action} {quantity}")
                
                # Emit order event
                self.event_bus.emit(OrderEvent(
                    order_id=result.get('order_id'),
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=quantity,
                    price=signal.price,
                    status='PLACED',
                    timestamp=datetime.now()
                ))
            else:
                logger.error(f"Order failed: {result.get('message')}")
        
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
    
    def _handle_order(self, event: OrderEvent):
        """Handle order events."""
        # Order tracking and management
        logger.info(f"Order event: {event.symbol} {event.action} {event.status}")
    
    def _handle_trade(self, event):
        """Handle trade events."""
        # Trade tracking and performance calculation
        pass
    
    def _close_position(self, position: Position, reason: str):
        """Close a position."""
        try:
            from ..execution.broker_interface import BrokerOrder, OrderType
            
            # Determine transaction type (opposite of position)
            transaction_type = "SELL" if position.quantity > 0 else "BUY"
            
            # Create close order
            order = BrokerOrder(
                order_id="",
                symbol=position.symbol,
                exchange="NSE",
                transaction_type=transaction_type,
                order_type=OrderType.MARKET,
                quantity=abs(position.quantity),
                price=0,  # Market order
                tag=f"close_{reason}"
            )
            
            # Place order
            result = self.broker.place_order(order)
            
            if result.get('status') == 'SUCCESS':
                logger.info(f"Position closed: {position.symbol} - {reason}")
            else:
                logger.error(f"Failed to close position: {result.get('message')}")
        
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _close_all_positions(self):
        """Close all open positions."""
        for position in self.current_positions.values():
            self._close_position(position, "Engine shutdown")
    
    def _update_risk_metrics(self):
        """Update risk management metrics."""
        try:
            positions_list = list(self.current_positions.values())
            self.risk_manager.update_metrics(
                self.account_balance, positions_list, self.daily_pnl
            )
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions."""
        if self.risk_manager.emergency_stop:
            logger.critical("EMERGENCY STOP TRIGGERED - Closing all positions")
            self._close_all_positions()
            self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'running': self.is_running,
            'account_balance': self.account_balance,
            'positions': len(self.current_positions),
            'active_strategies': len(self.active_strategies),
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'data_feed_connected': self.data_feed.is_connected,
            'broker_connected': self.broker.is_connected,
            'risk_metrics': self.risk_manager.get_risk_report()
        }
    
    def add_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]) -> bool:
        """Add a new strategy during runtime."""
        try:
            if not STRATEGIES_AVAILABLE or not strategy_registry:
                logger.error("Strategy registry not available")
                return False
                
            strategy = strategy_registry.create_strategy(
                strategy_name, strategy_config.get('params', {})
            )
            
            if strategy:
                self.active_strategies[strategy_name] = {
                    'instance': strategy,
                    'config': strategy_config,
                    'last_signal_time': None,
                    'positions': [],
                    'daily_pnl': 0.0
                }
                
                # Subscribe to additional instruments if needed
                instruments = strategy_config.get('instruments', [])
                if instruments:
                    self.data_feed.subscribe(instruments)
                
                logger.info(f"Added strategy: {strategy_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add strategy {strategy_name}: {e}")
            return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy during runtime."""
        try:
            if strategy_name in self.active_strategies:
                # Close any positions for this strategy
                strategy_positions = [
                    pos for pos in self.current_positions.values()
                    if getattr(pos, 'strategy_name', '') == strategy_name
                ]
                
                for position in strategy_positions:
                    self._close_position(position, "Strategy removed")
                
                # Remove strategy
                del self.active_strategies[strategy_name]
                logger.info(f"Removed strategy: {strategy_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove strategy {strategy_name}: {e}")
            return False