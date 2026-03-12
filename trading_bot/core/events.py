"""
Event system for the trading bot.
Implements the event bus pattern for loose coupling between components.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
import uuid
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the trading system."""
    MARKET_DATA = "market_data"
    TRADING_SIGNAL = "trading_signal"
    TRADE_DECISION = "trade_decision"
    RISK_APPROVED_TRADE = "risk_approved_trade"
    RISK_REJECTED = "risk_rejected"
    ORDER_STATUS = "order_status"
    SYSTEM_ERROR = "system_error"
    SYSTEM_SHUTDOWN = "system_shutdown"


@dataclass
class BaseEvent(ABC):
    """Base class for all events in the system."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = field(init=False)
    timestamp: datetime = field(default_factory=datetime.now)
    source_component: str = ""
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            raise NotImplementedError("Event must define event_type")


@dataclass
class MarketDataEvent(BaseEvent):
    """Event containing market data updates."""
    event_type: EventType = field(default=EventType.MARKET_DATA, init=False)
    symbol: str = ""
    exchange: str = ""
    price: Decimal = Decimal('0')
    volume: int = 0
    bid_price: Decimal = Decimal('0')
    ask_price: Decimal = Decimal('0')
    bid_size: int = 0
    ask_size: int = 0


@dataclass
class TradingSignalEvent(BaseEvent):
    """Event containing trading signals from technical analysis."""
    event_type: EventType = field(default=EventType.TRADING_SIGNAL, init=False)
    symbol: str = ""
    signal_type: str = ""  # BUY/SELL/HOLD
    strength: float = 0.0
    indicators_used: List[str] = field(default_factory=list)
    price_target: Optional[Decimal] = None


@dataclass
class SignalEvent(BaseEvent):
    """Event containing trading signals (alias for compatibility)."""
    event_type: EventType = field(default=EventType.TRADING_SIGNAL, init=False)
    signal: Any = None  # Signal object
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderEvent(BaseEvent):
    """Event containing order information."""
    event_type: EventType = field(default=EventType.ORDER_STATUS, init=False)
    order_id: str = ""
    symbol: str = ""
    action: str = ""  # BUY/SELL
    quantity: int = 0
    price: float = 0.0
    status: str = ""  # PLACED/FILLED/CANCELLED etc.
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeDecisionEvent(BaseEvent):
    """Event containing trade decisions from strategy engine."""
    event_type: EventType = field(default=EventType.TRADE_DECISION, init=False)
    symbol: str = ""
    action: str = ""  # BUY/SELL
    quantity: int = 0
    order_type: str = ""  # MARKET/LIMIT
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    strategy_id: str = ""


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle_event(self, event: BaseEvent) -> None:
        """Handle an incoming event."""
        pass


class EventBus:
    """
    Central event bus for component communication.
    Implements publish-subscribe pattern with async event handling.
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, Set[EventHandler]] = {}
        self._sync_handlers: Dict[str, list] = {}  # For string-based subscriptions
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        
    def subscribe(self, event_type, handler) -> None:
        """Subscribe a handler to specific event type (supports both EventType and string)."""
        if isinstance(event_type, str):
            # Synchronous string-based subscription
            if event_type not in self._sync_handlers:
                self._sync_handlers[event_type] = []
            self._sync_handlers[event_type].append(handler)
            logger.info(f"Handler subscribed to string event: {event_type}")
        else:
            # Original EventType subscription
            if event_type not in self._handlers:
                self._handlers[event_type] = set()
            self._handlers[event_type].add(handler)
            logger.info(f"Handler {handler.__class__.__name__} subscribed to {event_type.value}")
    
    def emit(self, event: BaseEvent) -> None:
        """Emit an event synchronously to all registered handlers."""
        try:
            # Dispatch to EventType-based handlers
            event_type = getattr(event, 'event_type', None)
            if event_type and event_type in self._handlers:
                for handler in self._handlers[event_type]:
                    try:
                        if hasattr(handler, 'handle_event'):
                            import asyncio as _asyncio
                            try:
                                loop = _asyncio.get_event_loop()
                                if loop.is_running():
                                    loop.create_task(handler.handle_event(event))
                                else:
                                    loop.run_until_complete(handler.handle_event(event))
                            except RuntimeError:
                                pass
                        elif callable(handler):
                            handler(event)
                    except Exception as e:
                        logger.error(f"Error in handler {handler}: {e}")

            # Dispatch to string-based sync handlers
            event_type_str = getattr(event, 'event_type', '')
            if isinstance(event_type_str, str) and event_type_str in self._sync_handlers:
                for handler in self._sync_handlers[event_type_str]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error in sync handler: {e}")
        except Exception as e:
            logger.error(f"Error in EventBus.emit(): {e}")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from specific event type."""
        if event_type in self._handlers:
            self._handlers[event_type].discard(handler)
            logger.info(f"Handler {handler.__class__.__name__} unsubscribed from {event_type.value}")
    
    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to the bus."""
        await self._event_queue.put(event)
        logger.debug(f"Published event {event.event_type.value} from {event.source_component}")
    
    async def start(self) -> None:
        """Start the event processing loop."""
        if self._running:
            return
            
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                # Wait for event with timeout to allow graceful shutdown
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _dispatch_event(self, event: BaseEvent) -> None:
        """Dispatch event to all registered handlers."""
        handlers = self._handlers.get(event.event_type, set())
        if not handlers:
            logger.warning(f"No handlers registered for event type {event.event_type.value}")
            return
        
        # Process handlers concurrently
        tasks = [handler.handle_event(event) for handler in handlers]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error dispatching event {event.event_type.value}: {e}")


# Global event bus instance
event_bus = EventBus()