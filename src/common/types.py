"""
Common type definitions for the trading bot.
"""

from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

# Basic types
Symbol = NewType('Symbol', str)
Price = NewType('Price', Decimal)
Volume = NewType('Volume', Decimal)
Quantity = NewType('Quantity', Decimal)
Percentage = NewType('Percentage', float)
Timestamp = NewType('Timestamp', int)

# Trading types
Side = Literal['buy', 'sell', 'long', 'short']
OrderType = Literal['market', 'limit', 'stop', 'stop_limit']
OrderStatus = Literal['pending', 'open', 'filled', 'cancelled', 'rejected']
PositionSide = Literal['long', 'short', 'neutral']
TimeInForce = Literal['GTC', 'IOC', 'FOK', 'GTD']

# Model types
ModelVersion = NewType('ModelVersion', str)
FeatureVector = NewType('FeatureVector', List[float])
Prediction = NewType('Prediction', float)
Confidence = NewType('Confidence', float)

# Market data types
KlineData = Dict[str, Union[float, int, str]]
OrderBookLevel = Tuple[Price, Volume]
OrderBookData = Dict[str, List[OrderBookLevel]]
TradeData = Dict[str, Union[float, int, str]]
LiquidationData = Dict[str, Union[float, int, str]]

# Configuration types
ConfigDict = Dict[str, Any]
EnvironmentType = Literal['development', 'staging', 'production']


class MarketDataType(str, Enum):
    """Market data types."""
    KLINE = "kline"
    ORDERBOOK = "orderbook" 
    TRADE = "trade"
    LIQUIDATION = "liquidation"
    TICKER = "ticker"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"


class TradingAction(str, Enum):
    """Trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class RiskLevel(str, Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemStatus(str, Enum):
    """System status states."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class PricePoint:
    """Price point with timestamp."""
    price: Price
    timestamp: datetime
    volume: Optional[Volume] = None


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: Symbol
    action: TradingAction
    confidence: Confidence
    expected_pnl: float
    risk_level: RiskLevel
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'confidence': float(self.confidence),
            'expected_pnl': self.expected_pnl,
            'risk_level': self.risk_level.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


@dataclass
class OrderInfo:
    """Order information."""
    order_id: str
    symbol: Symbol
    side: Side
    order_type: OrderType
    quantity: Quantity
    price: Optional[Price]
    status: OrderStatus
    filled_quantity: Quantity
    remaining_quantity: Quantity
    timestamp: datetime
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == 'filled'
    
    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in ['pending', 'open']
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity)


@dataclass
class PositionInfo:
    """Position information."""
    symbol: Symbol
    side: PositionSide
    size: Quantity
    entry_price: Price
    current_price: Price
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: datetime
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL percentage."""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == 'long':
            return float((self.current_price - self.entry_price) / self.entry_price)
        elif self.side == 'short':
            return float((self.entry_price - self.current_price) / self.entry_price)
        else:
            return 0.0
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        return float(self.unrealized_pnl) > 0


@dataclass
class MarketDataPoint:
    """Market data point."""
    symbol: Symbol
    data_type: MarketDataType
    timestamp: datetime
    data: Dict[str, Any]
    
    def get_price(self) -> Optional[Price]:
        """Extract price from data."""
        if 'price' in self.data:
            return Price(Decimal(str(self.data['price'])))
        elif 'close' in self.data:
            return Price(Decimal(str(self.data['close'])))
        return None
    
    def get_volume(self) -> Optional[Volume]:
        """Extract volume from data."""
        if 'volume' in self.data:
            return Volume(Decimal(str(self.data['volume'])))
        return None


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: Decimal
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: float
    
    @property
    def loss_rate(self) -> float:
        """Calculate loss rate."""
        return 1.0 - self.win_rate if self.win_rate <= 1.0 else 0.0
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio."""
        if self.avg_loss == 0:
            return float('inf')
        return float(self.avg_win / abs(self.avg_loss))


@dataclass
class FeatureSet:
    """Feature set for ML model."""
    symbol: Symbol
    timestamp: datetime
    features: FeatureVector
    feature_names: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'features': list(self.features),
            'feature_names': self.feature_names,
            'metadata': self.metadata or {}
        }
    
    def get_feature_by_name(self, name: str) -> Optional[float]:
        """Get feature value by name."""
        try:
            index = self.feature_names.index(name)
            return self.features[index]
        except ValueError:
            return None


@dataclass
class ModelPrediction:
    """Model prediction result."""
    symbol: Symbol
    timestamp: datetime
    prediction: Prediction
    confidence: Confidence
    model_version: ModelVersion
    features_used: int
    inference_time_ms: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'prediction': float(self.prediction),
            'confidence': float(self.confidence),
            'model_version': self.model_version,
            'features_used': self.features_used,
            'inference_time_ms': self.inference_time_ms,
            'metadata': self.metadata or {}
        }


@dataclass
class RiskMetrics:
    """Risk management metrics."""
    max_position_size: Decimal
    current_exposure: Decimal
    available_balance: Decimal
    daily_pnl: Decimal
    max_daily_loss: Decimal
    current_drawdown: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    
    @property
    def exposure_percentage(self) -> float:
        """Calculate exposure percentage."""
        if self.available_balance == 0:
            return 0.0
        return float(self.current_exposure / self.available_balance)
    
    @property
    def is_risk_limit_breached(self) -> bool:
        """Check if any risk limits are breached."""
        return (
            self.daily_pnl < -self.max_daily_loss or
            self.current_drawdown > self.max_drawdown or
            self.current_exposure > self.max_position_size
        )


# Type aliases for complex types
SymbolDataMap = Dict[Symbol, Any]
TimestampedData = Dict[Timestamp, Any]
FeatureDict = Dict[str, float]
ConfigurationMap = Dict[str, ConfigDict]
MetricsMap = Dict[str, Union[int, float, str]]

# Callback types
DataCallback = Callable[[MarketDataPoint], None]
SignalCallback = Callable[[TradingSignal], None]
ErrorCallback = Callable[[Exception], None]
StatusCallback = Callable[[SystemStatus], None]