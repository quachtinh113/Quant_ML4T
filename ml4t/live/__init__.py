"""ML4T Live Trading Platform.

Lifecycle-compatible strategies can share decision logic across backtest and live runtimes.
"""

try:
    from ml4t.live._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

try:
    from .brokers.alpaca import AlpacaBroker
except ImportError:
    AlpacaBroker = None

try:
    from .brokers.ib import IBBroker
except ImportError:
    IBBroker = None

from .engine import (
    LiveEngine,
    RuntimeCleanupError,
    RuntimeErrorContext,
    RuntimeFailureError,
    RuntimeState,
    RuntimeTransition,
    runtime_error_context,
)
from .feeds.aggregator import BarAggregator, BarBuffer

try:
    from .feeds.alpaca_feed import AlpacaDataFeed
except ImportError:
    AlpacaDataFeed = None

from .feeds.crypto_feed import CryptoFeed

try:
    from .feeds.databento_feed import DataBentoFeed
except ImportError:
    DataBentoFeed = None

from .feeds.events import FeedContinuityError, FeedContractError
from .feeds.experimental import ExperimentalFeedError, ExperimentalFeedWarning

try:
    from .feeds.ib_feed import IBDataFeed
except ImportError:
    IBDataFeed = None

from .feeds.okx_feed import OKXFundingFeed
from .feeds.queue import FeedOverflowError, FeedQueueSnapshot
from .lifecycle import (
    LifecycleInvocation,
    LiveLifecycleDispatcher,
    StrategyCallbackTimeoutError,
    callback_trace,
)
from .orders import (
    BrokerOrderContractError,
    CanonicalOrderRequest,
    OrderValidationError,
    UnsupportedOrderCapabilityError,
)
from .persistence import (
    AcceptedOrderPersistenceError,
    AuditJournalError,
    ConcurrentStateWriterError,
    CorruptStateError,
    PersistenceSafetyError,
    UnsafePersistencePathError,
)
from .protocols import AsyncBrokerProtocol, BrokerProtocol, DataFeedProtocol
from .runtime import (
    LiveIntentError,
    LiveStrategyRuntime,
    LiveStrategyRuntimeError,
    ReducingRiskExecutionError,
    UnsupportedLiveCapabilityError,
    default_live_execution_policy,
)
from .safety import (
    BrokerSnapshotError,
    ExecutionMode,
    ExecutionModeError,
    LiveRiskConfig,
    OrderReplacementGapError,
    ReconciliationMismatchError,
    RiskLimitError,
    RiskState,
    SafeBroker,
    VirtualPortfolio,
)
from .wrappers import ThreadSafeBrokerWrapper

__all__ = [
    # Brokers
    "AlpacaBroker",
    "IBBroker",
    # Data Feeds
    "AlpacaDataFeed",
    "BarAggregator",
    "BarBuffer",
    "FeedContractError",
    "FeedContinuityError",
    "FeedOverflowError",
    "FeedQueueSnapshot",
    "IBDataFeed",
    "OKXFundingFeed",
    # Experimental Data Feeds
    "CryptoFeed",
    "DataBentoFeed",
    "ExperimentalFeedError",
    "ExperimentalFeedWarning",
    # Engine
    "LifecycleInvocation",
    "LiveEngine",
    "RuntimeCleanupError",
    "RuntimeErrorContext",
    "RuntimeFailureError",
    "RuntimeState",
    "RuntimeTransition",
    "runtime_error_context",
    "BrokerOrderContractError",
    "CanonicalOrderRequest",
    "OrderValidationError",
    "UnsupportedOrderCapabilityError",
    "LiveLifecycleDispatcher",
    "StrategyCallbackTimeoutError",
    "callback_trace",
    "LiveIntentError",
    "LiveStrategyRuntime",
    "LiveStrategyRuntimeError",
    "ReducingRiskExecutionError",
    "UnsupportedLiveCapabilityError",
    "default_live_execution_policy",
    # Protocols
    "AsyncBrokerProtocol",
    "BrokerProtocol",
    "DataFeedProtocol",
    # Safety
    "AcceptedOrderPersistenceError",
    "AuditJournalError",
    "BrokerSnapshotError",
    "ExecutionMode",
    "ExecutionModeError",
    "ConcurrentStateWriterError",
    "CorruptStateError",
    "LiveRiskConfig",
    "OrderReplacementGapError",
    "PersistenceSafetyError",
    "ReconciliationMismatchError",
    "RiskLimitError",
    "RiskState",
    "SafeBroker",
    "UnsafePersistencePathError",
    "VirtualPortfolio",
    # Wrappers
    "ThreadSafeBrokerWrapper",
]
