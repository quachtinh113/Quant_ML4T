"""Stable-supported and explicitly opt-in experimental data-feed components."""

from ml4t.live.feeds.aggregator import BarAggregator, BarBuffer

try:
    from ml4t.live.feeds.alpaca_feed import AlpacaDataFeed
except ImportError:
    AlpacaDataFeed = None

from ml4t.live.feeds.crypto_feed import CryptoFeed

try:
    from ml4t.live.feeds.databento_feed import DataBentoFeed
except ImportError:
    DataBentoFeed = None

from ml4t.live.feeds.events import FeedContinuityError, FeedContractError
from ml4t.live.feeds.experimental import ExperimentalFeedError, ExperimentalFeedWarning

try:
    from ml4t.live.feeds.ib_feed import IBDataFeed
except ImportError:
    IBDataFeed = None

from ml4t.live.feeds.okx_feed import OKXFundingFeed
from ml4t.live.feeds.queue import FeedOverflowError, FeedQueueSnapshot

__all__ = [
    "AlpacaDataFeed",
    "BarAggregator",
    "BarBuffer",
    "FeedContractError",
    "FeedContinuityError",
    "FeedOverflowError",
    "FeedQueueSnapshot",
    "IBDataFeed",
    "OKXFundingFeed",
    # Experimental opt-in surface
    "CryptoFeed",
    "DataBentoFeed",
    "ExperimentalFeedError",
    "ExperimentalFeedWarning",
]
