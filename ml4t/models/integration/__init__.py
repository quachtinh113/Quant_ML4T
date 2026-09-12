"""Integration helpers for cross-library data contracts."""

from ml4t.models.integration.backtest import (
    BacktestDataFeedInputs,
    backtest_datafeed_inputs,
    backtest_inputs_from_asset_forecast,
    backtest_inputs_from_asset_signal,
    backtest_inputs_from_weights,
    resolve_feed_spec_mapping,
)
from ml4t.models.integration.data import (
    ResolvedDatasetSchema,
    cross_section_batch_from_long_frame,
    persistent_panel_batch_from_long_frame,
    resolve_dataset_schema,
)
from ml4t.models.integration.surfaces import (
    ContextFrame,
    PredictionsFrame,
    ResultsFrame,
    SignalsFrame,
    WeightsFrame,
    context_frame_from_weights,
    predictions_frame_from_asset_forecast,
    predictions_frame_from_asset_signal,
    signals_frame_from_asset_weights,
    signals_frame_from_portfolio_weights,
    weights_frame_from_asset_weights,
    weights_frame_from_portfolio_weights,
    write_backtest_frames,
)

__all__ = [
    "BacktestDataFeedInputs",
    "ContextFrame",
    "PredictionsFrame",
    "ResolvedDatasetSchema",
    "ResultsFrame",
    "SignalsFrame",
    "WeightsFrame",
    "backtest_datafeed_inputs",
    "backtest_inputs_from_asset_forecast",
    "backtest_inputs_from_asset_signal",
    "backtest_inputs_from_weights",
    "context_frame_from_weights",
    "cross_section_batch_from_long_frame",
    "predictions_frame_from_asset_forecast",
    "predictions_frame_from_asset_signal",
    "persistent_panel_batch_from_long_frame",
    "resolve_feed_spec_mapping",
    "resolve_dataset_schema",
    "signals_frame_from_asset_weights",
    "signals_frame_from_portfolio_weights",
    "weights_frame_from_asset_weights",
    "weights_frame_from_portfolio_weights",
    "write_backtest_frames",
]
