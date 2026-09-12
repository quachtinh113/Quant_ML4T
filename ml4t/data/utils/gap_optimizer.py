"""Optimized gap filling strategies for data updates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from ml4t.data.utils.gaps import DataGap

logger = structlog.get_logger()


class GapOptimizer:
    """Optimize gap filling by consolidating fetch requests."""

    def __init__(self, max_batch_days: int = 30) -> None:
        """
        Initialize gap optimizer.

        Args:
            max_batch_days: Maximum days to fetch in a single batch
        """
        self.max_batch_days = max_batch_days

    def consolidate_gaps(
        self, gaps: list[DataGap], merge_threshold_days: int = 7
    ) -> list[tuple[datetime, datetime]]:
        """
        Consolidate multiple gaps into fewer fetch requests.

        Instead of fetching data for each gap individually, this method
        consolidates adjacent or nearby gaps into single fetch requests,
        reducing API calls and improving efficiency.

        Args:
            gaps: List of detected gaps
            merge_threshold_days: Merge gaps within this many days

        Returns:
            List of (start, end) date ranges to fetch
        """
        if not gaps:
            return []

        # Sort gaps by start time
        sorted_gaps = sorted(gaps, key=lambda g: g.start)

        # Consolidate adjacent gaps
        consolidated = []
        current_start = sorted_gaps[0].start
        current_end = sorted_gaps[0].end

        for gap in sorted_gaps[1:]:
            # Check if this gap is close enough to merge
            gap_between = (gap.start - current_end).days

            if gap_between <= merge_threshold_days:
                # Merge this gap with the current range
                current_end = max(current_end, gap.end)
                logger.debug(
                    "Merging gaps",
                    gap_between_days=gap_between,
                    new_range_days=(current_end - current_start).days,
                )
            else:
                # Save current range and start a new one
                consolidated.append((current_start, current_end))
                current_start = gap.start
                current_end = gap.end

        # Add the last range
        consolidated.append((current_start, current_end))

        logger.info(
            "Consolidated gaps",
            original_gaps=len(gaps),
            consolidated_ranges=len(consolidated),
            reduction_pct=round((1 - len(consolidated) / len(gaps)) * 100, 1),
        )

        # Split large ranges if needed
        final_ranges = []
        for start, end in consolidated:
            range_days = (end - start).days

            if range_days <= self.max_batch_days:
                final_ranges.append((start, end))
            else:
                # Split into smaller batches
                current = start
                while current < end:
                    batch_end = min(current + timedelta(days=self.max_batch_days), end)
                    final_ranges.append((current, batch_end))
                    current = batch_end + timedelta(days=1)

        return final_ranges

    def estimate_api_calls(
        self, gaps: list[DataGap], merge_threshold_days: int = 7
    ) -> dict[str, Any]:
        """
        Estimate the number of API calls needed for gap filling.

        Args:
            gaps: List of detected gaps
            merge_threshold_days: Merge gaps within this many days

        Returns:
            Dictionary with estimation details
        """
        if not gaps:
            return {"individual_calls": 0, "consolidated_calls": 0, "savings": 0, "savings_pct": 0}

        individual_calls = len(gaps)
        consolidated_ranges = self.consolidate_gaps(gaps, merge_threshold_days)
        consolidated_calls = len(consolidated_ranges)

        savings = individual_calls - consolidated_calls
        savings_pct = round((savings / individual_calls) * 100, 1) if individual_calls > 0 else 0

        return {
            "individual_calls": individual_calls,
            "consolidated_calls": consolidated_calls,
            "savings": savings,
            "savings_pct": savings_pct,
            "total_days": sum((end - start).days for start, end in consolidated_ranges),
        }

    def should_consolidate(self, gaps: list[DataGap], threshold_gaps: int = 3) -> bool:
        """
        Determine if gap consolidation would be beneficial.

        Args:
            gaps: List of detected gaps
            threshold_gaps: Minimum gaps to trigger consolidation

        Returns:
            True if consolidation is recommended
        """
        if len(gaps) < threshold_gaps:
            return False

        estimation = self.estimate_api_calls(gaps)

        # Consolidate if we can save at least 30% of API calls
        return estimation["savings_pct"] >= 30
