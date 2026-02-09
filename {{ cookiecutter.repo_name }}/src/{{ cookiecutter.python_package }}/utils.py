"""Utility functions for retrieval assessment."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_metrics(metrics: dict[str, Any], method_name: str) -> None:
    """Log evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary of metric names to values
        method_name: Name of the retrieval method
    """
    logger.info(f"\n{'=' * 40}")
    logger.info(f"Results for: {method_name}")
    logger.info(f"{'=' * 40}")

    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


def calculate_improvement(baseline: float, current: float) -> float:
    """Calculate percentage improvement over baseline.

    Args:
        baseline: Baseline metric value
        current: Current metric value

    Returns:
        Percentage improvement
    """
    if baseline == 0:
        return 0.0

    return ((current - baseline) / baseline) * 100
