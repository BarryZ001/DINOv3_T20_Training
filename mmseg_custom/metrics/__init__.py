"""MMSeg Custom Metrics Module

This module provides custom metric implementations for semantic segmentation tasks.
"""

from .iou_metric import IoUMetric

__all__ = ['IoUMetric']