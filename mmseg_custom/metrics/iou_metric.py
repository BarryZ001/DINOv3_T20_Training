"""IoU Metric Implementation for Semantic Segmentation

This module provides IoU (Intersection over Union) metric calculation
compatible with MMEngine's evaluation framework.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Sequence, Union

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU metric for semantic segmentation evaluation.
    
    This metric calculates Intersection over Union (IoU) for semantic segmentation
    tasks, supporting multiple IoU-based metrics like mIoU, mDice, and mFscore.
    
    Args:
        iou_metrics (list[str]): List of IoU metrics to compute.
            Options: ['mIoU', 'mDice', 'mFscore']. Default: ['mIoU'].
        nan_to_num (int, optional): The value to replace NaN values.
            If None, NaN values are kept. Default: None.
        beta (int): The beta value for F-score calculation. Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Default: 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
    """

    def __init__(self,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        
        self.iou_metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        
        # Validate metrics
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        for metric in iou_metrics:
            if metric not in allowed_metrics:
                raise ValueError(f"Unsupported metric: {metric}. "
                               f"Allowed metrics: {allowed_metrics}")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze()
            
            # Convert to numpy if tensor
            if isinstance(pred_label, torch.Tensor):
                pred_label = pred_label.cpu().numpy()
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()
            
            self.results.append({
                'pred_label': pred_label,
                'label': label
            })

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # Collect all predictions and labels
        pred_labels = []
        gt_labels = []
        
        for result in results:
            pred_labels.append(result['pred_label'])
            gt_labels.append(result['label'])
        
        # Calculate metrics
        metrics = {}
        
        # Get number of classes from the data
        all_labels = np.concatenate([pred.flatten() for pred in pred_labels] + 
                                  [gt.flatten() for gt in gt_labels])
        num_classes = int(np.max(all_labels)) + 1
        
        # Calculate confusion matrix
        confusion_matrix = self._calculate_confusion_matrix(
            pred_labels, gt_labels, num_classes)
        
        # Calculate IoU for each class
        iou_per_class = self._calculate_iou_per_class(confusion_matrix)
        
        # Calculate requested metrics
        for metric in self.iou_metrics:
            if metric == 'mIoU':
                miou = np.nanmean(iou_per_class)
                if self.nan_to_num is not None:
                    miou = np.nan_to_num(miou, nan=self.nan_to_num)
                metrics['mIoU'] = miou
                
            elif metric == 'mDice':
                dice_per_class = self._calculate_dice_per_class(confusion_matrix)
                mdice = np.nanmean(dice_per_class)
                if self.nan_to_num is not None:
                    mdice = np.nan_to_num(mdice, nan=self.nan_to_num)
                metrics['mDice'] = mdice
                
            elif metric == 'mFscore':
                fscore_per_class = self._calculate_fscore_per_class(
                    confusion_matrix, self.beta)
                mfscore = np.nanmean(fscore_per_class)
                if self.nan_to_num is not None:
                    mfscore = np.nan_to_num(mfscore, nan=self.nan_to_num)
                metrics['mFscore'] = mfscore
        
        # Add per-class IoU for debugging
        for i, iou in enumerate(iou_per_class):
            if not np.isnan(iou):
                metrics[f'IoU.class_{i}'] = iou
        
        return metrics

    def _calculate_confusion_matrix(self, pred_labels: List[np.ndarray], 
                                  gt_labels: List[np.ndarray], 
                                  num_classes: int) -> np.ndarray:
        """Calculate confusion matrix."""
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        for pred, gt in zip(pred_labels, gt_labels):
            # Flatten arrays
            pred_flat = pred.flatten()
            gt_flat = gt.flatten()
            
            # Filter out ignore index (255)
            valid_mask = gt_flat != 255
            pred_flat = pred_flat[valid_mask]
            gt_flat = gt_flat[valid_mask]
            
            # Update confusion matrix
            for p, g in zip(pred_flat, gt_flat):
                if 0 <= p < num_classes and 0 <= g < num_classes:
                    confusion_matrix[g, p] += 1
        
        return confusion_matrix

    def _calculate_iou_per_class(self, confusion_matrix: np.ndarray) -> np.ndarray:
        """Calculate IoU for each class."""
        intersection = np.diag(confusion_matrix)
        union = (confusion_matrix.sum(axis=1) + 
                confusion_matrix.sum(axis=0) - intersection)
        
        # Avoid division by zero
        iou = np.divide(intersection, union, 
                       out=np.zeros_like(intersection, dtype=float), 
                       where=union != 0)
        
        return iou

    def _calculate_dice_per_class(self, confusion_matrix: np.ndarray) -> np.ndarray:
        """Calculate Dice coefficient for each class."""
        intersection = np.diag(confusion_matrix)
        pred_sum = confusion_matrix.sum(axis=0)
        gt_sum = confusion_matrix.sum(axis=1)
        
        # Dice = 2 * intersection / (pred + gt)
        dice = np.divide(2 * intersection, pred_sum + gt_sum,
                        out=np.zeros_like(intersection, dtype=float),
                        where=(pred_sum + gt_sum) != 0)
        
        return dice

    def _calculate_fscore_per_class(self, confusion_matrix: np.ndarray, 
                                   beta: int) -> np.ndarray:
        """Calculate F-score for each class."""
        intersection = np.diag(confusion_matrix)
        pred_sum = confusion_matrix.sum(axis=0)
        gt_sum = confusion_matrix.sum(axis=1)
        
        # Precision and Recall
        precision = np.divide(intersection, pred_sum,
                            out=np.zeros_like(intersection, dtype=float),
                            where=pred_sum != 0)
        recall = np.divide(intersection, gt_sum,
                         out=np.zeros_like(intersection, dtype=float),
                         where=gt_sum != 0)
        
        # F-score = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
        beta_sq = beta ** 2
        fscore = np.divide((1 + beta_sq) * precision * recall,
                          beta_sq * precision + recall,
                          out=np.zeros_like(precision, dtype=float),
                          where=(beta_sq * precision + recall) != 0)
        
        return fscore