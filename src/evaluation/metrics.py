import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    balanced_accuracy_score
)
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from scipy.stats import hmean

class MetricsCalculator:
    """Class for calculating various performance metrics."""
    
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        # Ensure arrays are 1D
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'specificity': MetricsCalculator.specificity_score(y_true, y_pred)
        }
        
        if y_prob is not None and y_prob.shape[1] == 2:  # Only for binary classification
            try:
                metrics.update({
                    'auc_roc': roc_auc_score(y_true, y_prob[:, 1]),
                    'auc_pr': average_precision_score(y_true, y_prob[:, 1])
                })
            except ValueError as e:
                print(f"Warning: Could not calculate AUC metrics - {str(e)}")
                metrics.update({
                    'auc_roc': float('nan'),
                    'auc_pr': float('nan')
                })
        
        return metrics
    
    @staticmethod
    def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # If only one class, specificity is undefined
        if len(classes) == 1:
            return float('nan')
            
        # Ensure binary classification
        if len(classes) > 2:
            # Convert to binary by treating the most frequent class as negative
            # and all others as positive
            most_frequent = np.bincount(y_true).argmax()
            y_true_bin = (y_true != most_frequent).astype(int)
            y_pred_bin = (y_pred != most_frequent).astype(int)
        else:
            y_true_bin = y_true
            y_pred_bin = y_pred
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate specificity
        if tn + fp == 0:  # No negative cases
            return float('nan')
        return tn / (tn + fp)
    
    @staticmethod
    def calculate_class_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              classes: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        class_metrics = {}
        
        for i, class_name in enumerate(classes):
            class_metrics[class_name] = {
                'precision': precision_score(y_true, y_pred, labels=[i], average=None)[0],
                'recall': recall_score(y_true, y_pred, labels=[i], average=None)[0],
                'f1': f1_score(y_true, y_pred, labels=[i], average=None)[0]
            }
        
        return class_metrics
    
    @staticmethod
    def calculate_confidence_metrics(y_prob: np.ndarray,
                                  y_true: np.ndarray) -> Dict[str, float]:
        """Calculate metrics related to prediction confidence."""
        correct = y_prob.argmax(axis=1) == y_true
        confidences = np.max(y_prob, axis=1)
        
        return {
            'mean_confidence': np.mean(confidences),
            'mean_confidence_correct': np.mean(confidences[correct]),
            'mean_confidence_incorrect': np.mean(confidences[~correct]),
            'confidence_auroc': roc_auc_score(correct, confidences)
        }
    
    @staticmethod
    def calibration_metrics(y_prob: np.ndarray,
                          y_true: np.ndarray,
                          n_bins: int = 10) -> Dict[str, float]:
        """Calculate calibration-related metrics."""
        confidences = np.max(y_prob, axis=1)
        predictions = y_prob.argmax(axis=1)
        correct = predictions == y_true
        
        # Expected Calibration Error
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(correct[in_bin])
                confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
        
        # Maximum Calibration Error
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            if np.any(in_bin):
                accuracy_in_bin = np.mean(correct[in_bin])
                confidence_in_bin = np.mean(confidences[in_bin])
                mce = max(mce, np.abs(accuracy_in_bin - confidence_in_bin))
        
        return {
            'ece': ece,
            'mce': mce
        }
    
    @staticmethod
    def threshold_metrics(y_prob: np.ndarray,
                         y_true: np.ndarray,
                         thresholds: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Calculate metrics at different decision thresholds."""
        if thresholds is None:
            thresholds = np.linspace(0, 1, 100)
        
        metrics_at_thresholds = {
            'thresholds': thresholds,
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': []
        }
        
        for threshold in thresholds:
            y_pred = (y_prob[:, 1] > threshold).astype(int)
            
            metrics_at_thresholds['precision'].append(
                precision_score(y_true, y_pred)
            )
            metrics_at_thresholds['recall'].append(
                recall_score(y_true, y_pred)
            )
            metrics_at_thresholds['f1'].append(
                f1_score(y_true, y_pred)
            )
            metrics_at_thresholds['specificity'].append(
                MetricsCalculator.specificity_score(y_true, y_pred)
            )
        
        # Convert lists to numpy arrays
        for key in metrics_at_thresholds:
            if key != 'thresholds':
                metrics_at_thresholds[key] = np.array(metrics_at_thresholds[key])
        
        return metrics_at_thresholds
    
    @staticmethod
    def optimal_threshold(y_prob: np.ndarray,
                         y_true: np.ndarray,
                         metric: str = 'f1') -> Dict[str, float]:
        """Find optimal decision threshold based on specified metric."""
        thresholds = np.linspace(0, 1, 100)
        metrics = MetricsCalculator.threshold_metrics(y_prob, y_true, thresholds)
        
        if metric == 'f1':
            best_idx = np.argmax(metrics['f1'])
        elif metric == 'balanced':
            # Find threshold that balances sensitivity and specificity
            sensitivity = metrics['recall']
            specificity = metrics['specificity']
            balance = np.abs(sensitivity - specificity)
            best_idx = np.argmin(balance)
        elif metric == 'youden':
            # Youden's J statistic (sensitivity + specificity - 1)
            j_statistic = metrics['recall'] + metrics['specificity'] - 1
            best_idx = np.argmax(j_statistic)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return {
            'threshold': thresholds[best_idx],
            'precision': metrics['precision'][best_idx],
            'recall': metrics['recall'][best_idx],
            'f1': metrics['f1'][best_idx],
            'specificity': metrics['specificity'][best_idx]
        }
    
    @staticmethod
    def cross_entropy_metrics(logits: torch.Tensor,
                            targets: torch.Tensor,
                            label_smoothing: float = 0.0) -> Dict[str, float]:
        """Calculate cross-entropy based metrics."""
        with torch.no_grad():
            # Standard cross-entropy
            ce_loss = F.cross_entropy(logits, targets)
            
            # Label smoothed cross-entropy
            if label_smoothing > 0:
                smooth_loss = F.cross_entropy(
                    logits,
                    torch.full_like(targets.float(), label_smoothing / (logits.size(-1) - 1))
                )
                loss = (1 - label_smoothing) * ce_loss + label_smoothing * smooth_loss
            else:
                loss = ce_loss
            
            # Confidence penalty
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            
        return {
            'cross_entropy': ce_loss.item(),
            'smoothed_loss': loss.item(),
            'entropy': entropy.item()
        }
