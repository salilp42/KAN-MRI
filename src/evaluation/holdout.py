import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from .metrics import MetricsCalculator
from ..analysis.statistical_tests import StatisticalAnalyzer
import pandas as pd
from tqdm.auto import tqdm

class HoldoutEvaluator:
    """Class for performing holdout testing and evaluation."""
    
    def __init__(self,
                 save_dir: str = 'results/holdout',
                 metrics_calculator: Optional[MetricsCalculator] = None,
                 stats_analyzer: Optional[StatisticalAnalyzer] = None):
        """Initialize holdout evaluator."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.stats_analyzer = stats_analyzer or StatisticalAnalyzer()
    
    def create_holdout_split(self,
                           dataset: torch.utils.data.Dataset,
                           test_size: float = 0.2,
                           stratify: Optional[np.ndarray] = None,
                           random_state: int = 42) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Create train and holdout splits."""
        indices = np.arange(len(dataset))
        
        # Ensure stratification
        if stratify is None and hasattr(dataset, 'targets'):
            stratify = dataset.targets
        
        # Create stratified split
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state
        )
        
        # Verify both classes are present in test set
        if stratify is not None:
            test_labels = stratify[test_idx]
            unique_labels = np.unique(test_labels)
            if len(unique_labels) < 2:
                print("Warning: Test set contains only one class. Adjusting split...")
                # Find indices for each class
                class_indices = {label: np.where(stratify == label)[0] for label in np.unique(stratify)}
                # Ensure test set has at least one sample from each class
                test_idx = []
                for label in class_indices:
                    label_indices = class_indices[label]
                    n_samples = max(1, int(len(label_indices) * test_size))
                    test_idx.extend(np.random.choice(label_indices, size=n_samples, replace=False))
                train_idx = np.array([i for i in indices if i not in test_idx])
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        return train_dataset, test_dataset
    
    def evaluate_model(self,
                      model: torch.nn.Module,
                      test_loader: torch.utils.data.DataLoader,
                      device: str = 'cuda') -> Dict[str, Any]:
        """Evaluate model on holdout set."""
        model.eval()
        model = model.to(device)
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Evaluating batches", leave=False):
                # Unpack the batch - DataLoader always returns a tuple of (inputs, targets)
                inputs, targets = batch_data
                
                # Move to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets)
        
        # Calculate metrics
        print("Computing metrics...")
        with tqdm(total=3, desc="Computing metrics", leave=False) as pbar:
            basic_metrics = self.metrics_calculator.calculate_basic_metrics(
                targets, predictions, probabilities
            )
            pbar.update(1)
            
            confidence_metrics = self.metrics_calculator.calculate_confidence_metrics(
                probabilities, targets
            )
            pbar.update(1)
            
            calibration_metrics = self.metrics_calculator.calibration_metrics(
                probabilities, targets
            )
            pbar.update(1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'targets': targets,
            'basic_metrics': basic_metrics,
            'confidence_metrics': confidence_metrics,
            'calibration_metrics': calibration_metrics
        }
    
    def bootstrap_evaluation(self,
                           predictions: np.ndarray,
                           targets: np.ndarray,
                           n_iterations: int = 1000,
                           confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """Perform bootstrap evaluation of metrics."""
        n_samples = len(targets)
        bootstrap_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for _ in range(n_iterations):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_preds = predictions[indices]
            boot_targets = targets[indices]
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_basic_metrics(
                boot_targets, boot_preds
            )
            
            for metric in bootstrap_metrics:
                bootstrap_metrics[metric].append(metrics[metric])
        
        # Calculate confidence intervals
        alpha = (1 - confidence_level) / 2
        results = {}
        
        for metric, values in bootstrap_metrics.items():
            values = np.array(values)
            results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 100 * alpha),
                'ci_upper': np.percentile(values, 100 * (1 - alpha))
            }
        
        return results
    
    def compare_with_baseline(self,
                            model_predictions: np.ndarray,
                            baseline_predictions: np.ndarray,
                            targets: np.ndarray,
                            alpha: float = 0.05) -> Dict[str, Any]:
        """Compare model performance with baseline."""
        # Statistical tests
        mcnemar_results = self.stats_analyzer.mcnemar_test(
            model_predictions,
            baseline_predictions,
            targets,
            alpha
        )
        
        # Performance difference
        model_metrics = self.metrics_calculator.calculate_basic_metrics(
            targets, model_predictions
        )
        baseline_metrics = self.metrics_calculator.calculate_basic_metrics(
            targets, baseline_predictions
        )
        
        metric_differences = {
            metric: model_metrics[metric] - baseline_metrics[metric]
            for metric in model_metrics
        }
        
        return {
            'statistical_tests': mcnemar_results,
            'model_metrics': model_metrics,
            'baseline_metrics': baseline_metrics,
            'differences': metric_differences
        }
    
    def generate_holdout_report(self,
                              evaluation_results: Dict[str, Any],
                              bootstrap_results: Optional[Dict[str, Dict[str, float]]] = None,
                              baseline_comparison: Optional[Dict[str, Any]] = None,
                              save_path: str = 'holdout_report.json'):
        """Generate comprehensive holdout evaluation report."""
        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        report = {
            'basic_metrics': convert_to_native(evaluation_results['basic_metrics']),
            'confidence_metrics': convert_to_native(evaluation_results['confidence_metrics']),
            'calibration_metrics': convert_to_native(evaluation_results['calibration_metrics'])
        }
        
        if bootstrap_results:
            report['bootstrap_analysis'] = convert_to_native(bootstrap_results)
        
        if baseline_comparison:
            report['baseline_comparison'] = convert_to_native(baseline_comparison)
        
        # Save report
        with open(self.save_dir / save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report
    
    def evaluate_and_report(self,
                          model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          baseline_model: Optional[torch.nn.Module] = None,
                          device: str = 'cuda') -> Dict[str, Any]:
        """Perform complete holdout evaluation and generate report."""
        # Evaluate model
        evaluation_results = self.evaluate_model(model, test_loader, device)
        
        # Bootstrap evaluation
        bootstrap_results = self.bootstrap_evaluation(
            evaluation_results['predictions'],
            evaluation_results['targets']
        )
        
        # Baseline comparison if provided
        baseline_comparison = None
        if baseline_model is not None:
            baseline_results = self.evaluate_model(baseline_model, test_loader, device)
            baseline_comparison = self.compare_with_baseline(
                evaluation_results['predictions'],
                baseline_results['predictions'],
                evaluation_results['targets']
            )
        
        # Generate report
        report = self.generate_holdout_report(
            evaluation_results,
            bootstrap_results,
            baseline_comparison
        )
        
        return report
