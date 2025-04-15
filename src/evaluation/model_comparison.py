import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import roc_curve, precision_recall_curve
from ..analysis.statistical_tests import StatisticalAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelComparator:
    """Class for comparing multiple models' performance."""
    
    def __init__(self, save_dir: str = 'results/model_comparison'):
        """Initialize model comparator."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stats_analyzer = StatisticalAnalyzer()
    
    def convert_to_native(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_native(item) for item in obj]
        return obj
    
    def compare_predictions(self, predictions: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compare predictions from different models using statistical tests."""
        results = {}
        model_names = list(predictions.keys())
        
        # Skip comparison if only one class is present
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            print("Warning: Only one class present in labels, skipping statistical comparison")
            return {}
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                try:
                    # Ensure predictions only contain classes present in labels
                    pred1 = np.clip(predictions[model1], min(unique_labels), max(unique_labels))
                    pred2 = np.clip(predictions[model2], min(unique_labels), max(unique_labels))
                    
                    mcnemar_results = self.stats_analyzer.mcnemar_test(
                        pred1,
                        pred2,
                        labels
                    )
                    results[f"{model1}_vs_{model2}"] = {
                        'statistic': float(mcnemar_results[0]),
                        'p_value': float(mcnemar_results[1]),
                        'significant': float(mcnemar_results[1]) < 0.05
                    }
                except Exception as e:
                    print(f"Warning: Failed to compare {model1} vs {model2}: {str(e)}")
                    results[f"{model1}_vs_{model2}"] = {
                        'statistic': 0.0,
                        'p_value': 1.0,
                        'significant': False
                    }
        
        return results
    
    def compare_roc_curves(self,
                          probabilities: Dict[str, np.ndarray],
                          labels: np.ndarray,
                          save_name: str = 'roc_comparison.png'):
        """Compare ROC curves from multiple models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, probs in probabilities.items():
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            plt.plot(fpr, tpr, label=f'{model_name}')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def compare_pr_curves(self,
                         probabilities: Dict[str, np.ndarray],
                         labels: np.ndarray,
                         save_name: str = 'pr_comparison.png'):
        """Compare Precision-Recall curves from multiple models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, probs in probabilities.items():
            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def compare_metrics(self,
                       metrics: Dict[str, Dict[str, float]],
                       save_name: str = 'metrics_comparison.png'):
        """Compare multiple metrics across models."""
        plt.figure(figsize=(12, 6))
        
        df = pd.DataFrame(metrics).T
        df.plot(kind='bar', width=0.8)
        
        plt.title('Model Metrics Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def compare_confusion_matrices(self,
                                 predictions: Dict[str, np.ndarray],
                                 labels: np.ndarray,
                                 save_prefix: str = 'confusion_matrix'):
        """Compare confusion matrices from multiple models."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        for model_name, preds in predictions.items():
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(labels, preds)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(self.save_dir / f'{save_prefix}_{model_name}.png')
            plt.close()
    
    def compare_calibration(self,
                          probabilities: Dict[str, np.ndarray],
                          labels: np.ndarray,
                          n_bins: int = 10,
                          save_name: str = 'calibration_comparison.png'):
        """Compare calibration curves from multiple models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, probs in probabilities.items():
            # Calculate calibration curve
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            true_probs = []
            mean_predicted_probs = []
            
            for i in range(len(bin_edges) - 1):
                mask = (probs[:, 1] >= bin_edges[i]) & (probs[:, 1] < bin_edges[i + 1])
                if np.any(mask):
                    true_probs.append(np.mean(labels[mask]))
                    mean_predicted_probs.append(np.mean(probs[mask, 1]))
            
            plt.plot(mean_predicted_probs, true_probs, marker='o', label=model_name)
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def generate_comparison_report(self,
                                 predictions: Dict[str, np.ndarray],
                                 probabilities: Dict[str, np.ndarray],
                                 labels: np.ndarray,
                                 metrics: Dict[str, Dict[str, float]],
                                 save_path: str = 'comparison_report.json') -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        # Statistical comparisons
        statistical_comparison = self.compare_predictions(predictions, labels)
        
        # Generate all plots
        self.compare_roc_curves(probabilities, labels)
        self.compare_pr_curves(probabilities, labels)
        self.compare_metrics(metrics)
        self.compare_confusion_matrices(predictions, labels)
        self.compare_calibration(probabilities, labels)
        
        # Compile report
        report = {
            'statistical_comparison': self.convert_to_native(statistical_comparison),
            'metrics': self.convert_to_native(metrics),
            'plots': self.convert_to_native({
                'roc_curve': str(self.save_dir / 'roc_comparison.png'),
                'pr_curve': str(self.save_dir / 'pr_comparison.png'),
                'metrics_comparison': str(self.save_dir / 'metrics_comparison.png'),
                'calibration': str(self.save_dir / 'calibration_comparison.png')
            })
        }
        
        # Save report
        import json
        with open(self.save_dir / save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report
