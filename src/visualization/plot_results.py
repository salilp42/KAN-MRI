import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
from pathlib import Path
import torch

class PerformancePlotter:
    """Class for plotting model performance metrics."""
    def __init__(self, save_dir: str):
        """
        Initialize plotter.
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style without interactive elements
        plt.style.use('default')
        plt.ioff()  # Turn off interactive mode
        # Configure common plot settings
        plt.rcParams.update({
            'figure.figsize': [10, 6],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
    
    def plot_learning_curves(self,
                           history: Dict[str, List[float]],
                           title: str = "Learning Curves",
                           save_name: str = "learning_curves.png"):
        """Plot training and validation metrics over epochs."""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f"{title} - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title(f"{title} - Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            classes: List[str],
                            title: str = "Confusion Matrix",
                            save_name: str = "confusion_matrix.png"):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_scores: Union[np.ndarray, List[np.ndarray]],
                       model_names: Optional[List[str]] = None,
                       title: str = "ROC Curves",
                       save_name: str = "roc_curves.png"):
        """Plot ROC curves for one or multiple models."""
        plt.figure(figsize=(8, 6))
        
        if isinstance(y_scores, list):
            if model_names is None:
                model_names = [f"Model {i+1}" for i in range(len(y_scores))]
            
            for scores, name in zip(y_scores, model_names):
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        else:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_distribution(self,
                                   y_scores: np.ndarray,
                                   y_true: np.ndarray,
                                   title: str = "Prediction Distribution",
                                   save_name: str = "pred_distribution.png"):
        """Plot distribution of model predictions."""
        plt.figure(figsize=(10, 6))
        
        for i, label in enumerate(['Negative', 'Positive']):
            mask = y_true == i
            sns.kdeplot(y_scores[mask], label=f'True {label}')
        
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self,
                              metrics: Dict[str, Dict[str, float]],
                              title: str = "Model Comparison",
                              save_name: str = "model_comparison.png"):
        """Plot comparison of different metrics across models."""
        metrics_df = pd.DataFrame(metrics).T
        
        plt.figure(figsize=(10, 6))
        metrics_df.plot(kind='bar', width=0.8)
        
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_time(self,
                          epoch_times: List[float],
                          title: str = "Training Time per Epoch",
                          save_name: str = "training_time.png"):
        """Plot training time per epoch."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(epoch_times, marker='o')
        plt.axhline(y=np.mean(epoch_times), color='r', linestyle='--',
                   label=f'Mean: {np.mean(epoch_times):.2f}s')
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_rate(self,
                          lrs: List[float],
                          losses: List[float],
                          title: str = "Learning Rate vs Loss",
                          save_name: str = "lr_loss.png"):
        """Plot learning rate finder results."""
        plt.figure(figsize=(10, 6))
        
        plt.semilogx(lrs, losses)
        plt.grid(True)
        plt.title(title)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ensemble_weights(self,
                            weights: List[float],
                            model_names: Optional[List[str]] = None,
                            title: str = "Ensemble Model Weights",
                            save_name: str = "ensemble_weights.png"):
        """Plot ensemble model weights."""
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(weights))]
        
        plt.figure(figsize=(10, 6))
        
        plt.bar(model_names, weights)
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel("Weight")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
