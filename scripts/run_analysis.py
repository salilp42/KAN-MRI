import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import traceback
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score

# Add the repository root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Updated model imports
from src.models.resnet import ResNet2D, ResNet3D
from src.models.vgg import VGG2D, VGG3D
from src.models.convkan import ConvKAN2D, ConvKAN3D
from src.models.gnn import GNN2D, GNN3D

# Updated evaluation and visualization imports
from src.evaluation.power_analysis import PowerAnalyzer
from src.evaluation.model_comparison import ModelComparator
from src.evaluation.holdout import HoldoutEvaluator
from src.visualization.feature_visualization import FeatureVisualizer
from src.visualization.plot_results import PerformancePlotter
from src.evaluation.statistical_tests import StatisticalAnalyzer
from src.evaluation.metrics import MetricsCalculator

# Updated data loading imports
from src.datasets.data_loaders import (
    load_ppmi_data,
    load_taowu_data,
    load_neurocon_data,
    create_data_loaders
)
from src.datasets.mri_dataset import MRIDataset

from tqdm.auto import tqdm
import argparse
import gc
import traceback
import copy
import torch.optim as optim
import torch.nn as nn
import scipy.stats as st
from collections import Counter

class ComprehensiveAnalyzer:
    """Class for running comprehensive analysis of MRI models."""
    
    def __init__(self, base_output_dir: str = 'analysis_results'):
        """Initialize analyzers and create output directories."""
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all analyzers
        self.model_comparator = ModelComparator(save_dir=str(self.base_dir / 'model_comparison'))
        self.holdout_evaluator = HoldoutEvaluator(save_dir=str(self.base_dir / 'holdout'))
        self.feature_visualizer = FeatureVisualizer(save_dir=str(self.base_dir / 'visualizations'))
        self.power_analyzer = PowerAnalyzer(save_dir=str(self.base_dir / 'power_analysis'))
        self.stats_analyzer = StatisticalAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.performance_plotter = PerformancePlotter(save_dir=str(self.base_dir / 'performance_plots'))
    
    def run_isolated_analysis(self,
                            models: dict,
                            dataset,
                            test_loader: torch.utils.data.DataLoader,
                            device: str = 'cuda') -> dict:
        """Run isolated analysis for each model, calculating metrics and inference time."""
        results = {}
        all_predictions = {}
        all_inference_times = {}

        print("\nRunning Isolated Model Testing:")
        for name, model in tqdm(models.items(), desc="Testing models"): 
            print(f"\n{name}:")
            model.eval()
            model = model.to(device)
            
            # Create model-specific directory for plots
            model_dir = self.base_dir / 'model_analysis' / name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Test basic forward pass
            print("Testing forward pass...")
            try:
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    outputs = model.predict(inputs)  # Use predict instead of forward
                    if outputs.shape[1] != 2:  # Check output dimensions
                        raise ValueError(f"Expected 2 output classes, got {outputs.shape[1]}")
                    # break # Removed erroneous break
                    continue
                print("✓ Forward pass successful")
            except Exception as e:
                print(f"✗ Forward pass failed: {str(e)}")
                continue
            
            # Test predictions
            print("Testing predictions...")
            try:
                start_inference_time = time.time()
                if '2D' in name:
                    # Use slice aggregation for 2D models
                    y_true, y_pred, y_prob = predict_with_slice_aggregation(model, test_loader, device)
                    num_subjects = len(y_true)
                else:
                    y_true, y_pred, y_prob = self.get_predictions(model, test_loader, device)
                    num_subjects = len(y_true)
                inference_duration = time.time() - start_inference_time
                inference_time_per_subject = inference_duration / num_subjects if num_subjects > 0 else 0
                all_inference_times[name] = inference_time_per_subject

                all_predictions[name] = {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}
                print(f"✓ Predictions successful (Inference time: {inference_time_per_subject:.4f}s/subject)")
            except Exception as e:
                print(f"✗ Prediction failed: {str(e)}")
                traceback.print_exc()
                continue

            # Calculate metrics
            try:
                metrics = self.metrics_calculator.calculate_basic_metrics(y_true, y_pred, y_prob)
                results[name] = {'metrics': metrics}
                # Add inference time to results
                results[name]['inference_time_s_per_subject'] = inference_time_per_subject
                print(f"✓ Metrics calculation successful (AUC: {metrics.get('auc_roc', 'N/A'):.3f})")
            except Exception as e:
                print(f"✗ Metrics calculation failed: {str(e)}")
                continue
            
        return results

    def run_complete_analysis(self, models, data_loader, device, save_dir):
        """Run complete analysis pipeline, including training and evaluation."""
        analysis_results = {}
        trained_models = {}
        training_times = {}

        # Assuming data_loader yields (train_loader, val_loader, test_loader)
        # This structure needs confirmation based on how data is split
        train_loader, val_loader, test_loader = data_loader 

        for name, model_instance in models.items():
            print(f"\n--- Analyzing Model: {name} ---")
            model = model_instance.to(device)
            
            # --- Training --- (Assuming a config exists for each model)
            model_config = {'label_smoothing': 0.1, 'use_class_weights': True, 'weight_decay': 0.01} # Placeholder: Need actual config loading
            print("Starting training...")
            # Pass relevant config options to train_model
            trained_model, avg_epoch_time = train_model(model, train_loader, val_loader, device, config=model_config)
            trained_models[name] = trained_model
            training_times[name] = avg_epoch_time

            # --- Evaluation --- 
            print("Starting evaluation...")
            start_inference_time = time.time()
            if '2D' in name:
                y_true, y_pred, y_prob = predict_with_slice_aggregation(trained_model, test_loader, device)
                num_subjects = len(y_true)
            else:
                y_true, y_pred, y_prob = self.get_predictions(trained_model, test_loader, device)
                num_subjects = len(y_true)
            inference_duration = time.time() - start_inference_time
            inference_time_per_subject = inference_duration / num_subjects if num_subjects > 0 else 0

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_basic_metrics(y_true, y_pred, y_prob)
            
            analysis_results[name] = {
                'metrics': metrics,
                'predictions': {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob},
                'avg_training_time_s_per_epoch': avg_epoch_time,
                'inference_time_s_per_subject': inference_time_per_subject
            }

            # Save model checkpoint (optional)
            # save_checkpoint(trained_model.state_dict(), Path(save_dir) / f"{name}_final.pth")

            # --- Visualization & Comparison (using ModelComparator, etc.) --- 
            # These would use the collected 'analysis_results'
            print(f"Finished analysis for {name}.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # After analyzing all models, perform comparisons
        # self.model_comparator.plot_roc_curves(analysis_results, ...)
        # self.performance_plotter.plot_metric_bars(analysis_results, ...)
        # ... other comparison plots and statistical tests ...

        return analysis_results

def predict_with_slice_aggregation(model, data_loader, device, confidence_scale=1.0):
    """Make predictions with slice aggregation for 2D models using confidence weighting."""
    model.eval()
    all_probs = []
    all_labels = []
    all_subject_ids = []

    with torch.no_grad():
        for images, labels, subject_ids in tqdm(data_loader, desc='Aggregating Slices', leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subject_ids.extend(subject_ids)

    # Group by subject
    subject_data = {}
    for i, subject_id in enumerate(all_subject_ids):
        if subject_id not in subject_data:
            subject_data[subject_id] = {'probs': [], 'label': all_labels[i]}
        subject_data[subject_id]['probs'].append(all_probs[i])

    # Aggregate per subject
    subject_level_probs = []
    subject_level_labels = []
    for subject_id, data in subject_data.items():
        slice_probs = np.array(data['probs'])
        confidences = np.abs(slice_probs - 0.5)

        # Sort by confidence
        sorted_indices = np.argsort(confidences)[::-1] # Descending order

        # Select top 33%
        num_slices = len(slice_probs)
        top_k = max(1, int(np.ceil(0.33 * num_slices))) # Ensure at least 1 slice
        top_indices = sorted_indices[:top_k]

        # Exponential confidence weighting
        selected_probs = slice_probs[top_indices]
        selected_confidences = confidences[top_indices]
        weights = np.exp(confidence_scale * selected_confidences)

        # Weighted average of probabilities
        if np.sum(weights) > 0:
            aggregated_prob = np.sum(selected_probs * weights) / np.sum(weights)
        else: # Handle case where all weights are zero (e.g., confidences are all zero)
            aggregated_prob = np.mean(selected_probs) if len(selected_probs) > 0 else 0.5

        subject_level_probs.append(aggregated_prob)
        subject_level_labels.append(data['label'])

    # Return probabilities for class 1 and true labels
    # Reshape probs to be [n_subjects, 2] if needed by downstream metrics
    final_probs = np.array(subject_level_probs)
    final_labels = np.array(subject_level_labels)
    # Create a [N, 2] array for compatibility with metrics expecting multi-class probs
    probs_2_class = np.vstack([1 - final_probs, final_probs]).T

    # Return predicted labels (thresholded) and probabilities
    # Thresholding might be done later based on optimal threshold
    y_pred = (final_probs > 0.5).astype(int)

    return final_labels, y_pred, probs_2_class

def calculate_class_weights(data_loader):
    """Calculate class weights based on inverse frequency."""
    counter = Counter()
    print("Calculating class weights...")
    for _, labels, *_ in tqdm(data_loader, desc='Counting classes', leave=False): # Adjusted loop
        counter.update(labels.tolist())

    if not counter:
        print("Warning: Could not calculate class weights, dataset might be empty.")
        return None

    total_count = sum(counter.values())
    num_classes = len(counter)
    weights = torch.zeros(num_classes, dtype=torch.float)

    for class_idx, count in counter.items():
        if count > 0:
            weights[class_idx] = total_count / (num_classes * count)
        else:
             weights[class_idx] = 0 # Should not happen if class is present

    print(f"Calculated class weights: {weights.tolist()}")
    return weights

def train_model(model, train_loader, val_loader, device, num_epochs=100, debug=False, config=None):
    """Train the model with learning rate scheduling, early stopping, label smoothing, and class weighting."""
    config = config or {}
    lr = config.get('learning_rate', 0.001)
    patience = config.get('early_stopping_patience', 15)
    label_smoothing = config.get('label_smoothing', 0.0)
    use_class_weights = config.get('use_class_weights', False)
    weight_decay = config.get('weight_decay', 0.0) # Added weight decay

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Added weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(train_loader)
        if class_weights is not None:
            class_weights = class_weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    epoch_times = []

    # Add progress bar for epochs
    pbar = tqdm(range(num_epochs), desc='Training epochs')

    try:
        for epoch in pbar:
            start_time = time.time()
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Add progress bar for training batches
            train_iter = tqdm(train_loader, desc=f'Training batches (epoch {epoch+1})', leave=False)
            
            for images, labels, *_ in train_iter:
                try:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    train_iter.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*train_correct/train_total:.2f}%'
                    })
                    
                    # Clear memory
                    del images, labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
            
        # Add progress bar for validation batches
        val_iter = tqdm(val_loader, desc=f'Validation batches (epoch {epoch+1})', leave=False)
        
        with torch.no_grad():
            for images, labels, *_ in val_iter:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    val_iter.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
                    
                    # Clear memory
                    del images, labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping check
            val_loss_avg = val_loss / len(val_loader)
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch}')
                    break
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss/len(train_loader):.4f}',
                'train_acc': f'{100.*train_correct/train_total:.2f}%',
                'val_loss': f'{val_loss_avg:.4f}',
                'val_acc': f'{100.*val_correct/val_total:.2f}%',
                'epoch_time': f'{time.time() - start_time:.2f}s' # Add epoch time to pbar
            })
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            epoch_times.append(time.time() - start_time)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    if best_model_state:
        model.load_state_dict(best_model_state)

    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    print(f"\nTraining finished. Best validation loss: {best_val_loss:.4f}, Average epoch time: {avg_epoch_time:.2f}s")
    return model, avg_epoch_time # Return trained model and avg time

def create_data_loaders(data, labels, batch_size, mode='2d', debug=False, config=None, subject_ids=None):
    """Create data loaders for training and validation."""
    # For 2D data with all slices, replicate labels
    if mode == '2d' and len(data.shape) == 4:  # (N*D, 1, H, W)
        # Calculate number of slices per sample
        total_slices = data.shape[0]
        n_samples = len(labels)
        slices_per_sample = total_slices // n_samples
        
        # Replicate labels to match the number of slices
        labels = np.repeat(labels, slices_per_sample)
        
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
        print(f"Samples: {n_samples}, Slices per sample: {slices_per_sample}")
    
    # Verify data and label lengths match
    assert len(data) == len(labels), f"Data length ({len(data)}) does not match labels length ({len(labels)})"
    
    # Split data into training and validation
    indices = np.random.permutation(len(data))
    split = int(0.8 * len(data))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets
    train_dataset = MRIDataset(
        data[train_indices], 
        labels[train_indices], 
        mode=mode,
        subject_ids=subject_ids[train_indices] if subject_ids is not None else None,
        augment=config.get('use_augmentation', False) if (debug and config is not None) else False
    )
    val_dataset = MRIDataset(
        data[val_indices], 
        labels[val_indices], 
        mode=mode,
        subject_ids=subject_ids[val_indices] if subject_ids is not None else None,
        augment=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader

def initialize_models(device, debug_mode=False):
    """Initialize all models."""
    print("\nInitializing models...")
    
    # Common configuration
    config = {
        'in_channels': 1,  # Single channel for MRI
        'num_classes': 2,  # Binary classification
        'input_shape': (1, 16, 16, 16) if debug_mode else (1, 64, 64, 64),  # Smaller shape for debug
        'normalize_mean': [0.485],  # Using only first channel from ImageNet
        'normalize_std': [0.229],   # Using only first channel from ImageNet
        'num_node_features': 3,     # For GNN models
        'hidden_channels': 32 if debug_mode else 64,  # Smaller for debug mode
        'pretrained': False if debug_mode else True,   # No pretrained in debug mode
        # Add VGG-specific configurations
        'vgg_fc_size': 512 if debug_mode else 4096,  # Smaller FC layers for debug mode
        'vgg_input_size': 32 if debug_mode else 224,  # Smaller input size for debug mode
        'vgg3d_input_size': 16 if debug_mode else 64  # Smaller 3D input size for debug mode
    }
    
    models = {}
    model_classes = {
        # 3D models (except VGG3D)
        'ResNet3D': ResNet3D,
        'ConvKAN3D': ConvKAN3D,
        'GNN3D': GNN3D,
        # 2D models
        'ResNet2D': ResNet2D,
        'VGG2D': VGG2D,
        'ConvKAN2D': ConvKAN2D,
        'GNN2D': GNN2D
    }
    
    with tqdm(total=len(model_classes), desc="Initializing models") as pbar:
        for name, model_class in model_classes.items():
            try:
                print(f"\nInitializing {name}...")
                
                # Initialize model
                model = model_class(config=config)
                model = model.to(device)
                
                # Test with small input
                if '2D' in name:
                    test_input = torch.randn(1, 1, config['vgg_input_size'], config['vgg_input_size']).to(device)  # 2D input
                else:
                    test_input = torch.randn(1, 1, config['vgg3d_input_size'], config['vgg3d_input_size'], config['vgg3d_input_size']).to(device)  # 3D input
                
                with torch.no_grad():
                    outputs = model(test_input)
                    if outputs.shape[1] != 2:  # Check output dimensions
                        raise ValueError(f"Expected 2 output classes, got {outputs.shape[1]}")
                print(f"✓ Model test successful")
                
                models[name] = model
                print(f"✓ Successfully initialized {name}")
                
            except Exception as e:
                print(f"✗ Failed to initialize {name}: {str(e)}")
                continue
            finally:
                # Clear memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                pbar.update(1)
    
    if not models:
        raise RuntimeError("No models were successfully initialized")
    
    return models

def process_data_for_dimension(data, dimension, debug_mode=False):
    """Process data for a specific dimension (2D or 3D)."""
    if isinstance(data, dict):
        input_data = data['data']
        labels = data['labels']
    else:
        input_data = data
        labels = np.zeros(len(data))  # Default labels if not provided
    
    # Get data shape information
    if len(input_data.shape) == 4:  # (N, D, H, W)
        n_samples, depth, height, width = input_data.shape
    elif len(input_data.shape) == 5:  # (N, C, D, H, W)
        n_samples, channels, depth, height, width = input_data.shape
        input_data = input_data.squeeze(1)  # Remove channel dimension if present
    else:
        raise ValueError(f"Unexpected input shape: {input_data.shape}")
    
    print(f"Processed data shape: {input_data.shape}")
    print(f"Number of samples: {n_samples}, Depth: {depth}")
    
    if dimension.lower() == '2d':
        # For 2D, we'll create slices from the 3D volume
        processed_data = input_data.reshape(n_samples * depth, 1, height, width)
        processed_labels = np.repeat(labels, depth)
    else:  # 3D
        # For 3D, we'll use a subset of slices to reduce memory
        target_depth = 32 if debug_mode else depth
        if depth > target_depth:
            # Take evenly spaced slices
            indices = np.linspace(0, depth-1, target_depth, dtype=int)
            input_data = input_data[:, indices, :, :]
        
        # Add channel dimension for 3D
        processed_data = input_data.reshape(n_samples, 1, -1, height, width)
        processed_labels = labels
    
    return {
        'data': processed_data,
        'labels': processed_labels,
        'mode': dimension
    }

def run_analysis_with_holdout(train_data, test_data, holdout_name, debug=False, config=None):
    """Run analysis with holdout data."""
    results = {}
    
    try:
        # Process data for both 2D and 3D
        train_data_2d = process_data_for_dimension(train_data, '2d', debug)
        train_data_3d = process_data_for_dimension(train_data, '3d', debug)
        test_data_2d = process_data_for_dimension(test_data, '2d', debug)
        test_data_3d = process_data_for_dimension(test_data, '3d', debug)
        
        # Initialize models
        print("\nInitializing models...")
        models = initialize_models('cpu', debug)  # Use CPU for debug mode
        
        # Create data loaders for both dimensions
        train_loader_2d, val_loader_2d = create_data_loaders(
            train_data_2d['data'], 
            train_data_2d['labels'],
            batch_size=4 if debug else 32,
            mode='2d',
            debug=debug,
            config=config
        )
        
        train_loader_3d, val_loader_3d = create_data_loaders(
            train_data_3d['data'],
            train_data_3d['labels'],
            batch_size=4 if debug else 32,
            mode='3d',
            debug=debug,
            config=config
        )
        
        test_loader_2d, _ = create_data_loaders(
            test_data_2d['data'],
            test_data_2d['labels'],
            batch_size=4 if debug else 32,
            mode='2d',
            debug=debug,
            config=config
        )
        
        test_loader_3d, _ = create_data_loaders(
            test_data_3d['data'],
            test_data_3d['labels'],
            batch_size=4 if debug else 32,
            mode='3d',
            debug=debug,
            config=config
        )
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            try:
                # Select appropriate data loaders based on model dimension
                if '2d' in model_name.lower():
                    train_loader = train_loader_2d
                    val_loader = val_loader_2d
                    test_loader = test_loader_2d
                else:
                    train_loader = train_loader_3d
                    val_loader = val_loader_3d
                    test_loader = test_loader_3d
                
                # Train model
                model = train_model(
                    model,
                    train_loader,
                    val_loader,
                    'cpu',  # Use CPU for debug mode
                    num_epochs=2 if debug else 100,
                    debug=debug,
                    config=config
                )
                
                # Get predictions
                predictions = []
                probabilities = []
                targets = []
                
                model.eval()
                with torch.no_grad():
                    for batch_data, batch_labels, *_ in test_loader:
                        batch_data = batch_data.to('cpu')
                        outputs = model(batch_data)
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)
                        
                        predictions.extend(preds.cpu().numpy())
                        probabilities.extend(probs.cpu().numpy())
                        targets.extend(batch_labels.numpy())
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                metrics = {
                    'accuracy': accuracy_score(targets, predictions),
                    'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
                    'f1': f1_score(targets, predictions, average='weighted', zero_division=0)
                }
                
                try:
                    metrics['auc'] = roc_auc_score(targets, np.array(probabilities)[:, 1])
                except:
                    metrics['auc'] = 0.5  # Default value if AUC cannot be calculated
                
                results[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'targets': targets,
                    'metrics': metrics
                }
                
                print("✓ Successfully evaluated", model_name)
                
            except Exception as e:
                print(f"✗ Failed to evaluate {model_name}: {str(e)}")
                results[model_name] = {
                    'error': str(e)
                }
                continue
    
    except Exception as e:
        print(f"Error in holdout analysis: {str(e)}")
        results['error'] = str(e)
    
    return results

def run_analysis_for_dimension(dimension, dataset_name, dataset, debug_mode=False):
    """Run analysis for specified dimension(s)."""
    print(f"\nStarting {'both 2D and 3D' if dimension == 'both' else dimension} analysis...")
    
    # Unpack dataset tuple and convert to numpy arrays if needed
    data, labels, identifiers = dataset
    if isinstance(data, str):
        print(f"Error: Received string instead of data array for {dataset_name}")
        return {}
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Debug configuration
    debug_config = {
        'batch_size': 4,
        'epochs': 2,
        'learning_rate': 0.001,
        'early_stopping_patience': 3,
        'use_augmentation': False,
        'augmentation_prob': 0.0
    }
    
    results = {}
    dimensions_to_process = ['2d', '3d'] if dimension == 'both' else [dimension.lower()]
    
    for dim in dimensions_to_process:
        try:
            # Process current dimension
            print(f"\nProcessing {dim} analysis...")
            processed_data = process_data_for_dimension(
                {'data': data, 'labels': labels}, 
                dim, 
                debug_mode
            )
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                processed_data['data'],
                processed_data['labels'],
                batch_size=4 if debug_mode else 32,
                mode=dim,
                debug=debug_mode,
                config=debug_config
            )
            
            # Initialize models for this dimension
            device = 'cpu' if debug_mode else 'cuda'
            all_models = initialize_models(device, debug_mode=debug_mode)
            dimension_models = {name: model for name, model in all_models.items() 
                             if dim in name.lower()}
            
            # Train and evaluate models
            dimension_results = {}
            for name, model in dimension_models.items():
                try:
                    print(f"\nTraining {name}...")
                    # Train model
                    trained_model = train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        num_epochs=2 if debug_mode else 100,
                        debug=debug_mode,
                        config=debug_config
                    )
                    
                    # Evaluate model
                    evaluator = HoldoutEvaluator()
                    evaluation_results = evaluator.evaluate_model(
                        model=trained_model,
                        test_loader=val_loader,
                        device=device
                    )
        
                    dimension_results[name] = evaluation_results
                    print(f"✓ Successfully evaluated {name}")
                except Exception as e:
                    print(f"✗ Failed to evaluate {name}: {str(e)}")
                    continue
            
            results[dim] = dimension_results
        except Exception as e:
            print(f"✗ Failed to process {dim} analysis: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            continue
    
    return results

def get_timestamp_dir():
    """Create timestamped directory name."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_checkpoint(data, filepath):
    """Save checkpoint with error handling."""
    try:
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        torch.save(data, filepath)
        print(f"Checkpoint saved: {filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")

def main(debug_mode=False):
    """Run the analysis pipeline with three types of analyses."""
    # Create timestamped output directory
    timestamp = get_timestamp_dir()
    output_dir = Path(f'analysis_results_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize results dictionary
    all_results = {
        'independent_analysis': {},
        'holdout_analysis': {},
        'combined_analysis': {},
        'errors': []
    }
    
    try:
        # Load all datasets
        print("\nLoading datasets...")
        datasets = {}
        dataset_info = {}
        
        # Dataset configurations with correct paths
        dataset_configs = {
            'ppmi': {
                'data_dir': 'data/ppmi',
                'info_csv': 'data/ppmi/ppmi_info.csv'
            },
            'taowu': {
                'data_dir': 'data/taowu',
                'info_csv': 'data/taowu/taowu_patients.tsv'
            },
            'neurocon': {
                'data_dir': 'data/Raw Data',  # Fixed path for NEUROCON
                'info_csv': 'data/Raw Data/neurocon_info.csv'
            }
        }
        
        for dataset_name in ['ppmi', 'taowu', 'neurocon']:
            try:
                config = dataset_configs[dataset_name]
                config['debug_config'] = {'samples_per_dataset': 100} if debug_mode else None
                
                if dataset_name == 'ppmi':
                    datasets[dataset_name] = load_ppmi_data(config)
                elif dataset_name == 'taowu':
                    datasets[dataset_name] = load_taowu_data(config)
                else:  # neurocon
                    datasets[dataset_name] = load_neurocon_data(config)
                
                print(f"Successfully loaded {dataset_name} dataset")
            except Exception as e:
                error_msg = f"Error loading {dataset_name} dataset: {str(e)}"
                print(error_msg)
                all_results['errors'].append(error_msg)
                continue
        
        if not datasets:
            raise RuntimeError("No datasets were successfully loaded")
        
        # 1. Independent Dataset Analysis
        print("\n=== Running Independent Dataset Analysis ===")
        for dataset_name, dataset in datasets.items():
            try:
                print(f"\nAnalyzing {dataset_name} dataset...")
                results = run_analysis_for_dimension(
                    dimension="both",
                    dataset_name=dataset_name,
                    dataset=dataset,
                    debug_mode=debug_mode
                )
                all_results['independent_analysis'][dataset_name] = results
            except Exception as e:
                error_msg = f"Error in independent analysis for {dataset_name}: {str(e)}"
                print(error_msg)
                all_results['errors'].append(error_msg)
                continue
        
        # 2. Two vs One Analysis
        print("\n=== Running Two vs One Analysis ===")
        for holdout_dataset in datasets.keys():
            try:
                print(f"\nHolding out {holdout_dataset}...")
                # Combine training datasets
                train_datasets = {name: datasets[name] for name in datasets if name != holdout_dataset}
                combined_data = combine_datasets(train_datasets)
                
                # Convert test_data tuple to dictionary format
                test_data_tuple = datasets[holdout_dataset]
                test_data = {
                    'data': test_data_tuple[0],
                    'labels': test_data_tuple[1],
                    'identifiers': test_data_tuple[2],
                    'mode': combined_data.get('mode', '2d')
                }
                
                results = run_analysis_with_holdout(
                    train_data=combined_data,
                    test_data=test_data,
                    holdout_name=holdout_dataset,
                    debug=debug_mode
                )
                all_results['holdout_analysis'][holdout_dataset] = results
            except Exception as e:
                error_msg = f"Error in holdout analysis for {holdout_dataset}: {str(e)}"
                print(error_msg)
                all_results['errors'].append(error_msg)
                continue
        
        # 3. Combined Dataset Analysis
        print("\n=== Running Combined Dataset Analysis ===")
        try:
            combined_data = combine_datasets(datasets)
            combined_results = run_analysis_for_dimension(
                dimension="both",
                dataset_name="combined",
                dataset=(combined_data['data'], combined_data['labels'], combined_data['identifiers']),
                debug_mode=debug_mode
            )
            all_results['combined_analysis'] = combined_results
        except Exception as e:
            error_msg = f"Error in combined analysis: {str(e)}"
            print(error_msg)
            all_results['errors'].append(error_msg)
        
        # Save all results
        try:
            with open(output_dir / 'all_results.json', 'w') as f:
                json.dump(convert_to_serializable(all_results), f, indent=4)
            
            # Create summary DataFrame if we have results
            if all_results['independent_analysis'] or all_results['holdout_analysis'] or all_results['combined_analysis']:
                summary_df = create_summary_dataframe(
                    independent_results=all_results['independent_analysis'],
                    holdout_results=all_results['holdout_analysis'],
                    combined_results=all_results['combined_analysis']
                )
                summary_df.to_csv(output_dir / 'analysis_summary.csv', index=True)
        except Exception as e:
            print(f"Error saving results: {str(e)}")
        
        # Save completion status
        with open(output_dir / 'completion_status.json', 'w') as f:
            status = {
                'status': 'complete with errors' if all_results['errors'] else 'complete',
                'timestamp': get_timestamp_dir(),
                'analyses_completed': {
                    'independent_dataset': bool(all_results['independent_analysis']),
                    'two_vs_one': bool(all_results['holdout_analysis']),
                    'combined_dataset': bool(all_results['combined_analysis'])
                },
                'errors': all_results['errors'],
                'debug_mode': debug_mode
            }
            json.dump(status, f, indent=4)
        
        print("\nAnalysis complete! Results saved in:", output_dir)
        if all_results['errors']:
            print("\nWarning: Some analyses encountered errors. Check completion_status.json for details.")
    
    except Exception as e:
        print(f"\nCritical error in analysis pipeline: {str(e)}")
        with open(output_dir / 'completion_status.json', 'w') as f:
            json.dump({
                'status': 'failed',
                'timestamp': get_timestamp_dir(),
                'error': str(e),
                'debug_mode': debug_mode
            }, f, indent=4)
        print("Partial results (if any) saved in:", output_dir)

def combine_datasets(datasets):
    """Combine multiple datasets into one."""
    combined_data = []
    combined_labels = []
    combined_identifiers = []
    
    for name, (data, labels, identifiers) in datasets.items():
        combined_data.append(data)
        combined_labels.append(labels)
        combined_identifiers.extend([f"{name}_{id}" for id in identifiers])
    
    return {
        'data': np.concatenate(combined_data, axis=0),
        'labels': np.concatenate(combined_labels, axis=0),
        'identifiers': combined_identifiers,
        'mode': '2d'  # Default mode, can be overridden
    }

def create_summary_dataframe(independent_results, holdout_results, combined_results):
    """Create a comprehensive summary DataFrame of all results with confidence intervals."""
    summary_data = []
    
    # Helper function to calculate confidence intervals
    def calculate_ci(scores, probabilities=None, labels=None):
        # Accuracy CI
        if len(scores) < 2:  # Need at least 2 points for CI
            return (0, 0), (0, 0)
        
        # Calculate accuracy CI using Wilson score interval
        from statsmodels.stats.proportion import proportion_confint
        acc_mean = np.mean(scores)
        acc_ci = proportion_confint(count=int(acc_mean * len(scores)), 
                                  nobs=len(scores),
                                  alpha=0.05,
                                  method='wilson')
        
        # Calculate AUC CI if probabilities are provided
        if probabilities is not None and labels is not None:
            try:
                # Bootstrap AUC CI
                auc_scores = []
                for _ in range(1000):
                    indices = np.random.choice(len(labels), len(labels))
                    auc_scores.append(roc_auc_score(labels[indices], 
                                                  probabilities[indices]))
                auc_ci = np.percentile(auc_scores, [2.5, 97.5])
            except:
                auc_ci = (0.5, 0.5)
        else:
            auc_ci = (0.5, 0.5)
        
        return acc_ci, auc_ci
    
    # Process independent results
    for dataset_name, results in independent_results.items():
        for dimension, dim_results in results.items():
            for model_name, metrics in dim_results.items():
                if 'predictions' in metrics and 'targets' in metrics:
                    # Calculate scores
                    pred_scores = (metrics['predictions'] == metrics['targets']).astype(float)
                    probs = metrics.get('probabilities', None)
                    if probs is not None and probs.shape[1] == 2:
                        probs = probs[:, 1]  # Get positive class probabilities
                    
                    # Calculate CIs
                    acc_ci, auc_ci = calculate_ci(pred_scores, probs, metrics['targets'])
                    
                    # Calculate AUC
                    try:
                        auc_score = roc_auc_score(metrics['targets'], probs) if probs is not None else 0.5
                    except:
                        auc_score = 0.5
                    
                    summary_data.append({
                        'Analysis Type': 'Independent',
                        'Dataset': dataset_name,
                        'Dimension': dimension,
                        'Model': model_name,
                        'Accuracy': metrics.get('basic_metrics', {}).get('accuracy', 0),
                        'Accuracy_CI_Low': acc_ci[0],
                        'Accuracy_CI_High': acc_ci[1],
                        'AUC': auc_score,
                        'AUC_CI_Low': auc_ci[0],
                        'AUC_CI_High': auc_ci[1],
                        'F1': metrics.get('basic_metrics', {}).get('f1', 0),
                        'Balanced_Accuracy': metrics.get('basic_metrics', {}).get('balanced_accuracy', 0),
                        'Specificity': metrics.get('basic_metrics', {}).get('specificity', 0), # Assuming specificity is in metrics
                        'Avg Training Time (s/epoch)': metrics.get('avg_training_time_s_per_epoch', float('nan')),
                        'Inference Time (s/subject)': metrics.get('inference_time_s_per_subject', float('nan'))
                    })
    
    # Process holdout results
    for holdout_name, results in holdout_results.items():
        for model_name, metrics in results.items():
            if 'predictions' in metrics and 'targets' in metrics:
                # Calculate scores
                pred_scores = (metrics['predictions'] == metrics['targets']).astype(float)
                probs = metrics.get('probabilities', None)
                if probs is not None and probs.shape[1] == 2:
                    probs = probs[:, 1]
                
                # Calculate CIs
                acc_ci, auc_ci = calculate_ci(pred_scores, probs, metrics['targets'])
                
                # Calculate AUC
                try:
                    auc_score = roc_auc_score(metrics['targets'], probs) if probs is not None else 0.5
                except:
                    auc_score = 0.5
                
                summary_data.append({
                    'Analysis Type': 'Holdout',
                    'Dataset': f'Test_{holdout_name}',
                    'Dimension': model_name.split('_')[0],  # Extract dimension from model name
                    'Model': model_name,
                    'Accuracy': metrics.get('basic_metrics', {}).get('accuracy', 0),
                    'Accuracy_CI_Low': acc_ci[0],
                    'Accuracy_CI_High': acc_ci[1],
                    'AUC': auc_score,
                    'AUC_CI_Low': auc_ci[0],
                    'AUC_CI_High': auc_ci[1],
                    'F1': metrics.get('basic_metrics', {}).get('f1', 0),
                    'Balanced_Accuracy': metrics.get('basic_metrics', {}).get('balanced_accuracy', 0),
                    'Specificity': metrics.get('basic_metrics', {}).get('specificity', 0), # Assuming specificity is in metrics
                    'Avg Training Time (s/epoch)': metrics.get('avg_training_time_s_per_epoch', float('nan')),
                    'Inference Time (s/subject)': metrics.get('inference_time_s_per_subject', float('nan'))
                })
    
    # Process combined results
    for dimension, dim_results in combined_results.items():
        for model_name, metrics in dim_results.items():
            if 'predictions' in metrics and 'targets' in metrics:
                # Calculate scores
                pred_scores = (metrics['predictions'] == metrics['targets']).astype(float)
                probs = metrics.get('probabilities', None)
                if probs is not None and probs.shape[1] == 2:
                    probs = probs[:, 1]
                
                # Calculate CIs
                acc_ci, auc_ci = calculate_ci(pred_scores, probs, metrics['targets'])
                
                # Calculate AUC
                try:
                    auc_score = roc_auc_score(metrics['targets'], probs) if probs is not None else 0.5
                except:
                    auc_score = 0.5
                
                summary_data.append({
                    'Analysis Type': 'Combined',
                    'Dataset': 'All',
                    'Dimension': dimension,
                    'Model': model_name,
                    'Accuracy': metrics.get('basic_metrics', {}).get('accuracy', 0),
                    'Accuracy_CI_Low': acc_ci[0],
                    'Accuracy_CI_High': acc_ci[1],
                    'AUC': auc_score,
                    'AUC_CI_Low': auc_ci[0],
                    'AUC_CI_High': auc_ci[1],
                    'F1': metrics.get('basic_metrics', {}).get('f1', 0),
                    'Balanced_Accuracy': metrics.get('basic_metrics', {}).get('balanced_accuracy', 0),
                    'Specificity': metrics.get('basic_metrics', {}).get('specificity', 0), # Assuming specificity is in metrics
                    'Avg Training Time (s/epoch)': metrics.get('avg_training_time_s_per_epoch', float('nan')),
                    'Inference Time (s/subject)': metrics.get('inference_time_s_per_subject', float('nan'))
                })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Round numeric columns to 3 decimal places
    numeric_columns = ['Accuracy', 'Accuracy_CI_Low', 'Accuracy_CI_High', 
                      'AUC', 'AUC_CI_Low', 'AUC_CI_High',
                      'F1', 'Balanced_Accuracy', 'Specificity', 
                      'Avg Training Time (s/epoch)', 'Inference Time (s/subject)']
    df[numeric_columns] = df[numeric_columns].round(3)

    # Add formatted columns with CIs
    df['Accuracy (95% CI)'] = df.apply(
        lambda x: f"{x['Accuracy']:.3f} ({x['Accuracy_CI_Low']:.3f}-{x['Accuracy_CI_High']:.3f})", 
        axis=1
    )
    df['AUC (95% CI)'] = df.apply(
        lambda x: f"{x['AUC']:.3f} ({x['AUC_CI_Low']:.3f}-{x['AUC_CI_High']:.3f})", 
        axis=1
    )
    
    # Drop individual CI columns and raw time columns, reorder
    df = df.drop(columns=['Accuracy_CI_Low', 'Accuracy_CI_High', 'AUC_CI_Low', 'AUC_CI_High', 
                         'Avg Training Time (s/epoch)', 'Inference Time (s/subject)'])

    # Reorder columns - Add Specificity and Time columns
    final_columns = [
        'Analysis Type', 'Dataset', 'Dimension', 'Model',
        'Accuracy (95% CI)', 'AUC (95% CI)', 'F1', 'Balanced_Accuracy', 'Specificity',
        # Add formatted time columns if needed, or keep raw times
    ]
    # Ensure all expected columns exist before selecting
    df = df[[col for col in final_columns if col in df.columns]]

    # Sort the DataFrame
    df = df.sort_values(['Analysis Type', 'Dataset', 'Dimension', 'Model'])

    return df

def calculate_power_analysis(best_correct, curr_correct):
    """Calculate power analysis between two models."""
    if len(best_correct) != len(curr_correct):
        print("Warning: Arrays have different lengths")
        return None
    
    try:
        # Calculate effect size using Cohen's d
        mean_diff = np.mean(best_correct) - np.mean(curr_correct)
        pooled_var = (np.var(best_correct) + np.var(curr_correct)) / 2
        effect_size = mean_diff / np.sqrt(pooled_var)
        
        power_results = {}
        
        # Only calculate power curve if effect size is non-zero
        if abs(effect_size) > 0:
            curve_data = self.power_analyzer.sample_size_curve(effect_size=abs(effect_size))
            power_results = {
                'effect_size': float(effect_size),
                'required_n': int(curve_data['required_n'])
            }
        else:
            power_results = {
                'effect_size': 0.0,
                'note': 'No effect size detected between models'
            }
        
        return power_results
    
    except Exception as e:
        print(f"Error in power analysis: {str(e)}")
        return None

def convert_to_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run neuroimaging analysis pipeline')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with minimal dataset')
    # Add arguments for config file path, output dir etc.
    # parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    # Load config file here if using one
    # with open(args.config, 'r') as f:
    #     config = yaml.safe_load(f)

    main(debug_mode=args.debug) # Pass config to main if loaded
