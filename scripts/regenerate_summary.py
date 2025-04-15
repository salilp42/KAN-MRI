import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score
from scipy import stats
from collections import defaultdict
import warnings

def aggregate_2d_predictions_to_participant(slice_predictions, slice_probabilities):
    """
    Aggregate slice-level predictions to participant level using weighted confidence voting.
    
    Args:
        slice_predictions: List of binary predictions for each slice
        slice_probabilities: List of prediction probabilities for each slice
    
    Returns:
        participant_pred: Single prediction for the participant
        participant_prob: Aggregated probability for the participant
    """
    # Convert inputs to numpy arrays
    slice_predictions = np.array(slice_predictions)
    slice_probabilities = np.array(slice_probabilities)
    
    # Add class balance check
    unique_classes = np.unique(slice_predictions)
    if len(unique_classes) == 1:
        warnings.warn("Only one class present in slice predictions")
    
    # If probabilities are 2D (class probabilities), take the probability for class 1
    if len(slice_probabilities.shape) > 1 and slice_probabilities.shape[1] > 1:
        slice_probabilities = slice_probabilities[:, 1]
    
    # Calculate confidence weights based on distance from decision boundary
    confidence_weights = np.abs(slice_probabilities - 0.5)
    
    # Use exponential weighting to emphasize high confidence predictions
    confidence_weights = np.exp(confidence_weights) - 1
    
    # Weighted voting with normalized weights
    weighted_votes = np.sum(slice_predictions * confidence_weights)
    total_weight = np.sum(confidence_weights)
    
    # Final prediction based on weighted majority with threshold adjustment
    threshold = 0.5  # Can be adjusted based on class imbalance
    participant_pred = 1 if weighted_votes / total_weight > threshold else 0
    
    # Aggregate probability using weighted average of most confident predictions
    # Use top 33% most confident predictions to be more robust
    top_k = max(1, int(len(slice_probabilities) * 0.33))
    top_indices = np.argsort(confidence_weights)[-top_k:]
    
    # Weight the probabilities by their confidence
    top_weights = confidence_weights[top_indices]
    top_probs = slice_probabilities[top_indices]
    participant_prob = np.sum(top_probs * top_weights) / np.sum(top_weights)
    
    return participant_pred, participant_prob

def group_by_participant(y_true, y_pred, y_prob, max_slices_per_participant=None):
    """
    Group predictions by participant based on consecutive same labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        max_slices_per_participant: Maximum expected slices per participant
    
    Returns:
        Dictionary mapping participant ID to their data
    """
    participant_groups = defaultdict(list)
    current_participant = 0
    current_label = y_true[0]
    current_group_size = 0
    
    # Estimate max slices if not provided
    if max_slices_per_participant is None:
        # Find most common group size
        group_sizes = []
        current_size = 1
        for i in range(1, len(y_true)):
            if y_true[i] == y_true[i-1]:
                current_size += 1
            else:
                group_sizes.append(current_size)
                current_size = 1
        group_sizes.append(current_size)
        max_slices_per_participant = np.median(group_sizes)
    
    for i, (label, pred, prob) in enumerate(zip(y_true, y_pred, y_prob)):
        if (label != current_label and i > 0) or current_group_size >= max_slices_per_participant:
            current_participant += 1
            current_label = label
            current_group_size = 0
        participant_groups[current_participant].append((label, pred, prob))
        current_group_size += 1
    
    # Validation checks
    expected_min_participants = len(set(y_true))
    if len(participant_groups) < expected_min_participants:
        warnings.warn(
            f"Potential grouping issue: Found {len(participant_groups)} participants "
            f"but expected at least {expected_min_participants}"
        )
    
    return participant_groups

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate all metrics with proper error handling."""
    metrics = {}
    
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics that don't need probabilities
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    acc_ci = 1.96 * np.sqrt((acc * (1 - acc)) / n)
    metrics['Accuracy (95% CI)'] = f"{acc:.3f} ({(acc-acc_ci):.3f}-{(acc+acc_ci):.3f})"
    
    # AUC calculation
    if y_prob is not None:
        try:
            y_prob = np.array(y_prob)
            # Handle 2D probability arrays (take probability of positive class)
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
            
            if len(np.unique(y_true)) > 1:  # Only calculate if both classes present
                auc = roc_auc_score(y_true, y_prob)
                auc_ci = 1.96 * np.sqrt((auc * (1 - auc)) / n)
                metrics['AUC (95% CI)'] = f"{auc:.3f} ({(auc-auc_ci):.3f}-{(auc+auc_ci):.3f})"
            else:
                metrics['AUC (95% CI)'] = "nan (nan-nan)"
        except Exception as e:
            print(f"Warning: AUC calculation failed: {str(e)}")
            metrics['AUC (95% CI)'] = "nan (nan-nan)"
    else:
        metrics['AUC (95% CI)'] = "nan (nan-nan)"
    
    # F1 and balanced accuracy
    metrics['F1'] = f1_score(y_true, y_pred)
    metrics['Balanced_Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    return metrics

def regenerate_summary(results_dir):
    results_path = Path(results_dir) / 'all_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    summary_data = []
    
    # Map the analysis types to their keys in the JSON
    analysis_mapping = {
        'independent_analysis': 'Independent',
        'holdout_analysis': 'Holdout',
        'combined_analysis': 'Combined'
    }
    
    for json_key, analysis_type in analysis_mapping.items():
        if json_key not in results:
            continue
            
        if json_key == 'independent_analysis':
            # Handle independent analysis
            for dataset, dataset_results in results[json_key].items():
                for dimension in dataset_results:
                    for model_name, model_results in dataset_results[dimension].items():
                        if not isinstance(model_results, dict) or 'predictions' not in model_results:
                            continue
                            
                        y_true = model_results['targets']
                        y_pred = model_results['predictions']
                        y_prob = model_results.get('probabilities', [])
                        
                        # For 2D models, aggregate to participant level
                        if '2d' in dimension.lower() and len(y_prob) > 0:
                            try:
                                # Group predictions by participant
                                participant_groups = group_by_participant(y_true, y_pred, y_prob)
                                
                                # Aggregate predictions for each participant
                                participant_preds = []
                                participant_probs = []
                                participant_targets = []
                                
                                for participant_id, group in participant_groups.items():
                                    labels, preds, probs = zip(*group)
                                    pred, prob = aggregate_2d_predictions_to_participant(preds, probs)
                                    participant_preds.append(pred)
                                    participant_probs.append(prob)
                                    participant_targets.append(labels[0])  # Take first label as participant label
                                
                                y_true = participant_targets
                                y_pred = participant_preds
                                y_prob = participant_probs
                            except Exception as e:
                                print(f"Warning: Failed to aggregate 2D predictions for {model_name} on {dataset}: {str(e)}")
                                continue
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_true, y_pred, y_prob)
                        
                        summary_data.append({
                            'Analysis Type': analysis_type,
                            'Dataset': dataset,
                            'Dimension': dimension,
                            'Model': model_name,
                            **metrics
                        })
                        
        elif json_key == 'holdout_analysis':
            # Handle holdout analysis
            for holdout_name, holdout_results in results[json_key].items():
                for model_name, model_results in holdout_results.items():
                    if not isinstance(model_results, dict) or 'predictions' not in model_results:
                        continue
                        
                    y_true = model_results['targets']
                    y_pred = model_results['predictions']
                    y_prob = model_results.get('probabilities', [])
                    
                    dimension = '2d' if any(x in model_name for x in ['2D', '2d']) else '3d'
                    
                    # For 2D models, aggregate to participant level
                    if dimension == '2d' and len(y_prob) > 0:
                        try:
                            # Group predictions by participant
                            participant_groups = group_by_participant(y_true, y_pred, y_prob)
                            
                            # Aggregate predictions for each participant
                            participant_preds = []
                            participant_probs = []
                            participant_targets = []
                            
                            for participant_id, group in participant_groups.items():
                                labels, preds, probs = zip(*group)
                                pred, prob = aggregate_2d_predictions_to_participant(preds, probs)
                                participant_preds.append(pred)
                                participant_probs.append(prob)
                                participant_targets.append(labels[0])  # Take first label as participant label
                            
                            y_true = participant_targets
                            y_pred = participant_preds
                            y_prob = participant_probs
                        except Exception as e:
                            print(f"Warning: Failed to aggregate 2D predictions for {model_name} on Test_{holdout_name}: {str(e)}")
                            continue
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_true, y_pred, y_prob)
                    
                    summary_data.append({
                        'Analysis Type': analysis_type,
                        'Dataset': f'Test_{holdout_name}',
                        'Dimension': dimension,
                        'Model': model_name,
                        **metrics
                    })
                    
        else:  # combined_analysis
            # Handle combined analysis
            for dimension, dim_results in results[json_key].items():
                for model_name, model_results in dim_results.items():
                    if not isinstance(model_results, dict) or 'predictions' not in model_results:
                        continue
                        
                    y_true = model_results['targets']
                    y_pred = model_results['predictions']
                    y_prob = model_results.get('probabilities', [])
                    
                    # For 2D models, aggregate to participant level
                    if '2d' in dimension.lower() and len(y_prob) > 0:
                        try:
                            # Group predictions by participant
                            participant_groups = group_by_participant(y_true, y_pred, y_prob)
                            
                            # Aggregate predictions for each participant
                            participant_preds = []
                            participant_probs = []
                            participant_targets = []
                            
                            for participant_id, group in participant_groups.items():
                                labels, preds, probs = zip(*group)
                                pred, prob = aggregate_2d_predictions_to_participant(preds, probs)
                                participant_preds.append(pred)
                                participant_probs.append(prob)
                                participant_targets.append(labels[0])  # Take first label as participant label
                            
                            y_true = participant_targets
                            y_pred = participant_preds
                            y_prob = participant_probs
                        except Exception as e:
                            print(f"Warning: Failed to aggregate 2D predictions for {model_name} on combined analysis: {str(e)}")
                            continue
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_true, y_pred, y_prob)
                    
                    summary_data.append({
                        'Analysis Type': analysis_type,
                        'Dataset': 'All',
                        'Dimension': dimension,
                        'Model': model_name,
                        **metrics
                    })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(summary_data)
    df = df.sort_values(['Analysis Type', 'Dataset', 'Dimension', 'Model'])
    output_path = Path(results_dir) / 'analysis_summary_participant_level.csv'
    df.to_csv(output_path)
    print(f"Summary saved to {output_path}")
    return df

if __name__ == "__main__":
    results_dir = "analysis_results_20250212_112607"
    summary_df = regenerate_summary(results_dir)
    print("\nSummary DataFrame Head:")
    print(summary_df.head()) 