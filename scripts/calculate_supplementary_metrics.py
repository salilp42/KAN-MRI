import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
import os
import sys
import argparse

# Add the repository root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration (Now handled by argparse) ---
OUTPUT_FILENAME = 'supplementary_metrics.csv' # Fixed output filename

# --- Helper function to get data (adapted from refined_paper_figures.py) ---
def get_data_for_metrics(results_data, analysis_type, dataset, dimension, model):
    """
    Helper function to extract true labels and predicted probabilities.
    Handles the different structures in the JSON file:
    1. Independent analysis: independent_analysis -> dataset -> dimension -> model -> data
    2. Holdout analysis: holdout_analysis -> dataset -> model_dimension as single key -> data
    3. Combined analysis: combined_analysis -> dimension -> model -> data
    """
    try:
        if analysis_type == 'independent_analysis':
            data = results_data['independent_analysis'][dataset][dimension][model]
        elif analysis_type == 'combined_analysis':
            # Combined analysis doesn't have a 'dataset' key at this level
            data = results_data['combined_analysis'][dimension][model]
            dataset = 'Combined' # Use 'Combined' placeholder
        elif analysis_type == 'holdout_analysis':
            # Holdout uses a combined key like 'ResNet2D' or 'ConvKAN3D'
            # Need to construct this key if model and dimension are separate
            model_dim_key = f"{model}{dimension.upper()}"
            data = results_data['holdout_analysis'][dataset][model_dim_key]
        else:
            print(f"Unknown analysis type: {analysis_type}")
            return None, None, None # Return None for dataset as well

        y_true_raw = data.get('targets')
        y_prob_raw = data.get('probabilities')

        if y_true_raw is not None and y_prob_raw is not None:
            y_true = np.array(y_true_raw)
            # Assuming probabilities are [prob_class_0, prob_class_1]
            y_prob = np.array([prob[1] for prob in y_prob_raw])
            return y_true, y_prob, dataset
        else:
            # print(f"Missing targets or probabilities for {analysis_type}/{dataset}/{dimension}/{model}")
            return None, None, dataset

    except (KeyError, IndexError, TypeError) as e:
        # print(f"Error extracting data for {analysis_type}/{dataset}/{dimension}/{model}: {str(e)}")
        return None, None, dataset

# --- Main Calculation Logic ---
def calculate_metrics(results_data):
    metrics_list = []

    # Define structure based on observed keys and get_roc_data docstring
    analysis_types = ['independent_analysis', 'combined_analysis', 'holdout_analysis']

    for analysis_type in analysis_types:
        if analysis_type == 'independent_analysis':
            datasets = results_data.get('independent_analysis', {}).keys()
            for dataset in datasets:
                dimensions = results_data['independent_analysis'][dataset].keys()
                for dimension in dimensions:
                    models = results_data['independent_analysis'][dataset][dimension].keys()
                    for model in models:
                        process_model(results_data, analysis_type, dataset, dimension, model, metrics_list)

        elif analysis_type == 'combined_analysis':
            # Combined analysis: no dataset key, iterate dimensions then models
            dimensions = results_data.get('combined_analysis', {}).keys()
            for dimension in dimensions:
                models = results_data['combined_analysis'][dimension].keys()
                for model in models:
                    process_model(results_data, analysis_type, 'Combined', dimension, model, metrics_list)

        elif analysis_type == 'holdout_analysis':
            datasets = results_data.get('holdout_analysis', {}).keys() # Test datasets
            for dataset in datasets:
                model_dims = results_data['holdout_analysis'][dataset].keys()
                for model_dim_key in model_dims:
                    # Attempt to parse model and dimension from key (e.g., 'ResNet2D')
                    dimension = '2D' if '2D' in model_dim_key else '3D' if '3D' in model_dim_key else None
                    model = model_dim_key.replace('2D', '').replace('3D', '') if dimension else model_dim_key
                    if dimension:
                         process_model(results_data, analysis_type, dataset, dimension, model, metrics_list)
                    else:
                        print(f"Could not parse dimension from holdout key: {model_dim_key} for dataset {dataset}")

    return pd.DataFrame(metrics_list)

def process_model(results_data, analysis_type, dataset, dimension, model, metrics_list):
    y_true, y_prob, actual_dataset = get_data_for_metrics(results_data, analysis_type, dataset, dimension, model)

    if y_true is not None and y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]

            # Handle edge case where threshold might be infinite if only one class predicted
            if np.isinf(optimal_threshold):
                 # Fallback: find threshold closest to standard 0.5 if J is max at ends
                 finite_thresholds = thresholds[np.isfinite(thresholds)]
                 if len(finite_thresholds) > 0:
                    optimal_idx = np.argmin(np.abs(finite_thresholds - 0.5))
                    optimal_threshold = finite_thresholds[optimal_idx]
                 else: # If no finite thresholds (highly unlikely with probabilities), use 0.5
                    optimal_threshold = 0.5

            # Get predictions based on optimal threshold
            y_pred = (y_prob >= optimal_threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics_list.append({
                'AnalysisType': analysis_type,
                'Dataset': actual_dataset,
                'Dimension': dimension,
                'Model': model,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'OptimalThreshold': optimal_threshold
            })
        except Exception as e:
            print(f"Error calculating metrics for {analysis_type}/{actual_dataset}/{dimension}/{model}: {str(e)}")
    # else:
        # print(f"Skipping metrics calculation for {analysis_type}/{actual_dataset}/{dimension}/{model} due to missing data or single class.")

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate supplementary metrics (Sensitivity, Specificity, Threshold) from analysis results JSON.')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to the input JSON file containing analysis results (e.g., results/experiment_XYZ/all_results.json)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save the output CSV file (relative to project root, defaults to results/)')

    args = parser.parse_args()

    # Construct absolute paths
    results_file_path = args.results_file
    if not os.path.isabs(results_file_path):
        results_file_path = os.path.join(project_root, results_file_path)

    output_dir_path = args.output_dir
    if not os.path.isabs(output_dir_path):
        output_dir_path = os.path.join(project_root, output_dir_path)

    # Ensure output directory exists
    os.makedirs(output_dir_path, exist_ok=True)

    output_csv_path = os.path.join(output_dir_path, OUTPUT_FILENAME)

    print(f"Loading results from: {results_file_path}")
    try:
        with open(results_file_path, 'r') as f:
            all_results = json.load(f)
        print("Results loaded successfully.")

        print("Calculating supplementary metrics...")
        metrics_df = calculate_metrics(all_results)
        print("Metrics calculation complete.")

        metrics_df.to_csv(output_csv_path, index=False)
        print(f"Supplementary metrics saved to: {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {results_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
