import os
import sys
import json
import torch
import argparse
from datetime import datetime
from typing import Dict, Any

# Add the repository root to the Python path
# This allows us to use absolute imports like 'from src.models...'
# Assumes the script is in KAN-MRI/scripts/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now imports relative to the project root should work
from src.datasets.data_loaders import (
    load_ppmi_data,
    load_taowu_data,
    load_neurocon_data,
    create_data_loaders
)
from src.models.resnet import ResNet2D, ResNet3D
from src.models.vgg import VGG2D, VGG3D
from src.models.convkan import ConvKAN2D, ConvKAN3D
from src.models.gnn import GNN2D, GNN3D
from src.training.trainer import Trainer

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_experiment_dir(base_dir: str) -> str:
    """Create and return experiment directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def get_model(model_name: str, config: Dict[str, Any]):
    """Get model instance based on name and config."""
    model_specific_config = config.get(model_name, {}) 
    general_config_keys = ['in_channels', 'num_classes', 'pretrained', 'use_pretrained_vgg']
    for key in general_config_keys:
         if key in config and key not in model_specific_config:
              model_specific_config[key] = config[key]

    if 'in_channels' not in model_specific_config:
         raise ValueError(f"'in_channels' not found in config for model {model_name}")
    if 'num_classes' not in model_specific_config:
         raise ValueError(f"'num_classes' not found in config for model {model_name}")


    models = {
        'resnet18_2d': ResNet2D,
        'resnet18_3d': ResNet3D,
        'vgg11_2d': VGG2D,
        'vgg11_3d': VGG3D,
        'convkan_2d': ConvKAN2D,
        'convkan_3d': ConvKAN3D,
        'gnn_2d': GNN2D,
        'gnn_3d': GNN3D
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name](model_specific_config)

def main(args):
    model_config_path = os.path.join(project_root, args.model_config)
    training_config_path = os.path.join(project_root, args.training_config)

    try:
        model_config = load_config(model_config_path)
        training_config = load_config(training_config_path)
    except FileNotFoundError as e:
        print(f"Error loading config files: {e}. Ensure '{args.model_config}' and '{args.training_config}' exist relative to the project root ({project_root}).")
        return

    output_dir_abs = args.output_dir
    if not os.path.isabs(output_dir_abs):
        output_dir_abs = os.path.join(project_root, output_dir_abs)
    os.makedirs(output_dir_abs, exist_ok=True) 

    experiment_dir = setup_experiment_dir(output_dir_abs) 
    print(f"Experiment directory: {experiment_dir}")

    with open(os.path.join(experiment_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(experiment_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=4)

    if args.debug:
        print("Running in debug mode")
        training_config['num_epochs'] = 2
        if 'batch_size' in training_config:
             training_config['batch_size'] = 2

    data_dir_abs = args.data_dir
    if not os.path.isabs(data_dir_abs):
        data_dir_abs = os.path.join(project_root, data_dir_abs)

    datasets = {}
    data_config = training_config.get('datasets', {}) 

    try:
        if 'ppmi' in data_config:
             print("Loading PPMI dataset...")
             ppmi_path = data_config['ppmi'].get('path', 'PPMI_Processed') 
             ppmi_abs_path = os.path.join(data_dir_abs, ppmi_path)
             datasets['ppmi'] = load_ppmi_data(ppmi_abs_path) 

        if 'taowu' in data_config:
             print("Loading Tao Wu dataset...")
             taowu_path = data_config['taowu'].get('path', 'TAOWU_Processed')
             taowu_abs_path = os.path.join(data_dir_abs, taowu_path)
             datasets['taowu'] = load_taowu_data(taowu_abs_path)

        if 'neurocon' in data_config:
             print("Loading NEUROCON dataset...")
             neurocon_path = data_config['neurocon'].get('path', 'NEUROCON_Processed')
             neurocon_abs_path = os.path.join(data_dir_abs, neurocon_path)
             datasets['neurocon'] = load_neurocon_data(neurocon_abs_path)

    except FileNotFoundError as e:
         print(f"Error loading datasets: {e}. Ensure data exists at expected paths relative to '{data_dir_abs}'.")
         return
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return

    if not datasets:
         print("No datasets were loaded. Check configuration and data paths.")
         return

    for dataset_name, dataset_info in datasets.items():
        if not isinstance(dataset_info, (tuple, list)) or len(dataset_info) < 2:
             print(f"Warning: Skipping dataset '{dataset_name}' due to unexpected format: {type(dataset_info)}")
             continue
        data, labels = dataset_info[0], dataset_info[1] 
        print(f"\nProcessing {dataset_name} dataset")

        batch_size = training_config.get('batch_size', model_config.get('batch_size', 32)) 

        train_loader, val_loader = create_data_loaders(
            data,
            labels,
            batch_size=batch_size,
            mode=args.mode 
        )

        general_model_params = {k: v for k, v in model_config.items() if k != 'models'}
        model_definitions = model_config.get('models', {}) 

        for model_name in model_definitions.keys(): 
            if args.mode in model_name:  
                print(f"\nTraining {model_name} on {dataset_name}")

                try:
                    model_specific_cfg = model_definitions.get(model_name, {}) 
                    combined_cfg = {**general_model_params, **model_specific_cfg}

                    model = get_model(model_name, combined_cfg)

                    model_experiment_dir = os.path.join(experiment_dir, dataset_name, model_name)
                    os.makedirs(model_experiment_dir, exist_ok=True)

                    trainer = Trainer(
                        model=model,
                        config=training_config, 
                        experiment_dir=model_experiment_dir
                    )

                    history = trainer.train(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        num_epochs=training_config['num_epochs']
                    )

                    print(f"Finished training {model_name} on {dataset_name}")

                except Exception as e:
                    print(f"Error training {model_name} on {dataset_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI classification models")
    parser.add_argument("--model_config", type=str, default="configs/model_config.json",
                        help="Path to model configuration file (relative to project root)")
    parser.add_argument("--training_config", type=str, default="configs/training_config.json",
                        help="Path to training configuration file (relative to project root)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory containing datasets (relative to project root)")
    parser.add_argument("--output_dir", type=str, default="results/training_output", 
                        help="Directory to save results (relative to project root)")
    parser.add_argument("--mode", type=str, choices=['2d', '3d'], default='2d',
                        help="Training mode (2D or 3D)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with minimal epochs and batch size")

    args = parser.parse_args()

    main(args)
