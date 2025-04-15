import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from .preprocessing import resize_volume, normalize_volume, extract_2d_slices
from ..graph.graph_builder import build_2d_slic_graph, build_3d_supervoxel_graph
from torch_geometric.data import Batch

class MRIDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray, 
                 labels: np.ndarray, 
                 mode: str = '2d',
                 transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.mode = mode
        self.transform = transform
        
        if mode == '2d':
            if len(self.data.shape) == 4:  # If data is [N, H, W, S]
                self.data = self.data.permute(0, 3, 1, 2)  # Change to [N, S, H, W]
            self.data = self.data.unsqueeze(1)  # Add channel dim [N, 1, S, H, W]
        else:  # 3d
            self.data = self.data.unsqueeze(1)  # Add channel dim [N, 1, H, W, D]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

class MRISliceGraphDataset(Dataset):
    """Dataset for handling individual 2D slices and converting them to graphs."""
    def __init__(self, slices_data: torch.Tensor, slice_labels: torch.Tensor, slice_subject_ids: List[str]):
        """
        Args:
            slices_data: Tensor containing all slices [num_total_slices, H, W].
            slice_labels: Tensor containing labels for each slice [num_total_slices].
            slice_subject_ids: List containing subject ID for each slice.
        """
        if slices_data.ndim != 3:
            raise ValueError(f"Expected slices_data to be 3D (S, H, W), got {slices_data.shape}")
        if len(slices_data) != len(slice_labels) or len(slices_data) != len(slice_subject_ids):
            raise ValueError("Mismatch in number of slices, labels, or subject IDs.")

        self.slices_data = slices_data
        self.slice_labels = slice_labels
        self.slice_subject_ids = slice_subject_ids

    def __len__(self):
        return len(self.slices_data)

    def __getitem__(self, idx):
        slice_tensor = self.slices_data[idx] # Shape (H, W)
        label = self.slice_labels[idx]
        # subject_id = self.slice_subject_ids[idx] # Available if needed

        # Build graph on-the-fly
        graph_data = build_2d_slic_graph(
            slice_tensor,
            n_segments=1000, 
            compactness=10.0,
            sigma=1.0,
            k_neighbors=6, 
        )

        # Return the graph Data object and the label
        return graph_data, label

class MRIVolumeGraphDataset(Dataset):
    """Dataset for handling 3D volumes and converting them to graphs."""
    def __init__(self, volume_data: torch.Tensor, volume_labels: torch.Tensor):
        """
        Args:
            volume_data: Tensor containing all volumes [N, D, H, W] or [N, 1, D, H, W].
            volume_labels: Tensor containing labels for each volume [N].
        """
        if volume_data.ndim == 5 and volume_data.shape[1] == 1:
             self.volume_data = volume_data.squeeze(1) # Remove channel dim -> [N, D, H, W]
        elif volume_data.ndim == 4:
             self.volume_data = volume_data # Assume [N, D, H, W]
        else:
             raise ValueError(f"Expected volume_data to be 4D (N, D, H, W) or 5D (N, 1, D, H, W), got {volume_data.shape}")

        if len(self.volume_data) != len(volume_labels):
            raise ValueError("Mismatch in number of volumes and labels.")

        self.volume_labels = volume_labels

    def __len__(self):
        return len(self.volume_data)

    def __getitem__(self, idx):
        volume_tensor = self.volume_data[idx] # Shape (D, H, W)
        label = self.volume_labels[idx]

        # Build graph on-the-fly
        # TODO: Make n_supervoxels, compactness configurable?
        graph_data = build_3d_supervoxel_graph(
            volume_tensor,
            n_supervoxels=1000, # From manuscript
            compactness=10.0, # May need tuning for SimpleITK
            k_neighbors=6, # From manuscript
        )

        # Return the graph Data object and the label
        return graph_data, label

def load_ppmi_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load PPMI dataset."""
    try:
        info_df = pd.read_csv(config['info_csv'])
        mri_data = []
        labels = []
        identifiers = []

        for _, row in tqdm(info_df.iterrows(), desc="Loading PPMI data"):
            subject_folder = row['ID_and_Label']
            npy_file = os.path.join(config['data_dir'], subject_folder, 'processed_scan.npy')
            
            if os.path.exists(npy_file):
                try:
                    scan = np.load(npy_file)
                    scan = normalize_volume(scan)
                    mri_data.append(scan)
                    labels.append(1 if row['Group'] == 'PD' else 0)
                    identifiers.append(subject_folder)
                except Exception as e:
                    print(f"Error processing {npy_file}: {str(e)}")
                    continue
            else:
                print(f"Warning: File not found - {npy_file}")

        if not mri_data:
            raise ValueError("No PPMI data could be loaded")

        return np.array(mri_data), np.array(labels), identifiers
    except Exception as e:
        raise Exception(f"Error loading PPMI dataset: {str(e)}")

def load_taowu_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Tao Wu dataset."""
    try:
        mri_data = []
        labels = []
        identifiers = []
        
        main_folder = config['data_dir']
        subfolders = [f for f in os.listdir(main_folder) 
                     if os.path.isdir(os.path.join(main_folder, f)) 
                     and (f.startswith('patient') or f.startswith('control'))]

        for subfolder in tqdm(subfolders, desc="Loading Tao Wu data"):
            scan_path = os.path.join(main_folder, subfolder, 'processed_scan.npy')
            if os.path.exists(scan_path):
                try:
                    scan = np.load(scan_path)
                    scan = normalize_volume(scan)
                    mri_data.append(scan)
                    labels.append(1 if 'patient' in subfolder else 0)
                    identifiers.append(subfolder)
                except Exception as e:
                    print(f"Error processing {scan_path}: {str(e)}")
                    continue
            else:
                print(f"Warning: File not found - {scan_path}")
        
        if not mri_data:
            raise ValueError("No Tao Wu data could be loaded")
        
        return np.array(mri_data), np.array(labels), identifiers
    except Exception as e:
        raise Exception(f"Error loading Tao Wu dataset: {str(e)}")

def load_neurocon_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load NEUROCON dataset."""
    try:
        mri_data = []
        labels = []
        identifiers = []
        
        main_folder = config['data_dir']
        subfolders = [f for f in os.listdir(main_folder) 
                     if os.path.isdir(os.path.join(main_folder, f))
                     and (f.startswith('patient') or f.startswith('control'))]
        
        for subfolder in tqdm(subfolders, desc="Loading NEUROCON data"):
            scan_path = os.path.join(main_folder, subfolder, 'processed_scan.npy')
            if os.path.exists(scan_path):
                try:
                    scan = np.load(scan_path)
                    scan = normalize_volume(scan)
                    mri_data.append(scan)
                    labels.append(1 if 'patient' in subfolder else 0)
                    identifiers.append(subfolder)
                except Exception as e:
                    print(f"Error processing {scan_path}: {str(e)}")
                    continue
            else:
                print(f"Warning: File not found - {scan_path}")
        
        if not mri_data:
            raise ValueError("No NEUROCON data could be loaded")
        
        return np.array(mri_data), np.array(labels), identifiers
    except Exception as e:
        raise Exception(f"Error loading NEUROCON dataset: {str(e)}")

def create_data_loaders(
    data: np.ndarray,
    labels: np.ndarray,
    subject_identifiers: List[str],
    config: Dict[str, Any],
    mode: str = '2d',
    model_type: str = 'cnn'
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation.

    Args:
        data: Input data numpy array. Shape depends on mode.
              For '2d' mode input expected by extract_2d_slices: (N, H, W, D) or (N, D, H, W).
              For '3d' mode: (N, D, H, W).
        labels: Labels (N,).
        subject_identifiers: List of subject IDs corresponding to data (N,).
        config: Dictionary containing parameters like batch_size, num_workers.
        mode: '2d' or '3d'.
        model_type: Type of model ('cnn', 'gcn', etc.) to determine dataset/loader setup.

    Returns:
        Tuple containing train_loader and val_loader.
    """
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 0)
    validation_split = config.get('validation_split', 0.2)
    seed = config.get('seed', 42)
    generator = torch.Generator().manual_seed(seed)

    if mode.lower() == '2d' and model_type.lower() == 'gcn':
        print("Configuring DataLoader for 2D GCN (Slice Graphs)...")
        if data.ndim == 4 and data.shape[1] > data.shape[-1]: 
             data_for_extract = data.transpose(0, 2, 3, 1) 
        elif data.ndim == 4:
             data_for_extract = data 
        else:
             raise ValueError(f"Unsupported data shape for 2D slice extraction: {data.shape}")

        all_slices, slice_labels, slice_subject_ids = extract_2d_slices(
            data_for_extract, labels, subject_identifiers
        )

        all_slices_tensor = torch.FloatTensor(all_slices)
        slice_labels_tensor = torch.LongTensor(slice_labels)

        dataset = MRISliceGraphDataset(all_slices_tensor, slice_labels_tensor, slice_subject_ids)

        num_total = len(dataset)
        num_val = int(validation_split * num_total)
        num_train = num_total - num_val
        train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=Batch.from_data_list 
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=Batch.from_data_list 
        )
        print(f"Created 2D GCN DataLoaders. Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples.")

    elif mode.lower() == '3d' and model_type.lower() == 'gcn':
        print("Configuring DataLoader for 3D GCN (Volume Graphs)...")
        volume_data_tensor = torch.FloatTensor(data) # Assumes data is (N, D, H, W)
        volume_labels_tensor = torch.LongTensor(labels)

        dataset = MRIVolumeGraphDataset(volume_data_tensor, volume_labels_tensor)

        num_total = len(dataset)
        num_val = int(validation_split * num_total)
        num_train = num_total - num_val
        train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=Batch.from_data_list # Use PyG Batch collation
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=Batch.from_data_list # Use PyG Batch collation
        )
        print(f"Created 3D GCN DataLoaders. Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples.")

    else:
        print(f"Configuring DataLoader for {mode.upper()} {model_type.upper()} (TensorDataset)...")
        data_tensor = torch.FloatTensor(data)
        labels_tensor = torch.LongTensor(labels)

        if data_tensor.ndim == 4: # (N, D, H, W) or (N, S, H, W)
            data_tensor = data_tensor.unsqueeze(1) # -> (N, 1, D, H, W) or (N, 1, S, H, W)
        elif data_tensor.ndim == 3: # (N, H, W) - less likely for raw loaded data?
            data_tensor = data_tensor.unsqueeze(1) # -> (N, 1, H, W)

        dataset = TensorDataset(data_tensor, labels_tensor)

        num_total = len(dataset)
        num_val = int(validation_split * num_total)
        num_train = num_total - num_val
        train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"Created {mode.upper()} {model_type.upper()} DataLoaders. Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples.")

    return train_loader, val_loader
