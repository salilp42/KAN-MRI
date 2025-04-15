import torch
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from typing import Tuple
import SimpleITK as sitk
from scipy.spatial.distance import cdist

def build_2d_slic_graph(
    image_tensor: torch.Tensor,
    n_segments: int = 1000,
    compactness: float = 10.0,
    sigma: float = 1.0,
    k_neighbors: int = 6,
    connect_self: bool = False
) -> Data:
    """
    Builds a graph from a 2D image tensor using SLIC superpixels.

    Args:
        image_tensor: Input 2D PyTorch tensor (H, W) or (1, H, W).
                      Assumes single channel grayscale image.
        n_segments: Approximate number of superpixels.
        compactness: Balances color proximity and space proximity.
        sigma: Width of Gaussian smoothing kernel.
        k_neighbors: Number of neighbors for k-NN graph.
        connect_self: Whether nodes should connect to themselves in the graph.

    Returns:
        torch_geometric.data.Data: Graph data object containing node features (x)
                                     and edge connectivity (edge_index).
                                     Node features: [mean_intensity, rel_area, centroid_y, centroid_x]
    """
    # --- Input Validation and Preparation ---
    if image_tensor.ndim == 3:
        if image_tensor.shape[0] != 1:
            raise ValueError(f"Input tensor must be 2D (H, W) or single-channel (1, H, W), got shape {image_tensor.shape}")
        image_tensor = image_tensor.squeeze(0) # Remove channel dim
    elif image_tensor.ndim != 2:
        raise ValueError(f"Input tensor must be 2D (H, W) or single-channel (1, H, W), got shape {image_tensor.shape}")

    # Convert to NumPy float array for scikit-image
    image_np = image_tensor.cpu().numpy()
    image_np_float = img_as_float(image_np)
    height, width = image_np_float.shape
    total_pixels = height * width

    # --- SLIC Segmentation --- (Multichannel=False for grayscale)
    segments = slic(image_np_float, n_segments=n_segments, compactness=compactness,
                    sigma=sigma, start_label=1, channel_axis=None)

    # --- Feature Extraction (Region Properties) ---
    # Use intensity_image=image_np_float to calculate intensity stats on the original float image
    regions = regionprops(segments, intensity_image=image_np_float)

    node_features = []
    centroids = []
    valid_indices = [] # Store indices of regions that were successfully processed

    # regionprops uses 1-based labeling, adjust if needed, but slic(start_label=1) matches
    for i, props in enumerate(regions):
        # Ensure region is valid
        if props.area == 0:
            # print(f"Warning: Skipping region {props.label} with zero area.")
            continue

        # Features as per manuscript
        mean_intensity = props.mean_intensity
        relative_area = props.area / total_pixels
        centroid_y, centroid_x = props.centroid # Order: (row, col) or (y, x)

        node_features.append([mean_intensity, relative_area, centroid_y, centroid_x])
        centroids.append([centroid_y, centroid_x])
        valid_indices.append(i) # or props.label if mapping is needed

    if not node_features:
        # Handle cases where no valid superpixels were found (e.g., blank image)
        return Data(x=torch.empty((0, 4)), edge_index=torch.empty((2, 0), dtype=torch.long))

    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    centroids_np = np.array(centroids)

    # --- k-NN Graph Construction ---
    # Build graph based on centroid proximity
    # mode='connectivity' returns unweighted adjacency matrix
    adj_matrix = kneighbors_graph(
        centroids_np,
        n_neighbors=k_neighbors,
        mode='connectivity',
        include_self=connect_self,
        n_jobs=-1 # Use all available CPU cores
    )

    # Convert sparse adjacency matrix to edge_index format [2, num_edges]
    coo = adj_matrix.tocoo()
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)

    # --- Create Data Object ---
    graph_data = Data(
        x=node_features_tensor,          # Node features [num_nodes, num_features]
        edge_index=edge_index,          # Graph connectivity [2, num_edges]
        pos=torch.tensor(centroids_np, dtype=torch.float) # Optional: Store centroids [num_nodes, 2]
    )

    return graph_data

def build_3d_supervoxel_graph(
    volume_tensor: torch.Tensor,
    n_supervoxels: int = 1000,
    compactness: float = 10.0,
    k_neighbors: int = 6,
    connect_self: bool = False
) -> Data:
    """
    Builds a graph from a 3D volume tensor using SimpleITK's SLIC supervoxels.

    Args:
        volume_tensor: Input 3D PyTorch tensor (D, H, W) or (1, D, H, W).
                       Assumes single channel grayscale volume.
        n_supervoxels: Approximate number of supervoxels.
        compactness: Spatial proximity weight. Value range might differ from scikit-image.
                     Experimentation may be needed. Default in SITK is often lower.
        k_neighbors: Number of neighbors for k-NN graph.
        connect_self: Whether nodes should connect to themselves in the graph.

    Returns:
        torch_geometric.data.Data: Graph data object containing node features (x)
                                     and edge connectivity (edge_index).
                                     Node features: [mean_intensity, rel_volume, centroid_x, centroid_y, centroid_z]
    """
    # --- Input Validation and Preparation ---
    if volume_tensor.ndim == 4:
        if volume_tensor.shape[0] != 1:
            raise ValueError(f"Input tensor must be 3D (D, H, W) or single-channel (1, D, H, W), got shape {volume_tensor.shape}")
        volume_tensor = volume_tensor.squeeze(0) # Remove channel dim
    elif volume_tensor.ndim != 3:
        raise ValueError(f"Input tensor must be 3D (D, H, W) or single-channel (1, D, H, W), got shape {volume_tensor.shape}")

    # Convert PyTorch tensor to SimpleITK Image
    # Note: SimpleITK expects (x, y, z) order, PyTorch often (d, h, w)
    # We need to permute if tensor is (D, H, W)
    # Assuming input tensor is (D, H, W)
    volume_np = volume_tensor.cpu().numpy()
    volume_sitk = sitk.GetImageFromArray(volume_np) # Creates image with (z, y, x) index order
    # Ensure the image is treated as scalar float for intensity calculations
    volume_sitk_float = sitk.Cast(volume_sitk, sitk.sitkFloat32)
    total_voxels = np.prod(volume_np.shape)

    # --- SLIC Supervoxel Segmentation --- 
    slic_filter = sitk.SLICImageFilter()
    slic_filter.SetSuperGridSize([int(round(d / (n_supervoxels**(1/3)))) for d in volume_sitk.GetSize()]) # Estimate grid size
    slic_filter.SetSpatialProximityWeight(compactness)
    # slic_filter.SetNumberOfIterations(10) # Default is usually sufficient

    # Execute SLIC on the float image to potentially handle intensity better
    labels_sitk = slic_filter.Execute(volume_sitk_float)
    num_found_supervoxels = slic_filter.GetNumberOfLabels()

    if num_found_supervoxels == 0:
        print("Warning: SimpleITK SLIC found 0 supervoxels.")
        return Data(x=torch.empty((0, 5)), edge_index=torch.empty((2, 0), dtype=torch.long))

    # --- Feature Extraction (Label Statistics) ---
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    intensity_stats = sitk.LabelIntensityStatisticsImageFilter()

    shape_stats.Execute(labels_sitk)
    # Execute intensity stats on the original float volume using the labels
    intensity_stats.Execute(labels_sitk, volume_sitk_float)

    node_features = []
    centroids = []
    valid_labels = intensity_stats.GetLabels()

    for label in valid_labels:
        if label == 0: continue # Skip background label if present (depends on SLIC impl)

        # Features as per manuscript (using SITK outputs)
        mean_intensity = intensity_stats.GetMean(label)
        # GetPhysicalSize gives volume in physical units if spacing is set, else voxel count
        volume_voxels = shape_stats.GetNumberOfPixels(label)
        relative_volume = volume_voxels / total_voxels
        # GetCentroid returns physical coordinates (x, y, z)
        centroid_x, centroid_y, centroid_z = shape_stats.GetCentroid(label)

        node_features.append([mean_intensity, relative_volume, centroid_x, centroid_y, centroid_z])
        centroids.append([centroid_x, centroid_y, centroid_z])

    if not node_features:
        # Handle cases where only background label was found
        print("Warning: No valid supervoxel features extracted.")
        return Data(x=torch.empty((0, 5)), edge_index=torch.empty((2, 0), dtype=torch.long))

    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    centroids_np = np.array(centroids)

    # --- k-NN Graph Construction ---
    adj_matrix = kneighbors_graph(
        centroids_np,
        n_neighbors=k_neighbors,
        mode='connectivity',
        include_self=connect_self,
        n_jobs=-1
    )

    coo = adj_matrix.tocoo()
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)

    # --- Create Data Object ---
    graph_data = Data(
        x=node_features_tensor,       # Node features [num_nodes, 5]
        edge_index=edge_index,       # Graph connectivity [2, num_edges]
        pos=torch.tensor(centroids_np, dtype=torch.float) # Store centroids [num_nodes, 3]
    )

    return graph_data
