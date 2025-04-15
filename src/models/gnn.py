import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch # Keep Batch
import numpy as np
from typing import Dict, Any

from .base import BaseModel

class GNN2D(BaseModel):
    """2D Graph Neural Network (Superpixel-based as per manuscript)."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Node features based on manuscript: mean intensity, rel area, centroid (x,y)
        self.num_node_features = config.get('num_node_features', 4)
        if self.num_node_features != 4:
            print(f"Warning: Expected 4 node features for GNN2D based on manuscript, got {self.num_node_features}")

        # Layer dimensions from manuscript
        channels = [64, 128, 256]
        fc_hidden = 512

        # Graph convolution layers
        self.conv1 = GCNConv(self.num_node_features, channels[0])
        self.conv2 = GCNConv(channels[0], channels[1])
        self.conv3 = GCNConv(channels[1], channels[2])

        self.bn1 = nn.BatchNorm1d(channels[0])
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.bn3 = nn.BatchNorm1d(channels[2])

        self.dropout = nn.Dropout(config.get('dropout_rate', 0.5))

        # Fully connected layers from manuscript
        self.fc1 = nn.Linear(channels[2], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, self.num_classes)

    # _tensor_to_graph removed - graph creation must happen upstream

    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass expects a Batch object from torch_geometric."""
        # Extract graph components from Batch object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # Global pooling and classification layers
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class GNN3D(BaseModel):
    """3D Graph Neural Network (Supervoxel-based as per manuscript)."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Node features based on manuscript: mean intensity, rel volume, centroid (x,y,z)
        self.num_node_features = config.get('num_node_features', 5)
        if self.num_node_features != 5:
             print(f"Warning: Expected 5 node features for GNN3D based on manuscript, got {self.num_node_features}")

        # Layer dimensions from manuscript
        channels = [64, 128, 256]
        fc_hidden = 512

        # Graph convolution layers
        self.conv1 = GCNConv(self.num_node_features, channels[0])
        self.conv2 = GCNConv(channels[0], channels[1])
        self.conv3 = GCNConv(channels[1], channels[2])

        self.bn1 = nn.BatchNorm1d(channels[0])
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.bn3 = nn.BatchNorm1d(channels[2])

        self.dropout = nn.Dropout(config.get('dropout_rate', 0.5))

        # Fully connected layers from manuscript
        self.fc1 = nn.Linear(channels[2], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, self.num_classes)

    # _tensor_to_graph removed - graph creation must happen upstream

    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass expects a Batch object from torch_geometric."""
        # Extract graph components from Batch object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # Global pooling and classification layers
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
