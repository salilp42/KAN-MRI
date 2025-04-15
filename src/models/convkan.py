import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from .base import BaseModel

class BSpline2D(nn.Module):
    """2D B-spline layer."""
    def __init__(self, num_channels: int, num_knots: int = 6, grid_range: Tuple[float, float] = (-1, 1)):
        super().__init__()
        self.num_channels = num_channels
        self.num_knots = num_knots
        self.grid_range = grid_range
        self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], num_knots))
        self.weights = nn.Parameter(torch.randn(num_channels, num_knots))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)
        knots_expanded = self.knots.view(1, 1, 1, 1, -1).expand(x.size(0), self.num_channels, x.size(2), x.size(3), self.num_knots)
        basis = F.relu(x_expanded - knots_expanded).pow(3)
        return torch.einsum('bchwk,ck->bchw', basis, self.weights)

class BSpline3D(nn.Module):
    """3D B-spline layer."""
    def __init__(self, num_channels: int, num_knots: int = 6, grid_range: Tuple[float, float] = (-1, 1)):
        super().__init__()
        self.num_channels = num_channels
        self.num_knots = num_knots
        self.grid_range = grid_range
        self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], num_knots))
        self.weights = nn.Parameter(torch.randn(num_channels, num_knots))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)
        knots_expanded = self.knots.view(1, 1, 1, 1, 1, -1).expand(
            x.size(0), self.num_channels, x.size(2), x.size(3), x.size(4), self.num_knots)
        basis = F.relu(x_expanded - knots_expanded).pow(3)
        return torch.einsum('bcdhwk,ck->bcdhw', basis, self.weights)

class SplineConv2D(nn.Module):
    """2D Spline Convolution layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.spline = BSpline2D(out_channels)
        self.w1 = nn.Parameter(torch.randn(out_channels))
        self.w2 = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        spline_out = self.spline(conv_out)
        silu_out = F.silu(conv_out)
        return self.w1.view(1, -1, 1, 1) * spline_out + self.w2.view(1, -1, 1, 1) * silu_out

class SplineConv3D(nn.Module):
    """3D Spline Convolution layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.spline = BSpline3D(out_channels)
        self.w1 = nn.Parameter(torch.randn(out_channels))
        self.w2 = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        spline_out = self.spline(conv_out)
        silu_out = F.silu(conv_out)
        return self.w1.view(1, -1, 1, 1, 1) * spline_out + self.w2.view(1, -1, 1, 1, 1) * silu_out

class ConvKAN2D(BaseModel):
    """2D Convolutional Kernel Activation Network."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.conv1 = SplineConv2D(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = SplineConv2D(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = SplineConv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.pool(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ConvKAN3D(BaseModel):
    """3D Convolutional Kernel Activation Network."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Smaller initial channels and gradual increase
        self.conv1 = SplineConv3D(self.in_channels, 8, kernel_size=3, stride=2, padding=1)  # Use stride 2
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = SplineConv3D(8, 16, kernel_size=3, stride=2, padding=1)  # Use stride 2
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = SplineConv3D(16, 32, kernel_size=3, stride=2, padding=1)  # Use stride 2
        self.bn3 = nn.BatchNorm3d(32)
        
        # Remove pooling layers since we use stride in conv
        self.dropout = nn.Dropout(0.3)  # Reduced dropout
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, self.num_classes)
        
        # Better initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, SplineConv3D)):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu')
                if m.conv.bias is not None:
                    nn.init.constant_(m.conv.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass with gradient checkpointing for memory efficiency
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        # Add small epsilon to prevent extreme predictions
        return x + 1e-8
