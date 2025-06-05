import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import timm
from .base import BaseModel

class ViT2D(BaseModel):
    """
    2D Vision Transformer model for medical image classification.
    
    Uses ViT-Tiny architecture (5.7M parameters) with ImageNet pretraining.
    Optimized for small medical datasets with patch size 16x16.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ViT configuration
        self.patch_size = config.get('patch_size', 16)
        self.embed_dim = config.get('embed_dim', 192)  # ViT-Tiny
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 3)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.drop_rate = config.get('drop_rate', 0.1)
        self.attn_drop_rate = config.get('attn_drop_rate', 0.1)
        
        # Create ViT model using timm
        model_name = 'vit_tiny_patch16_224'
        
        if self.pretrained:
            # Load with ImageNet pretrained weights
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                in_chans=self.in_channels,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate
            )
        else:
            # Create without pretrained weights
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,  # Remove classification head
                in_chans=self.in_channels,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate
            )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head with medical imaging optimizations
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize classification head
        self._init_classifier()
        
        # Add input normalization for medical images
        self.input_norm = nn.BatchNorm2d(self.in_channels)
        
    def _init_classifier(self):
        """Initialize the classification head with proper scaling."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization with small gain for stability
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT2D.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Input normalization for medical images
        x = self.input_norm(x)
        
        # Ensure input size is compatible with patch size
        B, C, H, W = x.shape
        
        # Resize if needed to be divisible by patch size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            new_H = ((H // self.patch_size) + 1) * self.patch_size
            new_W = ((W // self.patch_size) + 1) * self.patch_size
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        # Forward through ViT backbone
        features = self.backbone(x)  # Shape: (B, feature_dim)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from the backbone."""
        x = self.input_norm(x)
        
        # Handle input size
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            new_H = ((H // self.patch_size) + 1) * self.patch_size
            new_W = ((W // self.patch_size) + 1) * self.patch_size
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        return self.backbone(x)


class ViT3D(BaseModel):
    """
    3D Vision Transformer model using hybrid 2D ViT + 3D attention approach.
    
    Processes 3D volumes by:
    1. Applying 2D ViT to each slice
    2. Using cross-slice attention to capture 3D context
    3. Aggregating features for final classification
    
    This approach avoids the computational complexity of full 3D transformers
    while maintaining the attention benefits for medical imaging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 3D-specific configuration
        self.slice_dim = config.get('slice_dim', -1)  # Which dimension represents slices
        self.max_slices = config.get('max_slices', 32)  # Maximum number of slices to process
        self.cross_slice_heads = config.get('cross_slice_heads', 4)
        
        # Create 2D ViT backbone for slice processing
        model_name = 'vit_tiny_patch16_224'
        
        if self.pretrained:
            self.slice_processor = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                in_chans=self.in_channels,
                drop_rate=0.1,
                attn_drop_rate=0.1
            )
        else:
            self.slice_processor = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,  # Remove classification head
                in_chans=self.in_channels,
                drop_rate=0.1,
                attn_drop_rate=0.1
            )
        
        # Get feature dimension
        self.feature_dim = self.slice_processor.num_features
        
        # Cross-slice attention mechanism
        self.cross_slice_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.cross_slice_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Positional encoding for slice positions
        self.slice_pos_encoding = nn.Parameter(
            torch.randn(1, self.max_slices, self.feature_dim) * 0.02
        )
        
        # Layer normalization
        self.ln_cross_slice = nn.LayerNorm(self.feature_dim)
        
        # Aggregation method
        self.aggregation_method = config.get('aggregation', 'attention')  # 'mean', 'max', 'attention'
        
        if self.aggregation_method == 'attention':
            # Learnable attention weights for slice aggregation
            self.slice_attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(),
                nn.Linear(self.feature_dim // 4, 1)
            )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Input normalization
        self.input_norm = nn.BatchNorm3d(self.in_channels)
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize cross-slice attention
        for m in [self.cross_slice_attention]:
            if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if hasattr(m, 'out_proj') and m.out_proj.weight is not None:
                nn.init.xavier_uniform_(m.out_proj.weight)
        
        # Initialize aggregation attention if used
        if hasattr(self, 'slice_attention'):
            for m in self.slice_attention.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def process_slices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process 3D volume slice by slice using 2D ViT.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W) or (B, C, H, W, D)
            
        Returns:
            Slice features of shape (B, num_slices, feature_dim)
        """
        B, C = x.shape[:2]
        
        # Handle different slice dimensions
        if len(x.shape) == 5:  # (B, C, D, H, W) or (B, C, H, W, D)
            if self.slice_dim == -1 or self.slice_dim == 4:
                # Slices are in the last dimension
                x = x.permute(0, 1, 4, 2, 3)  # (B, C, D, H, W)
            # Now x is (B, C, D, H, W)
            D, H, W = x.shape[2:]
        else:
            raise ValueError(f"Expected 5D input, got {x.shape}")
        
        # Limit number of slices if necessary
        if D > self.max_slices:
            # Sample slices uniformly
            indices = torch.linspace(0, D-1, self.max_slices, dtype=torch.long)
            x = x[:, :, indices]
            D = self.max_slices
        
        # Process each slice
        slice_features = []
        for i in range(D):
            slice_2d = x[:, :, i]  # (B, C, H, W)
            
            # Always resize to 224x224 for ViT compatibility
            # ViT models expect specific input sizes
            slice_2d = F.interpolate(slice_2d, size=(224, 224), 
                                   mode='bilinear', align_corners=False)
            
            # Process through 2D ViT
            features = self.slice_processor(slice_2d)  # (B, feature_dim)
            slice_features.append(features)
        
        # Stack slice features
        slice_features = torch.stack(slice_features, dim=1)  # (B, D, feature_dim)
        
        return slice_features
    
    def apply_cross_slice_attention(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-slice attention to capture 3D context.
        
        Args:
            slice_features: Tensor of shape (B, num_slices, feature_dim)
            
        Returns:
            Enhanced features of shape (B, num_slices, feature_dim)
        """
        B, num_slices, feature_dim = slice_features.shape
        
        # Add positional encoding
        pos_encoding = self.slice_pos_encoding[:, :num_slices, :]
        slice_features = slice_features + pos_encoding
        
        # Apply cross-slice attention
        attended_features, _ = self.cross_slice_attention(
            slice_features, slice_features, slice_features
        )
        
        # Residual connection and layer norm
        slice_features = self.ln_cross_slice(slice_features + attended_features)
        
        return slice_features
    
    def aggregate_slice_features(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate slice features into a single volume representation.
        
        Args:
            slice_features: Tensor of shape (B, num_slices, feature_dim)
            
        Returns:
            Aggregated features of shape (B, feature_dim)
        """
        if self.aggregation_method == 'mean':
            return slice_features.mean(dim=1)
        
        elif self.aggregation_method == 'max':
            return slice_features.max(dim=1)[0]
        
        elif self.aggregation_method == 'attention':
            # Compute attention weights for each slice
            attention_weights = self.slice_attention(slice_features)  # (B, num_slices, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Weighted aggregation
            aggregated = (slice_features * attention_weights).sum(dim=1)  # (B, feature_dim)
            return aggregated
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT3D.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Input normalization
        x = self.input_norm(x)
        
        # Process slices with 2D ViT
        slice_features = self.process_slices(x)  # (B, num_slices, feature_dim)
        
        # Apply cross-slice attention
        enhanced_features = self.apply_cross_slice_attention(slice_features)
        
        # Aggregate slice features
        volume_features = self.aggregate_slice_features(enhanced_features)
        
        # Final classification
        logits = self.classifier(volume_features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from the model."""
        x = self.input_norm(x)
        slice_features = self.process_slices(x)
        enhanced_features = self.apply_cross_slice_attention(slice_features)
        return self.aggregate_slice_features(enhanced_features) 
