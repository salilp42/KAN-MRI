import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseModel

# Helper function (same as in vgg.py)
def adapt_conv_weights(conv_layer, new_in_channels, pretrained_weights):
    """Adapts pretrained weights for a conv layer with different input channels."""
    original_weights = pretrained_weights
    original_in_channels = original_weights.shape[1]

    if new_in_channels == original_in_channels:
        # If pretrained and channels match, still need to assign weights
        conv_layer.weight = nn.Parameter(original_weights)
        return

    # Average weights across the original input channels
    new_weights = original_weights.mean(dim=1, keepdim=True)
    # Repeat for new input channels
    new_weights = new_weights.repeat(1, new_in_channels, 1, 1)
    new_weights.requires_grad_(original_weights.requires_grad)

    conv_layer.weight = nn.Parameter(new_weights)
    conv_layer.in_channels = new_in_channels

class ResNet2D(BaseModel):
    """2D ResNet18 model."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Load pretrained model
        weights = ResNet18_Weights.DEFAULT if self.pretrained else None
        self.model = resnet18(weights=weights)

        # Store original conv1 weights if pretrained
        original_conv1_weights = None
        if weights:
             original_conv1_weights = self.model.conv1.weight.data.clone()

        # Modify first conv layer structure for single channel and smaller settings
        new_conv1 = nn.Conv2d(
            self.in_channels,
            64,
            kernel_size=3,  # Smaller kernel
            stride=1,       # No stride
            padding=1,      # Same padding
            bias=False
        )
        self.model.conv1 = new_conv1 # Assign the new layer structure

        # Adapt weights if pretrained
        if original_conv1_weights is not None:
             adapt_conv_weights(self.model.conv1, self.in_channels, original_conv1_weights)

        # Reduce stride in first maxpool
        self.model.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # Modify layer strides to handle smaller input - adjusted for ResNet18 BasicBlock
        for layer in [self.model.layer2, self.model.layer3, self.model.layer4]:
            if layer[0].downsample is not None: # Check if downsample exists
                 layer[0].downsample[0].stride = (1, 1)
            layer[0].conv1.stride = (1, 1)
            # BasicBlock in ResNet18 doesn't have stride on conv2

        # Add adaptive pooling before FC layer
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Modify final fully connected layer with proper initialization
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        # Initialize the *new* FC layers only
        # The rest uses pretrained weights (except maybe adapted conv1 bias if it existed)
        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add batch normalization for input standardization
        if not hasattr(self, 'input_bn'):
            self.input_bn = nn.BatchNorm2d(self.in_channels).to(x.device)
        x = self.input_bn(x)
        return self.model(x)

class ResNet3D(BaseModel):
    """3D ResNet model using pre-trained weights from torch.hub."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Load pre-trained 3D ResNet from torch.hub
        if self.pretrained:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        else:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        
        # Modify first conv layer for single channel if needed
        if self.in_channels != 3:
            old_conv = self.model.blocks[0].conv
            new_conv = nn.Conv3d(
                self.in_channels, 
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Initialize from the mean of RGB channels if pretrained
            if self.pretrained:
                with torch.no_grad():
                    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            self.model.blocks[0].conv = new_conv
        
        # Add adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Get feature size from the last conv layer
        last_conv = None
        for block in reversed(self.model.blocks):
            for layer in reversed(block.children()):
                if isinstance(layer, nn.Conv3d):
                    last_conv = layer
                    break
            if last_conv:
                break
        
        if last_conv is None:
            raise ValueError("Could not find last conv layer")
            
        feature_size = last_conv.out_channels
        
        # Replace the classifier with a better initialization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        # Initialize classifier weights with better scaling
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier/Glorot initialization with small gain for better initial predictions
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add batch normalization for input standardization
        if not hasattr(self, 'input_bn'):
            self.input_bn = nn.BatchNorm3d(self.in_channels).to(x.device)
        x = self.input_bn(x)
        
        # Forward through main layers
        for block in self.model.blocks[:-1]:  # Exclude last block which had the old classifier
            x = block(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
