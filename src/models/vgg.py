import torch
import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights
import torch.nn.functional as F
from typing import Dict, Any

# Helper function to adapt the first conv layer's weights
def adapt_conv_weights(conv_layer, new_in_channels, pretrained_weights):
    """Adapts pretrained weights for a conv layer with different input channels."""
    original_weights = pretrained_weights
    original_in_channels = original_weights.shape[1]

    if new_in_channels == original_in_channels:
        return original_weights # No change needed

    # Average weights across the original input channels
    # Shape: (out_channels, in_channels, kH, kW)
    new_weights = original_weights.mean(dim=1, keepdim=True)

    # Repeat the averaged weights for the new number of input channels
    new_weights = new_weights.repeat(1, new_in_channels, 1, 1)

    # Ensure the new weights tensor requires gradients if the original did
    new_weights.requires_grad_(original_weights.requires_grad)

    # Update the conv layer's weight parameter
    conv_layer.weight = nn.Parameter(new_weights)

    # Update in_channels attribute
    conv_layer.in_channels = new_in_channels


class VGG2D(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(VGG2D, self).__init__()
        self.in_channels = config.get('in_channels', 1)
        self.num_classes = config.get('num_classes', 2)
        # Configurable pretraining
        use_pretrained = config.get('use_pretrained_vgg', True)

        # Load pretrained VGG11 model if requested
        if use_pretrained:
            weights = VGG11_Weights.IMAGENET1K_V1
            vgg_pretrained = vgg11(weights=weights)
            self.features = vgg_pretrained.features
            self.avgpool = vgg_pretrained.avgpool # Use the original avgpool

            # Adapt the first convolutional layer for the specified number of input channels
            first_conv_layer = self.features[0]
            adapt_conv_weights(first_conv_layer, self.in_channels, first_conv_layer.weight.data)

            # Replace the final layer of the classifier
            num_features = vgg_pretrained.classifier[6].in_features
            self.classifier = nn.Linear(num_features, self.num_classes)

        else:
            # Define VGG11 architecture from scratch if not using pretrained
            # (Keeping original structure for reference, but slightly simplified)
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # Standard VGG avgpool output size

            # Calculate input size for classifier
            # Assuming input size that results in 7x7 feature map before avgpool
            # For example, 224x224 input -> 7x7 map after 5 pool layers
            # We adapt dynamically instead

            # Temporarily run dummy input to get feature size
            with torch.no_grad():
                 dummy_input_size = config.get('vgg_input_size', 224) # Assume typical VGG size if not specified
                 dummy_input = torch.zeros(1, self.in_channels, dummy_input_size, dummy_input_size)
                 features_out = self.avgpool(self.features(dummy_input))
                 fc_input_features = features_out.view(features_out.size(0), -1).shape[1]

            self.classifier = nn.Sequential(
                nn.Linear(fc_input_features, 4096), # Standard VGG classifier sizes
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes),
            )
            # Initialize weights if not pretrained
            self._initialize_weights()

    # Only needed if NOT using pretrained weights
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # Added BatchNorm initialization just in case
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, x):
        """Forward pass with softmax activation."""
        x = self.forward(x)
        return torch.softmax(x, dim=1)

