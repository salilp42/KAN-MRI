import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any
import math
import numpy as np

class BaseModel(nn.Module, ABC):
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.mode = config.get('mode', '2d')
        self.in_channels = config.get('in_channels', 1)
        self.num_classes = config.get('num_classes', 2)
        self.pretrained = config.get('pretrained', True)
        self.features = None
        
        # Training parameters
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.warmup_steps = config.get('warmup_steps', 100)
        self.learning_rate = config.get('learning_rate', 1e-3)
        
        # Temperature scaling for better calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize slightly higher
        
        # Class weights for balanced loss - calculate from actual class distribution
        num_samples = config.get('class_counts', [100, 100])  # Default to balanced
        total_samples = sum(num_samples)
        self.class_weights = torch.FloatTensor([
            total_samples / (self.num_classes * count) for count in num_samples
        ])
        self.class_weights = self.class_weights / self.class_weights.sum()
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize bias to small positive value to prevent dead neurons
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use Xavier initialization with careful scaling
                fan_in = m.weight.size(1)
                fan_out = m.weight.size(0)
                std = np.sqrt(2.0 / (fan_in + fan_out))
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    # Initialize bias based on fan-in
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
    
    def get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier based on warmup schedule."""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0
    
    def clip_gradients(self):
        """Clip gradients to prevent exploding gradients."""
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                self.max_grad_norm
            )
    
    def get_loss_fn(self, device: str = 'cpu'):
        """Get class-balanced loss function."""
        weights = self.class_weights.to(device)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)  # Add label smoothing
    
    def forward_with_temperature(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling."""
        logits = self(x)
        return logits / self.temperature
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction with temperature scaling and proper normalization."""
        self.eval()
        with torch.no_grad():
            logits = self.forward_with_temperature(x)
            # Apply class weights to logits
            weighted_logits = logits * self.class_weights.to(x.device).view(1, -1)
            return torch.softmax(weighted_logits, dim=1)
    
    def training_step(self, batch: torch.Tensor, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Perform a training step with gradient clipping and temperature scaling."""
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass with temperature scaling
        outputs = self.forward_with_temperature(inputs)
        
        # Calculate loss with class weights and label smoothing
        loss_fn = self.get_loss_fn(device)
        loss = loss_fn(outputs, targets)
        
        # Add L2 regularization to temperature
        loss = loss + 0.01 * (self.temperature - 1.5).pow(2).mean()
        
        # Backward pass with gradient clipping
        loss.backward()
        self.clip_gradients()
        
        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets
        }
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from the last layer before classification."""
        # Forward pass until the last layer
        if hasattr(self, 'features'):
            return self.features(x)
        elif hasattr(self, 'backbone'):
            return self.backbone(x)
        else:
            raise NotImplementedError("Model must implement features or backbone")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary statistics."""
        return {
            'name': self.__class__.__name__,
            'mode': self.mode,
            'num_parameters': self.get_num_parameters(),
            'model_size_mb': self.get_model_size(),
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained
        }
