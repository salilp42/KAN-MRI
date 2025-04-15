import torch
import numpy as np
import scipy.ndimage

class DataAugmentation:
    """Class for data augmentation operations."""
    
    @staticmethod
    def random_rotation(image, max_angle=10):
        """Apply random rotation."""
        angle = np.random.uniform(-max_angle, max_angle)
        return torch.from_numpy(
            scipy.ndimage.rotate(image.numpy(), angle, axes=(-2, -1), reshape=False)
        )
    
    @staticmethod
    def random_scaling(image, scale_range=(0.9, 1.1)):
        """Apply random scaling."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled = torch.nn.functional.interpolate(
            image.unsqueeze(0), 
            scale_factor=scale, 
            mode='bilinear'
        ).squeeze(0)
        return torch.nn.functional.interpolate(
            scaled.unsqueeze(0),
            size=image.shape[-2:],
            mode='bilinear'
        ).squeeze(0)
    
    @staticmethod
    def random_intensity(image, intensity_range=(-0.1, 0.1)):
        """Apply random intensity variation."""
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        return image * (1 + intensity)
    
    @staticmethod
    def apply_augmentation(image, apply_prob=0.5):
        """Apply all augmentations with given probability."""
        image = torch.as_tensor(image)
        
        if np.random.random() < apply_prob:
            image = DataAugmentation.random_rotation(image)
        if np.random.random() < apply_prob:
            image = DataAugmentation.random_scaling(image)
        if np.random.random() < apply_prob:
            image = DataAugmentation.random_intensity(image)
        
        return image.numpy()

class MRIDataset(torch.utils.data.Dataset):
    """Dataset class with augmentation support."""
    def __init__(self, data, labels, mode='2d', augment=False):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.mode = mode
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.augment:
            image = DataAugmentation.apply_augmentation(image)
        
        return image, label 