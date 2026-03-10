"""
Configuration for Plant Disease Detection - Iteration 1
Binary Classification: Healthy vs. Diseased
"""

import os
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path


@dataclass
class Config:
    """Training configuration"""
    
    # Model settings
    model_name: str = "efficientnet_b0"
    num_classes: int = 2  # Binary: Healthy vs Diseased
    img_size: int = 224
    dropout_rate: float = 0.2
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    
    # Augmentation settings
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    blur_kernel_size: int = 5
    jpeg_quality_range: Tuple[int, int] = (70, 100)
    
    # Dataset split settings
    num_divisions: int = 2
    subsamples_per_division: int = 2
    subsample_ratio: float = 0.8
    val_split: float = 0.2
    
    # MC-Dropout settings
    mc_iterations: int = 20
    
    # Temperature scaling
    temp_lr: float = 0.01
    temp_epochs: int = 50
    
    # OOD detection
    energy_temp: float = 1.0
    
    # Paths
    data_root: str = "./data"
    output_root: str = "./outputs"
    kaggle_json: str = "~/.kaggle/kaggle.json"
    
    # Kaggle datasets
    plantvillage_dataset: str = "abdallahalidev/plantvillage-dataset"
    plantdoc_dataset: str = "pratikdaigavane/plantdoc-dataset"
    
    # System
    num_workers: int = 4
    seed: int = 42
    
    def __post_init__(self):
        """Create necessary directories"""
        Path(self.data_root).mkdir(exist_ok=True, parents=True)
        Path(self.output_root).mkdir(exist_ok=True, parents=True)


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names
CLASS_NAMES = ["Healthy", "Diseased"]