"""
Model architecture: EfficientNet-B0 with MC-Dropout
"""

import torch
import torch.nn as nn
import timm


class PlantDiseaseModel(nn.Module):
    """
    EfficientNet-B0 classifier with dropout for MC-Dropout uncertainty estimation
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2, pretrained: bool = True):
        super().__init__()
        
        # Load EfficientNet-B0 backbone (without classifier)
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            drop_rate=dropout_rate
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Custom classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def enable_dropout(self):
        """
        Enable dropout during inference for MC-Dropout
        Call this before doing MC-Dropout uncertainty estimation
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def get_features(self, x):
        """Extract features without classification"""
        return self.backbone(x)