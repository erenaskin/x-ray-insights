import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=3, feature_extract=True):
    """
    Creates a ResNet50 model for transfer learning.
    
    Args:
        num_classes (int): Number of output classes.
        feature_extract (bool): If True, freezes the backbone layers.
    """
    # Load pre-trained ResNet50
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    
    # Freeze parameters if feature extracting
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the final fully connected layer
    # ResNet50's fc layer input size is 2048
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
