import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_gradcam(model, input_tensor, target_layer, target_category=None):
    """
    Generates Grad-CAM heatmap.
    
    Args:
        model: PyTorch model.
        input_tensor: Input image tensor (1, C, H, W).
        target_layer: The target layer for Grad-CAM (e.g., model.layer4[-1]).
        target_category: Integer class index to explain. If None, explains the highest scoring class.
    """
    # Move to CPU for stability with hooks (MPS can be problematic with hooks)
    device = input_tensor.device
    model = model.to('cpu')
    input_tensor = input_tensor.to('cpu')
    
    try:
        # IMPORTANT: Force input to require grad. 
        # This ensures that gradients are propagated through the network even if the backbone is frozen.
        if not input_tensor.requires_grad:
            input_tensor.requires_grad = True
        
        # Construct the CAM object
        # Note: use_cuda=False because we moved to CPU
        # Newer versions of pytorch-grad-cam might not support use_cuda arg, 
        # it infers device from model.
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        # Define targets
        if target_category is not None:
            targets = [ClassifierOutputTarget(target_category)]
        else:
            targets = None # Explains the highest scoring class automatically

        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        # In this example grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam
        
    finally:
        # Restore device (CRITICAL: Must happen even if Grad-CAM fails)
        model = model.to(device)

def overlay_heatmap(original_image, heatmap):
    """
    Overlays heatmap on original image.
    original_image: numpy array (H, W, 3) normalized to [0, 1]
    heatmap: numpy array (H, W)
    """
    visualization = show_cam_on_image(original_image, heatmap, use_rgb=True)
    return visualization
