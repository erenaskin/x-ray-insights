import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from PIL import Image

class LungSegmenter:
    def __init__(self, device='cpu'):
        self.device = device
        # Initialize U-Net with ResNet34 encoder
        # Note: In a real scenario, you would load weights trained on a Lung Segmentation dataset.
        # Since we don't have those weights here, we will use a fallback method (Otsu's thresholding)
        # if no weights are provided, but the architecture is ready.
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.model.to(self.device)
        self.model.eval()
        self.has_trained_weights = False # Set to True if you load custom weights

    def segment(self, image_np):
        """
        Segments lungs from the image.
        
        Args:
            image_np (numpy.ndarray): Input image (H, W, 3) RGB.
            
        Returns:
            masked_image (numpy.ndarray): Image with background removed (H, W, 3).
            mask (numpy.ndarray): Binary mask (H, W).
        """
        if self.has_trained_weights:
            # DL based segmentation (Placeholder for when weights are available)
            # Preprocess
            # Predict
            # Postprocess
            pass
        else:
            # Fallback: Traditional Image Processing (Otsu's Thresholding + Contours)
            # This is a robust method for X-rays when a specific DL model isn't available.
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian Blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Otsu's thresholding
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if necessary (lungs should be black in X-ray, but we want them white in mask)
            # Usually X-rays: Bones/Dense = White, Air/Lungs = Black.
            # So thresholding usually makes lungs black (0) and background white (255).
            # We want mask where lungs are 1 (or 255).
            
            # Check if corners are white (background). If so, invert.
            if np.mean(thresh[:10, :10]) > 127:
                thresh = cv2.bitwise_not(thresh)
                
            # Morphological operations to remove noise
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create empty mask
            mask = np.zeros_like(gray)
            
            # Filter contours based on area (keep largest ones, likely lungs)
            img_area = gray.shape[0] * gray.shape[1]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > img_area * 0.05: # Keep contours larger than 5% of image
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Apply mask to original image
            masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)
            
            return masked_image, mask
