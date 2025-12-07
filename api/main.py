from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import get_model
from src.utils_dicom import read_dicom
from src.segmentation import LungSegmenter
from src.explainability import get_gradcam, overlay_heatmap
import base64

app = FastAPI(title="X-Ray Insight API", description="API for Pneumonia & COVID-19 Detection from Chest X-Rays")

# Load Model
MODEL_PATH = "models/pneumonia_covid_resnet50.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

model = get_model(num_classes=3, feature_extract=False)
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: Model file not found.")

model.to(DEVICE)
model.eval()

# Target layer for Grad-CAM (ResNet50)
target_layer = model.layer4[-1]

# Load Segmenter
segmenter = LungSegmenter(device='cpu') # Segmentation usually fine on CPU for inference

# Preprocessing
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

@app.get("/")
def read_root():
    return {"message": "X-Ray Insight API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename.lower()
        metadata = {}
        
        # Handle DICOM
        if filename.endswith('.dcm'):
            image, metadata = read_dicom(contents)
        else:
            # Handle standard images
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
        image_np = np.array(image)
        
        # 1. Segmentation (Optional but recommended)
        # masked_image, mask = segmenter.segment(image_np)
        # For now, we use the original image but you can switch to masked_image
        # image_to_process = masked_image 
        image_to_process = image_np # Using original for now as model was trained on full X-rays
        
        # 2. Preprocess
        transformed = transform(image=image_to_process)
        input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        # 3. Inference with TTA (Test Time Augmentation)
        with torch.no_grad():
            # Logit Bias for Calibration
            # Increasing Pneumonia sensitivity (Class 1) to address class imbalance/similarity with COVID
            PNEUMONIA_BIAS = 1.0  # Adds to the raw score (logit) of Pneumonia
            COVID_PENALTY = 0.5   # Subtracts from COVID score
            
            # Original
            outputs = model(input_tensor)
            outputs[0][1] += PNEUMONIA_BIAS
            outputs[0][2] -= COVID_PENALTY
            probs_original = torch.nn.functional.softmax(outputs, dim=1)
            
            # TTA: Horizontal Flip
            # Flip the numpy image first, then transform
            image_flipped = cv2.flip(image_to_process, 1)
            transformed_flipped = transform(image=image_flipped)
            input_tensor_flipped = transformed_flipped['image'].unsqueeze(0).to(DEVICE)
            
            outputs_flipped = model(input_tensor_flipped)
            outputs_flipped[0][1] += PNEUMONIA_BIAS
            outputs_flipped[0][2] -= COVID_PENALTY
            probs_flipped = torch.nn.functional.softmax(outputs_flipped, dim=1)
            
            # Average probabilities
            probs = (probs_original + probs_flipped) / 2.0
            
            # Get top 2 predictions to handle uncertainty
            top_k_probs, top_k_classes = probs.topk(2, dim=1)
            
            # Get all probabilities for debugging
            all_probs = probs[0].cpu().numpy().tolist()
            
        top_class_idx = top_k_classes[0][0].item()
        top_confidence = top_k_probs[0][0].item()
        
        # Threshold Logic for COVID-19 (Class 2)
        # If prediction is COVID but confidence is low (< 0.90), fallback to second best
        if top_class_idx == 2 and top_confidence < 0.90:
            class_idx = top_k_classes[0][1].item()
            confidence = top_k_probs[0][1].item()
            print(f"COVID-19 confidence ({top_confidence:.2f}) below threshold. Falling back to class {class_idx} ({confidence:.2f})")
        else:
            class_idx = top_class_idx
            confidence = top_confidence
        
        classes = {0: "NORMAL", 1: "PNEUMONIA", 2: "COVID-19"}
        prediction = classes[class_idx]
        
        # 4. Grad-CAM
        gradcam_b64 = None
        gradcam_error = None
        try:
            # Generate heatmap
            heatmap = get_gradcam(model, input_tensor, target_layer, target_category=class_idx)
            
            # Prepare image for overlay (Resize to 224x224 and normalize to [0, 1])
            # We use the original image_to_process, resize it to match model input size
            vis_img = cv2.resize(image_to_process, (224, 224))
            vis_img = np.float32(vis_img) / 255.0
            
            # Overlay
            visualization = overlay_heatmap(vis_img, heatmap)
            
            # Convert to base64
            # visualization is RGB (from show_cam_on_image with use_rgb=True), convert to BGR for cv2.imencode
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', visualization_bgr)
            gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            gradcam_error = str(e)
        
        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "NORMAL": all_probs[0],
                "PNEUMONIA": all_probs[1],
                "COVID-19": all_probs[2]
            },
            "metadata": metadata,
            "gradcam_image": gradcam_b64,
            "gradcam_error": gradcam_error
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
