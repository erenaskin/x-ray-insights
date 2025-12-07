# 🩻 X-Ray Insight: AI-Powered Pneumonia & COVID-19 Detection System

**X-Ray Insight** is an advanced, industry-standard **Clinical Decision Support System (CDSS)** designed to assist radiologists in detecting **Pneumonia** and **COVID-19** from Chest X-Rays.

It leverages state-of-the-art Deep Learning (ResNet50), Explainable AI (Grad-CAM), and medical imaging standards (DICOM) to provide accurate and interpretable results.

---

## 🚀 Key Features

### 1. 🏥 Medical Standard Compliance (DICOM)
- **DICOM Support:** Natively reads `.dcm` files used in hospitals.
- **Metadata Extraction:** Automatically extracts and anonymizes patient data (Age, Sex, Modality) from DICOM headers.

### 2. 🧠 Advanced AI Architecture
- **Multi-Class Classification:** Detects **Normal**, **Pneumonia**, and **COVID-19** with high accuracy.
- **Transfer Learning:** Fine-tuned **ResNet50** backbone pre-trained on ImageNet.
- **Lung Segmentation (U-Net):** Includes a U-Net based segmentation pipeline to isolate lungs from the background (improving focus).

### 3. 🔍 Explainable AI (XAI)
- **Grad-CAM Integration:** Visualizes the model's attention map, showing exactly *where* in the lungs the model detected an anomaly. This builds trust with medical professionals.

### 4. 🏗️ Modern Software Architecture
- **FastAPI Backend:** Decoupled inference engine serving a REST API (`POST /predict`).
- **Streamlit Frontend:** User-friendly web interface for doctors to upload images and view reports.
- **Dockerized:** Fully containerized application for easy deployment on any cloud platform (AWS, Azure, GCP).

### 5. ⚙️ Robust Training Pipeline
- **Hybrid Dataset:** Combines multiple open-source datasets (Paul Mooney, Tawsifur Rahman, Prashant268, NIH, Bachrr) to create a massive and diverse training set (~25,000 images).
- **Class Imbalance Handling:** Implements **Weighted Loss (CrossEntropy)** to ensure the model treats rare COVID-19 cases with equal importance to common Normal cases.
- **Data Augmentation:** Albumentations (Rotate, Affine, Noise, Blur) to prevent overfitting.
- **Optimization:** Adam optimizer, Weight Decay, and Learning Rate Scheduler (ReduceLROnPlateau).
- **MLflow Tracking:** Full experiment tracking for metrics and model versioning.

---

## 📊 Model Performance

The model was trained on a massive hybrid dataset of **25,194 Chest X-Rays**.

**Best Model Metrics (Validation Set):**
*   **Accuracy:** 97.72%
*   **Precision (Weighted):** 97%
*   **Recall (Weighted):** 97%
*   **F1-Score (Weighted):** 97%

**Class-wise Performance:**
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **NORMAL** | 96% | 99% | 98% |
| **PNEUMONIA** | 99% | 97% | 98% |
| **COVID-19** | 98% | 93% | 96% |

*Note: The model uses aggressive data augmentation and a confidence threshold mechanism to minimize false positives.*

---

## 🛠️ Tech Stack

*   **Core:** Python 3.9+
*   **Deep Learning:** PyTorch, Torchvision, Segmentation Models PyTorch
*   **Medical Imaging:** Pydicom, OpenCV, Albumentations
*   **Backend API:** FastAPI, Uvicorn
*   **Frontend UI:** Streamlit
*   **DevOps:** Docker
*   **ML Ops:** MLflow
*   **XAI:** pytorch-grad-cam

---

## 📂 Project Structure

```
x_ray_insights/
├── api/                # FastAPI Backend
│   └── main.py         # API Endpoints
├── web_app/            # Streamlit Frontend
│   └── app.py          # UI Logic
├── src/                # Core Logic
│   ├── model.py        # ResNet50 Classification Model
│   ├── segmentation.py # U-Net Lung Segmentation
│   ├── utils_dicom.py  # DICOM Handling Utilities
│   ├── train.py        # Training Loop
│   └── explainability.py # Grad-CAM
├── models/             # Trained Weights
├── Dockerfile          # Container Configuration
└── requirements.txt    # Dependencies
```

---

## ⚙️ Installation & Setup

### Option A: Using Docker (Recommended)

1.  **Build the Image:**
    ```bash
    docker build -t xray-insight .
    ```

2.  **Run the Container:**
    ```bash
    docker run -p 8000:8000 -p 8501:8501 xray-insight
    ```
    - **Frontend:** http://localhost:8501
    - **API Docs:** http://localhost:8000/docs

### Option B: Local Development

1.  **Clone & Setup Environment:**
    ```bash
    git clone <repo-url>
    cd x_ray_insights
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Start the API (Backend):**
    ```bash
    uvicorn api.main:app --reload --port 8000
    ```

3.  **Start the App (Frontend):**
    *(Open a new terminal)*
    ```bash
    streamlit run web_app/app.py
    ```

---

## 🏋️‍♂️ Training the Model

To retrain the model with new data:

1.  **Download Data:**
    ```bash
    python kaggle.py
    ```
2.  **Run Training:**
    ```bash
    python src/train.py
    ```
    *Metrics will be logged to MLflow.*

---

## 👨‍⚕️ Usage Scenario

1.  **Upload:** A radiologist uploads a patient's Chest X-Ray (DICOM or JPEG).
2.  **Process:**
    *   System extracts patient metadata.
    *   Segments the lung area.
    *   Classifies the image (Normal/Pneumonia/COVID).
3.  **Result:**
    *   Displays the diagnosis with a confidence score.
    *   Shows a **Grad-CAM heatmap** highlighting the infected area.
    *   Displays patient info from DICOM tags.

---
