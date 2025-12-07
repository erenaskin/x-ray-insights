import pydicom
import numpy as np
from PIL import Image
import io

def read_dicom(file_bytes):
    """
    Reads a DICOM file from bytes and returns the image and metadata.
    
    Args:
        file_bytes (bytes): The content of the DICOM file.
        
    Returns:
        image (PIL.Image): The converted image.
        metadata (dict): Extracted metadata (anonymized).
    """
    try:
        # Load DICOM from bytes
        dicom_data = pydicom.dcmread(io.BytesIO(file_bytes))
        
        # Extract metadata
        metadata = {
            "PatientID": str(dicom_data.get("PatientID", "Unknown")),
            "PatientSex": str(dicom_data.get("PatientSex", "Unknown")),
            "PatientAge": str(dicom_data.get("PatientAge", "Unknown")),
            "Modality": str(dicom_data.get("Modality", "Unknown")),
            "StudyDate": str(dicom_data.get("StudyDate", "Unknown")),
            "BodyPartExamined": str(dicom_data.get("BodyPartExamined", "Unknown"))
        }
        
        # Convert pixel array to image
        pixel_array = dicom_data.pixel_array
        
        # Handle Photometric Interpretation (Invert if MONOCHROME1)
        # MONOCHROME1: 0 = White, Max = Black
        # MONOCHROME2: 0 = Black, Max = White (Standard for X-Ray)
        photometric_interpretation = dicom_data.get("PhotometricInterpretation", "MONOCHROME2")
        if photometric_interpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array
        
        # Normalize to 0-255
        if pixel_array.max() > 0:
            pixel_array = pixel_array - pixel_array.min()
            pixel_array = (pixel_array / pixel_array.max()) * 255.0
            
        pixel_array = pixel_array.astype(np.uint8)
        
        # Handle color space if needed (usually grayscale for X-ray)
        if len(pixel_array.shape) == 2:
            image = Image.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            image = Image.fromarray(pixel_array).convert('RGB')
            
        return image, metadata
        
    except Exception as e:
        raise ValueError(f"Error reading DICOM file: {e}")
