import numpy as np
import cv2
from pathlib import Path
import os
import sys
from typing import Union, Tuple
import urllib.request
from tqdm import tqdm

HF_MODEL_REPOS = {
    "detection.onnx": "https://huggingface.co/similarfaces/face-detector/resolve/main/",
    "features_extraction.onnx": "https://huggingface.co/similarfaces/face-features/resolve/main/",
    "quality_assessment.onnx": "https://huggingface.co/similarfaces/face-quality/resolve/main/",
    "model.onnx.data": "https://huggingface.co/similarfaces/face-quality/resolve/main/",
}

def download_model(model_name: str, model_path: str) -> None:
    """Download a model from Hugging Face if it doesn't exist locally."""
    if not os.path.exists(model_path):
        base_url = HF_MODEL_REPOS.get(model_name, "https://huggingface.co/similarfaces/face-quality/resolve/main/")
        url = base_url + model_name
        print(f"Downloading {model_name} from Hugging Face...")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=model_name) as t:
                def reporthook(count, block_size, total_size):
                    if total_size is not None:
                        t.total = total_size
                    t.update(count * block_size - t.n)
                
                urllib.request.urlretrieve(url, model_path, reporthook=reporthook)
        except Exception as e:
            print(f"\nFailed to download {model_name}: {e}")
            raise RuntimeError(f"Could not download model {model_name} from {url}: {e}")

def get_model_path(model_name: str) -> str:
    """Get the absolute path to a model file within the package."""
    model_path = str(Path(__file__).parent / "models" / model_name)
    download_model(model_name, model_path)
    return model_path

def crop_face(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Crop face from image based on bounding box.
    
    Args:
        image: Original image in BGR/RGB format.
        box: [x1, y1, x2, y2] bounding box.
        
    Returns:
        Cropped face image.
    """
    x1, y1, x2, y2 = map(int, box)
    return image[max(0, y1):y2, max(0, x1):x2]

def adjust_keypoints(keypoints: np.ndarray, crop_box: np.ndarray) -> np.ndarray:
    """
    Adjust keypoints from original image coordinates to cropped region coordinates.
    
    Args:
        keypoints: Array of [x, y] points for the original image.
        crop_box: [x1, y1, x2, y2] bounding box of the cropped region.
    
    Returns:
        Adjusted keypoints for the cropped region.
    """
    x1, y1, _, _ = map(int, crop_box)
    keypoints = np.array(keypoints, dtype=np.float32)
    keypoints[:, 0] -= x1
    keypoints[:, 1] -= y1
    return keypoints

def draw_point(image: np.ndarray, point: Union[tuple, np.ndarray], color: tuple = (0, 255, 0), radius: int = 2) -> None:
    """Draw a point on the image with specified color and radius."""
    cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
