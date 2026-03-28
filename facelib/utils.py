import numpy as np
import cv2
from pathlib import Path

def get_model_path(model_name: str) -> str:
    """Get the absolute path to a model file within the package."""
    return str(Path(__file__).parent / "models" / model_name)

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

def draw_point(image: np.ndarray, point: tuple, color=(0, 255, 0), radius=2):
    """Draw a point on the image."""
    cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
