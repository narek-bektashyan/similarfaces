import cv2
import numpy as np
from skimage.transform import SimilarityTransform
from typing import Tuple, Optional


class FaceAligner:
    """
    Face Alignement utility using 5-point facial landmarks.
    """

    # Standard 5-point facial landmark reference (ArcFace-style)
    STANDARD_LANDMARK_TEMPLATE = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041],  # Right mouth corner
    ], dtype=np.float32)

    def __init__(self, target_size: int = 112) -> None:
        self.target_size = target_size

    def compute_alignment_matrix(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the transformation matrix to align a face.
        """
        points = np.array(points, dtype=np.float32)
        if points.shape != (5, 2):
            raise ValueError(f"Expected 5-point landmarks, got {points.shape}")

        scale = self.target_size / 112.0
        ref_points = self.STANDARD_LANDMARK_TEMPLATE.copy() * scale
        
        transform = SimilarityTransform()
        transform.estimate(points, ref_points)

        affine = transform.params[:2, :]
        affine_inv = np.linalg.inv(transform.params)[:2, :]
        return affine, affine_inv

    def align(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform facial alignment.
        
        Args:
            image: Original image (HWC).
            keypoints: 5-point landmarks (5, 2).
            
        Returns:
            The aligned face image and the inverse affine matrix.
        """
        affine_matrix, inverse_matrix = self.compute_alignment_matrix(keypoints)
        aligned_face = cv2.warpAffine(
            image, affine_matrix, (self.target_size, self.target_size), borderValue=0.0
        )
        return aligned_face, inverse_matrix
