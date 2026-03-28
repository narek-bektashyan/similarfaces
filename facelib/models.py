from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
import cv2

@dataclass
class Face:
    """
    Structured representation of a detected face.
    
    Attributes:
        bbox (np.ndarray): Bounding box in [x1, y1, x2, y2] format.
        score (float): Confidence score of the detection.
        landmarks (Optional[np.ndarray]): 5-point facial landmarks (eyes, nose, mouth corners).
        quality_score (Optional[float]): Quality score of the face (0.0 to 1.0).
        embedding (Optional[np.ndarray]): 512-dimensional face embedding (L2 normalized).
    """
    bbox: np.ndarray
    score: float
    landmarks: Optional[np.ndarray] = None
    quality_score: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert the Face object to a JSON-serializable dictionary."""
        return {
            "bbox": self.bbox.tolist(),
            "score": float(self.score),
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "quality_score": float(self.quality_score) if self.quality_score is not None else None,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    def draw(self, image: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        Draw bounding box and landmarks on the provided image.
        
        Args:
            image (np.ndarray): Image to draw on (BGR).
            color (tuple): BGR color for the box and points.
            thickness (int): Line thickness.
            
        Returns:
            np.ndarray: Image with visualizations.
        """
        img = image.copy()
        x1, y1, x2, y2 = map(int, self.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if self.landmarks is not None:
            for pt in self.landmarks:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 2, color, -1)
        
        # Draw quality score if available
        if self.quality_score is not None:
            cv2.putText(img, f"Q: {self.quality_score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return img
