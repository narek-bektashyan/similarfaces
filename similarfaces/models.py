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

    def to_dict(self, json_serializable: bool = True) -> dict:
        """
        Convert the Face object to a dictionary representation.
        
        Args:
            json_serializable (bool): If True, converts all numpy arrays to lists 
                for JSON compatibility. If False, preserves numpy arrays.
                Defaults to True.
            
        Returns:
            dict: A dictionary containing 'bbox', 'detection_score', 
                'landmarks', and 'quality_score'.
        """
        if json_serializable:
            return {
                "bbox": self.bbox.tolist(),
                "detection_score": round(float(self.score), 2),
                "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
                "quality_score": round(float(self.quality_score), 2) if self.quality_score is not None else None,
            }
        
        return {
            "bbox": self.bbox,
            "detection_score": round(self.score, 2),
            "landmarks": self.landmarks,
            "quality_score": round(self.quality_score, 2) if self.quality_score is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Face':
        """
        Create a Face object from a dictionary representation.
        
        Args:
            data (dict): Dictionary with keys such as 'bbox', 'score', etc.
            
        Returns:
            Face: A new Face instance populated from the dictionary.
        """
        # Handle both old 'score' and new 'detection_score' keys
        score = data.get("detection_score", data.get("score"))
        if score is None:
            raise KeyError("Dictionary must contain 'detection_score' or 'score'")
            
        return cls(
            bbox=np.array(data["bbox"]) if isinstance(data["bbox"], list) else data["bbox"],
            score=float(score),
            landmarks=np.array(data["landmarks"]) if data.get("landmarks") is not None and isinstance(data["landmarks"], list) else data.get("landmarks"),
            quality_score=float(data["quality_score"]) if data.get("quality_score") is not None else None,
            embedding=np.array(data["embedding"]) if data.get("embedding") is not None and isinstance(data["embedding"], list) else data.get("embedding"),
        )

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
