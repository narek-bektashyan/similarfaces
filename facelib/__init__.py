from .detector import FaceDetector
from .encoder import FaceEncoder
from .aligner import FaceAligner
from .scorer import FaceQualityScorer
from .models import Face

__version__ = "1.0.0"

# Lazy-loaded model instances
_detector = None
_encoder = None

def _get_detector():
    """Internal helper for lazy initialization of the FaceDetector."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector

def _get_encoder():
    """Internal helper for lazy initialization of the FaceEncoder."""
    global _encoder
    if _encoder is None:
        _encoder = FaceEncoder()
    return _encoder

def detect_faces(image: "np.ndarray", score_quality: bool = True) -> "List[Face]":
    """
    Detect faces in an image and calculate their quality scores.
    
    Args:
        image (np.ndarray): Image in BGR format.
        score_quality (bool): Whether to calculate quality scores (default: True).
        
    Returns:
        List[Face]: List of detected faces with bounding boxes, landmarks, and quality.
    """
    return _get_detector().detect(image, score_quality=score_quality)

def extract_features(image: "np.ndarray", face: "Face") -> "np.ndarray":
    """
    Extract facial features (512-d embedding) from an image.
    
    Args:
        image (np.ndarray): Original full image (BGR).
        face (Face): Face object containing landmarks.
        
    Returns:
        np.ndarray: L2-normalized face embedding vector.
    """
    return _get_encoder().encode(image, face.landmarks)

def align_face(image: "np.ndarray", face: "Face") -> "np.ndarray":
    """
    Align a face to a standard 112x112 size for identification tasks.
    
    Args:
        image (np.ndarray): Original full image (BGR).
        face (Face): Face object containing landmarks.
        
    Returns:
        np.ndarray: Aligned 112x112 face image.
    """
    aligned, _ = _get_encoder().aligner.align(image, face.landmarks)
    return aligned

def compare_faces(face1: "Face", face2: "Face") -> float:
    """
    Compute similarity between two faces using their embeddings.
    
    Args:
        face1 (Face): First face with embedding.
        face2 (Face): Second face with embedding.
        
    Returns:
        float: Similarity score (0.0 to 1.0).
    """
    if face1.embedding is None or face2.embedding is None:
        raise ValueError("Both faces must have embeddings for comparison.")
    import numpy as np
    return float(np.dot(face1.embedding, face2.embedding))

__all__ = [
    "Face",
    "FaceDetector",
    "FaceEncoder",
    "FaceAligner",
    "FaceQualityScorer",
    "detect_faces",
    "extract_features",
    "align_face",
    "compare_faces"
]
