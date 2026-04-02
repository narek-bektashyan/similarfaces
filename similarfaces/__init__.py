import numpy as np
from typing import List, Union, Optional
from .detector import FaceDetector
from .encoder import FaceEncoder
from .aligner import FaceAligner
from .scorer import FaceQualityScorer
from .models import Face

__version__ = "1.0.1"

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

def detect_faces(image: np.ndarray, score_quality: bool = True) -> List[dict]:
    """
    Detect faces in an image and calculate their quality scores.
    
    Args:
        image (np.ndarray): Image in BGR format.
        score_quality (bool): Whether to calculate quality scores for each face. 
            Defaults to True.
            
    Returns:
        List[dict]: A list of detected faces, each represented as a dictionary.
    """
    faces = _get_detector().detect(image, score_quality=score_quality)
    return [face.to_dict(json_serializable=False) for face in faces]

def extract_features(image: np.ndarray, face: Union[Face, dict]) -> dict:
    """
    Extract a 512-dimensional facial feature embedding from an image.
    
    Args:
        image (np.ndarray): Original full image in BGR format.
        face (Union[Face, dict]): Face object or dictionary containing landmarks.
        
    Returns:
        dict: A dictionary containing the 'embedding' (512-d np.ndarray).
    """
    if isinstance(face, dict):
        landmarks = np.array(face["landmarks"])
    else:
        landmarks = face.landmarks
        
    embedding = _get_encoder().encode(image, landmarks)
    return {"embedding": embedding}

def align_face(image: np.ndarray, face: Union[Face, dict]) -> np.ndarray:
    """
    Align a face to a standard 112x112 size suitable for identification tasks.
    
    Args:
        image (np.ndarray): Original full image in BGR format.
        face (Union[Face, dict]): Face object or dictionary containing landmarks.
        
    Returns:
        np.ndarray: Aligned 112x112 face image (BGR).
    """
    if isinstance(face, dict):
        landmarks = np.array(face["landmarks"])
    else:
        landmarks = face.landmarks
        
    aligned, _ = _get_encoder().aligner.align(image, landmarks)
    return aligned

def compare_faces(face1: Union[Face, dict], face2: Union[Face, dict]) -> float:
    """
    Compute cosine similarity between two faces using their embeddings.
    
    Args:
        face1 (Union[Face, dict]): First face with an 'embedding' key or attribute.
        face2 (Union[Face, dict]): Second face with an 'embedding' key or attribute.
        
    Returns:
        float: Similarity score, typically between 0.0 and 1.0.
    """
    def _get_emb(f: Union[Face, dict]) -> Optional[np.ndarray]:
        if isinstance(f, dict):
            return f.get("embedding")
        return getattr(f, "embedding", None)

    emb1 = _get_emb(face1)
    emb2 = _get_emb(face2)
    
    if emb1 is None or emb2 is None:
        raise ValueError("Both faces must have high-quality embeddings for comparison. "
                         "Please call extract_features() first.")
    
    return float(np.dot(emb1, emb2))

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
