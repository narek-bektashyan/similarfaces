import os
import cv2
import numpy as np
from onnxruntime import InferenceSession
from typing import List, Optional
from .utils import get_model_path
from .aligner import FaceAligner

from .models import Face
from .utils import get_model_path
from .aligner import FaceAligner


class FaceEncoder:
    """
    Face Encoder for extracting robust face embeddings.
    Uses ONNX Runtime for inference. Default embedding size is 512.
    """

    def __init__(self,        model_path: Optional[str] = None,
        providers: List[str] = ["CPUExecutionProvider"]
    ) -> None:
        """
        Initialize the FaceEncoder for embedding extraction.
        
        Args:
            model_path (str, optional): Path to the embedding ONNX model.
            providers (list): ONNX Runtime execution providers.
        """
        self.model_path = model_path or get_model_path("features_extraction.onnx")
        self.input_size = (112, 112)
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5
        self.aligner = FaceAligner()

        try:
            self.session = InferenceSession(self.model_path, providers=providers)
            input_config = self.session.get_inputs()[0]
            self.input_name = input_config.name
            self.output_names = [o.name for o in self.session.get_outputs()]
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Face Embedding model: {e}")

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Resize, normalize and convert cropped face to NCHW format.
        
        Args:
            face_image (np.ndarray): Cropped face image (BGR).
            
        Returns:
            np.ndarray: Preprocessed tensor (1, 3, 112, 112).
        """
        resized_face = cv2.resize(face_image, self.input_size)
        face_blob = cv2.dnn.blobFromImage(
            resized_face,
            scalefactor=1.0 / self.normalization_scale,
            size=self.input_size,
            mean=(self.normalization_mean,) * 3,
            swapRB=True
        )
        return face_blob

    def encode_aligned(self, aligned_face: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract embedding from an already aligned face image.
        
        Args:
            aligned_face (np.ndarray): Aligned face image (112x112).
            normalize (bool): Whether to L2-normalize the output vector.
            
        Returns:
            np.ndarray: Face embedding vector (512-d).
        """
        face_blob = self.preprocess(aligned_face)
        embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]
        embedding = embedding.flatten()

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding /= norm
        return embedding

    def encode(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Perform face alignment and embedding extraction in a single step.
        
        Args:
            image (np.ndarray): Original full image (BGR).
            keypoints (np.ndarray): 5-point facial landmarks for alignment.
            normalize (bool): Whether to L2-normalize the output vector.
            
        Returns:
            np.ndarray: A 512-dimensional face embedding vector.
        """
        aligned_face, _ = self.aligner.align(image, keypoints)
        return self.encode_aligned(aligned_face, normalize=normalize)
