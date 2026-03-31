import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from typing import List, Union, Optional
from .utils import get_model_path


from .models import Face
from .utils import get_model_path


class FaceQualityScorer:
    """
    ONNX-based Face Quality Scorer.
    Determines if a face image (aligned or cropped) is suitable for recognition.
    A higher score (closer to 1.0) means higher quality.
    """

    def __init__(self, model_path: Optional[str] = None, providers: List[str] = ["CPUExecutionProvider"]):
        """
        Initialize the FaceQualityScorer.
        
        Args:
            model_path (str, optional): Path to the quality assessment ONNX model.
            providers (list): ONNX Runtime execution providers.
        """
        self.model_path = model_path or get_model_path("quality_assessment.onnx")
        
        # Ensure the external weights data file is downloaded
        if not model_path:
            get_model_path("model.onnx.data")
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to load quality model from {self.model_path}: {e}")

        # Normalization parameters (ImageNet)
        self.input_size = (128, 128)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess single image for quality scoring.
        
        Args:
            image (Union[np.ndarray, Image.Image]): Input image.
            
        Returns:
            np.ndarray: Preprocessed tensor (1, 3, 128, 128).
        """
        # Handle color channel conversion (assuming BGR for numpy array from cv2)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        # Resize
        img = cv2.resize(image, self.input_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # CHW + Batch
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def score(self, image: Union[np.ndarray, Image.Image]) -> float:
        """
        Predict quality score for a single face image.
        
        Args:
            image (Union[np.ndarray, Image.Image]): Cropped or aligned face image.
            
        Returns:
            float: Quality score, typically between 0.0 and 1.0.
        """
        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return float(outputs[0][0, 0])

    def score_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[float]:
        """
        Predict quality scores for a batch of images.
        
        Args:
            images (list): List of cropped/aligned face images.
            
        Returns:
            List[float]: List of quality scores.
        """
        return [self.score(img) for img in images]
