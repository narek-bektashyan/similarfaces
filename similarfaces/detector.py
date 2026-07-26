import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Union
from .utils import get_model_path

from .models import Face
from .utils import get_model_path
from .aligner import FaceAligner
from .scorer import FaceQualityScorer


def letterbox_resize(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize image with unchanged aspect ratio using padding (letterbox).
    
    Args:
        image (np.ndarray): Input image in BGR format.
        target_size (tuple): Target (width, height).
        
    Returns:
        tuple: (padded_image, scale_factor, (dw, dh) padding).
    """
    img_h, img_w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(round(img_w * scale))
    new_h = int(round(img_h * scale))

    if (img_w, img_h) != (new_w, new_h):
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    dw = (target_w - new_w) / 2
    dh = (target_h - new_h) / 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return padded, scale, (dw, dh)


class FaceDetector:
    """
    High-performance face detector for localization and landmark extraction.
    Integrates quality scoring to provide a comprehensive detection result.
    """

    def __init__(
        self,
        model_path: str = None,
        input_size: Tuple[int, int] = (320, 320),
        score_threshold: float = 0.8,
        use_letterbox: bool = True,
        providers: List[str] = ["CPUExecutionProvider"]
    ) -> None:
        """
        Initialize the FaceDetector.
        
        Args:
            model_path (str, optional): Path to the detection ONNX model.
            input_size (Tuple[int, int]): Model input size (width, height). Defaults to (320, 320).
            score_threshold (float): Confidence threshold for detection. Defaults to 0.8.
            use_letterbox (bool): Whether to use letterbox resizing. Defaults to True.
            providers (List[str]): ONNX Runtime execution providers.
        """
        self.model_path = model_path or get_model_path("detect.onnx")
        self.input_width, self.input_height = input_size
        self.score_threshold = score_threshold
        self.use_letterbox = use_letterbox
        self.providers = providers

        # Core session
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
        except Exception as e:
            raise RuntimeError(f"Failed to load detection model: {e}")

        # Lazy-loaded assistants for integrated quality scoring
        self._aligner = None
        self._scorer = None

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess image for detection: resize, normalize, and transpose.
        
        Args:
            image (np.ndarray): Original BGR image.
            
        Returns:
            Tuple[np.ndarray, float, Tuple[float, float]]: Preprocessed tensor, resize scale, and padding.
        """
        if self.use_letterbox:
            resized, scale, pad = letterbox_resize(image, (self.input_width, self.input_height))
        else:
            resized = cv2.resize(image, (self.input_width, self.input_height))
            scale = min(self.input_width / image.shape[1], self.input_height / image.shape[0])
            pad = (0.0, 0.0)

        # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        blob = np.ascontiguousarray(
            resized[:, :, ::-1].transpose(2, 0, 1)[None, ...],
            dtype=np.float32
        ) / 255.0

        return blob, scale, pad

    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int]
    ) -> List[Face]:
        """Convert model outputs into Face objects."""
        # Output shape: [1, num_detections, 21] or [num_detections, 21]
        # Format per detection: [x1, y1, x2, y2, conf, class_id, kpt1_x, kpt1_y, kpt1_conf, ..., kpt5_x, kpt5_y, kpt5_conf]
        pred = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]

        if pred.shape[0] == 0:
            return []

        # Filter by confidence
        scores = pred[:, 4]
        mask = scores >= self.score_threshold
        if not np.any(mask):
            return []

        pred = pred[mask]
        scores = scores[mask]

        dw, dh = pad
        orig_h, orig_w = orig_shape

        results = []
        for detection in pred:
            confidence = float(detection[4])

            # Decode bounding box (undo letterbox)
            x1 = (detection[0] - dw) / scale
            y1 = (detection[1] - dh) / scale
            x2 = (detection[2] - dw) / scale
            y2 = (detection[3] - dh) / scale

            # Clip to image bounds
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)

            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Decode 5 keypoints (triplets: x, y, visibility)
            kpts_raw = detection[6:]
            landmarks = np.zeros((5, 2), dtype=np.float32)
            for i in range(5):
                kx = (kpts_raw[i * 3] - dw) / scale
                ky = (kpts_raw[i * 3 + 1] - dh) / scale
                landmarks[i, 0] = np.clip(kx, 0, orig_w)
                landmarks[i, 1] = np.clip(ky, 0, orig_h)

            results.append(Face(
                bbox=bbox,
                score=confidence,
                landmarks=landmarks
            ))

        return results

    def detect(self, image: np.ndarray, score_quality: bool = True) -> List[Face]:
        """
        Detect faces and optionally assess their quality.
        
        Args:
            image (np.ndarray): Image in BGR format.
            score_quality (bool): Whether to calculate quality scores for each face.
            
        Returns:
            List[Face]: Detected faces.
        """
        if image is None or image.size == 0:
            return []

        # 1. Base Detection
        orig_shape = image.shape[:2]
        input_tensor, scale, pad = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        faces = self.postprocess(outputs, scale, pad, orig_shape)

        # 2. Integrated Quality Check
        if score_quality and faces:
            if self._aligner is None:
                self._aligner = FaceAligner()
            if self._scorer is None:
                self._scorer = FaceQualityScorer(providers=self.providers)
            
            for face in faces:
                aligned, _ = self._aligner.align(image, face.landmarks)
                face.quality_score = self._scorer.score(aligned)
                
        return faces

