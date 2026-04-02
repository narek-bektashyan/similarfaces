import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Union
from .utils import get_model_path

from .models import Face
from .utils import get_model_path
from .aligner import FaceAligner
from .scorer import FaceQualityScorer


def letterbox_resize(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
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
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = target_w - new_w
    dh = target_h - new_h
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
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
        input_size: Tuple[int, int] = (640, 640),
        score_threshold: float = 0.8,
        nms_threshold: float = 0.4,
        use_letterbox: bool = True,
        providers: List[str] = ["CPUExecutionProvider"]
    ) -> None:
        """
        Initialize the FaceDetector.
        
        Args:
            model_path (str, optional): Path to the detection ONNX model.
            input_size (Tuple[int, int]): Model input size (width, height). Defaults to (640, 640).
            score_threshold (float): Confidence threshold for detection. Defaults to 0.8.
            nms_threshold (float): Non-maximum suppression threshold. Defaults to 0.4.
            use_letterbox (bool): Whether to use letterbox resizing. Defaults to True.
            providers (List[str]): ONNX Runtime execution providers.
        """
        self.model_path = model_path or get_model_path("detection.onnx")
        self.input_width, self.input_height = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.use_letterbox = use_letterbox
        self.providers = providers

        # Anchor settings
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.variance = [0.1, 0.2]

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
        
        self.priors = self._generate_priors((self.input_height, self.input_width))

    def _generate_priors(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Generate anchor boxes for detection."""
        anchors = []
        im_h, im_w = image_size

        for k, step in enumerate(self.steps):
            f_h = im_h // step
            f_w = im_w // step
            for i in range(f_h):
                for j in range(f_w):
                    for min_size in self.min_sizes[k]:
                        s_kx = min_size / im_w
                        s_ky = min_size / im_h
                        cx = (j + 0.5) * step / im_w
                        cy = (i + 0.5) * step / im_h
                        anchors.append([cx, cy, s_kx, s_ky])
        return np.array(anchors, dtype=np.float32)

    def _decode_boxes(self, loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """Decode bounding boxes from model output and priors."""
        boxes = np.concatenate([
            priors[:, :2] + loc[:, :2] * self.variance[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])
        ], axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _decode_landmarks(self, ldm: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """Decode facial landmarks from model output and priors."""
        landmarks = np.concatenate([
            priors[:, :2] + ldm[:, i:i+2] * self.variance[0] * priors[:, 2:]
            for i in range(0, 10, 2)
        ], axis=1)
        return landmarks.reshape(-1, 5, 2)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for detection: resize, normalize, and transpose.
        
        Args:
            image (np.ndarray): Original BGR image.
            
        Returns:
            Tuple[np.ndarray, float, Tuple[int, int]]: Preprocessed tensor, resize scale, and padding.
        """
        if self.use_letterbox:
            resized, scale, pad = letterbox_resize(image, (self.input_width, self.input_height))
        else:
            resized = cv2.resize(image, (self.input_width, self.input_height))
            scale = self.input_width / image.shape[1]
            pad = (0, 0)

        img = resized.astype(np.float32)
        img -= np.array([104, 117, 123], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img, scale, pad

    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[Face]:
        """Convert model outputs into Face objects."""
        loc, conf, landms = outputs[0][0], outputs[1][0], outputs[2][0]

        scores = conf[:, 1]
        mask = scores > self.score_threshold
        if not np.any(mask):
            return []

        loc, scores, priors, landms = loc[mask], scores[mask], self.priors[mask], landms[mask]

        boxes = self._decode_boxes(loc, priors)
        landmarks = self._decode_landmarks(landms, priors)

        # Rescale
        boxes[:, [0, 2]] *= self.input_width
        boxes[:, [1, 3]] *= self.input_height
        landmarks[:, :, 0] *= self.input_width
        landmarks[:, :, 1] *= self.input_height

        # Subtract padding
        dw, dh = pad
        boxes[:, [0, 2]] -= dw // 2
        boxes[:, [1, 3]] -= dh // 2
        landmarks[:, :, 0] -= dw // 2
        landmarks[:, :, 1] -= dh // 2

        # Final scale
        boxes /= scale
        landmarks /= scale

        # Clip
        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, orig_w)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, orig_h)

        # NMS
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.1, self.nms_threshold)
        
        results = []
        if len(keep) > 0:
            indices = np.array(keep).flatten()
            for idx in indices:
                results.append(Face(
                    bbox=boxes[idx],
                    score=float(scores[idx]),
                    landmarks=landmarks[idx]
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

