# facelib: High-Performance Face Recognition

A production-ready, clean, and robust face recognition pipeline powered by ONNX Runtime.

---

## 🌟 Overview

**facelib** is a streamlined, high-performance Python library for face detection, alignment, quality assessment, and recognition. Designed with modularity and ease of use in mind, it provides a functional API that leverages state-of-the-art models optimized for ONNX Runtime.

### ✨ New Key Features

- 🏗️ **Functional API**: Clean and intuitive functional wrappers (`detect_faces`, `extract_features`, `compare_faces`, `align_face`) without the need to manually manage processor objects.
- 📦 **Structured Data Models**: All functions utilize a unified `Face` dataclass, ensuring type safety and easy access to bounding boxes, landmarks, quality scores, and embeddings.
- 🎯 **Integrated Detection & Quality**: `detect_faces()` now performs both robust face localization and automatic quality assessment in a single, efficient pass.
- 📐 **Optimal Alignment**: Similarity transforms for standardized 112x112 face cropping.
- 🧠 **High-accuracy Recognition**: Extract deep feature embeddings for high-accuracy face comparison.
- ⚡ **ONNX Powered**: Sub-millisecond inference speeds with minimal dependencies across CPU and GPU environments.

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode for development
```

### Basic Usage

Compare two faces with high-quality filtering using the functional API:

```python
import cv2
from facelib import detect_faces, extract_features, compare_faces

# Load images (cv2 loads as BGR)
img1 = cv2.imread("images/image1.png")
img2 = cv2.imread("images/image2.png")

# Detect faces (includes quality scores by default)
faces1 = detect_faces(img1)
faces2 = detect_faces(img2)

if faces1 and faces2:
    # Pick the best face from each image based on quality score
    face1 = max(faces1, key=lambda x: x.quality_score)
    face2 = max(faces2, key=lambda x: x.quality_score)

    # Extract embeddings
    face1.embedding = extract_features(img1, face1)
    face2.embedding = extract_features(img2, face2)

    # Compare faces
    similarity = compare_faces(face1, face2)
    print(f"Similarity: {similarity:.4f}")
    
    if similarity > 0.6:
        print("Outcome: Matches (Same Person)")
    else:
        print("Outcome: No Match")
else:
    print("Error: Could not find faces in one or both images.")
```

---

## 🛠 Project Structure

The library is designed to be developer-friendly and easy to extend:

- `facelib.detector`: High-performance face detection logic.
- `facelib.aligner`: Face alignment and warping.
- `facelib.scorer`: Quality assessment model.
- `facelib.encoder`: Feature embedding extraction.
- `facelib.models`: Data models including the unified `Face` dataclass.

---

## 📊 Performance

| Module | Model | Input Size | Accuracy |
| :--- | :--- | :--- | :--- |
| **Detection** | MobileNet-based | 640x640 | High |
| **Recognition** | IR50-based | 112x112 | SOTA |
| **Quality** | FaceQuality-ONNX | 128x128 | Robust |

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Developed with ❤️ by Narek Bektashyan
</p>
