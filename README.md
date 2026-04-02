# similarfaces: High-Performance Face Recognition

A production-ready, clean, and robust face recognition pipeline powered by ONNX Runtime.

---

## 🌟 Overview

**similarfaces** is a streamlined Python library for face detection, alignment, quality assessment, and recognition. It provides a simple functional API and automatically downloads optimized ONNX models on demand.

### ✨ Key Features

- 🏗️ **Simple Functional API**: `detect_faces`, `extract_features`, `compare_faces`, `align_face`.
- 📦 **Dictionary-Based Results**: Functions return standard Python dicts for easy JSON serialization.
- ⏬ **Auto-Downloading**: Models are automatically fetched from Hugging Face with a beautiful `tqdm` progress bar.
- 🎯 **Integrated Quality**: Each detected face includes a quality score (0.0 to 1.0).
- ⚡ **ONNX Powered**: High-speed inference using ONNX Runtime.

---

## 🚀 Quick Start

### Installation

```bash
pip install similarfaces
```

### 1. Simple Face Detection
```python
from similarfaces import detect_faces
import cv2

image = cv2.imread("image.jpg")
faces = detect_faces(image)

for face in faces:
    print(f"Found face with confidence {face['score']:.2f}")
    print(f"Quality Score: {face['quality_score']:.2f}")
    print(f"Bounding Box: {face['bbox']}")
```

### 2. Compare Two Faces
```python
import cv2
from similarfaces import detect_faces, extract_features, compare_faces

# Load images
img1, img2 = cv2.imread("face1.jpg"), cv2.imread("face2.jpg")

# 1. Detect (returns list of dicts)
face1 = detect_faces(img1)[0]
face2 = detect_faces(img2)[0]

# 2. Extract 512-d embeddings
face1["embedding"] = extract_features(img1, face1)["embedding"]
face2["embedding"] = extract_features(img2, face2)["embedding"]

# 3. Compare (returns 0.0 to 1.0)
similarity = compare_faces(face1, face2)
print(f"Similarity: {similarity:.4f}")
print("Same person!" if similarity > 0.6 else "Different people.")
```

---

## 🛠 Advanced Usage

### Using the Face object
If you prefer objects over dictionaries, you can use the `Face` class:

```python
from similarfaces import detect_faces, Face

faces_dicts = detect_faces(image)
face_obj = Face.from_dict(faces_dicts[0])

# Now use object methods
annotated_img = face_obj.draw(image)
cv2.imwrite("result.jpg", annotated_img)
```

---

## 📝 License

This project is licensed under the MIT License.

---

<p align="center">
  Developed with ❤️ by Narek Bektashyan
</p>
