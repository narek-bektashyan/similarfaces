import cv2
import numpy as np
import os
import sys

# Add parent directory to path so facelib can be imported when running from examples/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functional wrappers directly for a cleaner look
from facelib import detect_faces, align_face

def main():
    """
    Demonstrates face detection and alignment with integrated quality scoring.
    """
    # 1. Load image
    image_path = "images/image1.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"--- facelib: Integrated Detection & Quality Demo ---")

    # 2. Detect faces using the simple functional wrapper
    # This automatically computes quality scores for each face.
    faces = detect_faces(image)

    print(f"Found {len(faces)} faces in {image_path}")

    # Create output directory for aligned faces
    output_dir = "output_aligned"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Process each face individually
    for i, face in enumerate(faces):
        print(f"\nProcessing Face {i+1}:")
        print(f"  Detection Score: {face.score:.4f}")
        print(f"  Quality Score: {face.quality_score:.4f}")

        # 4. Align face using the functional wrapper
        aligned_face = align_face(image, face)
        
        # Save results
        out_path = os.path.join(output_dir, f"face_{i+1}.jpg")
        cv2.imwrite(out_path, aligned_face)
        print(f"  Saved aligned face to: {out_path}")

    print(f"\nAll aligned faces saved to '{output_dir}/'")

if __name__ == "__main__":
    main()
