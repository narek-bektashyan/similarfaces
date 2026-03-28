import cv2
import numpy as np
import sys
import os

# Add parent directory to path so facelib can be imported when running from examples/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facelib import detect_faces, extract_features, compare_faces

def main():
    """
    Demonstrates the clean and robust functional API of facelib.
    """
    # 1. Load high-quality sample images
    image_path1 = "images/image1.png"
    image_path2 = "images/image2.png"

    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        print(f"Error: Could not load images from {image_path1} or {image_path2}")
        return

    print("--- facelib: Robust Functional Face Comparison Demo ---")
    print(f"Comparing: {image_path1} vs {image_path2}")

    # 2. Detect and automatically score quality
    faces1 = detect_faces(img1)
    faces2 = detect_faces(img2)

    if not faces1 or not faces2:
        print("No faces found in one or both images.")
        return

    # 3. Pick the best face from each image
    face1 = max(faces1, key=lambda x: x.quality_score)
    face2 = max(faces2, key=lambda x: x.quality_score)

    print(f"\nFace 1 Quality: {face1.quality_score:.4f} (Score: {face1.score:.4f})")
    print(f"Face 2 Quality: {face2.quality_score:.4f} (Score: {face2.score:.4f})")

    # 4. Extract embeddings
    face1.embedding = extract_features(img1, face1)
    face2.embedding = extract_features(img2, face2)

    # 5. Compute similarity
    similarity = compare_faces(face1, face2)
    
    is_same = similarity > 0.6
    print(f"Similarity Score: {similarity:.4f}")
    print(f"Decision: {'MATCH' if is_same else 'NO MATCH'}")

    # 6. Visualize
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/result1.jpg", face1.draw(img1))
    cv2.imwrite("output/result2.jpg", face2.draw(img2))
    print("\nVisualization saved to 'output/' directory.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
