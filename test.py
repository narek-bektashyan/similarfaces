from similarfaces import detect_faces, extract_features, compare_faces
import cv2

# Load images (cv2 loads as BGR)
img1 = cv2.imread("images/image1.png")
img2 = cv2.imread("images/image3.png")

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