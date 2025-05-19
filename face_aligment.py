import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]
LEFT_EYE_INDICES = [33, 133]
RIGHT_EYE_INDICES = [362, 263]

# Define the relevant indices for lips
LIPS_INDICES = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]))


def occlude_eyes(image, landmarks, image_shape, padding=10):
    h, w, _ = image_shape

    def get_coords(indices):
        xs = [int(landmarks.landmark[i].x * w) for i in indices]
        ys = [int(landmarks.landmark[i].y * h) for i in indices]
        return xs, ys

    # Get coordinates of both eyes
    left_xs, left_ys = get_coords(LEFT_EYE_INDICES)
    right_xs, right_ys = get_coords(RIGHT_EYE_INDICES)

    # Combine both sets of points
    all_xs = left_xs + right_xs
    all_ys = left_ys + right_ys

    x_min = max(0, min(all_xs) - padding - 5)
    x_max = min(w, max(all_xs) + padding + 5)
    y_min = max(0, min(all_ys) - padding - 6)
    y_max = min(h, max(all_ys) + padding + 6)

    # Draw a single black rectangle over both eyes
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

    return image

def occlude_lips(image, landmarks, image_shape, padding=10):
    h, w, _ = image_shape

    def get_coords(indices):
        xs = [int(landmarks.landmark[i].x * w) for i in indices]
        ys = [int(landmarks.landmark[i].y * h) for i in indices]
        return xs, ys


    # Lips
    lips_xs, lips_ys = get_coords(LIPS_INDICES)
    x_min_lips = max(0, min(lips_xs) - padding - 8)
    x_max_lips = min(w, max(lips_xs) + padding + 8)
    y_min_lips = max(0, min(lips_ys) - padding)
    y_max_lips = min(h, max(lips_ys) + padding)

    cv2.rectangle(image, (x_min_lips, y_min_lips), (x_max_lips, y_max_lips), (0, 0, 0), thickness=-1)

    return image

def detect_landmarks(image, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]


def extract_polygon_points(landmarks, image_shape):
    h, w, _ = image_shape
    points = []
    for idx in FACE_OVAL_INDICES:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])
    return np.array(points, dtype=np.int32)


def get_eye_centers(landmarks, image_shape):
    h, w, _ = image_shape

    def avg_point(indices):
        xs = [landmarks.landmark[i].x * w for i in indices]
        ys = [landmarks.landmark[i].y * h for i in indices]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    left_eye = avg_point(LEFT_EYE_INDICES)
    right_eye = avg_point(RIGHT_EYE_INDICES)
    return left_eye, right_eye


def align_face(image, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Rotate image around the center between eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return aligned


def crop_face_with_polygon(image, polygon_points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon_points, 255)
    masked_face = cv2.bitwise_and(image, image, mask=mask)

    x, y, w_box, h_box = cv2.boundingRect(polygon_points)
    # Ensure crop doesn't go outside the image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + w_box)
    y2 = min(image.shape[0], y + h_box)

    face_crop = masked_face[y1:y2, x1:x2]

    if face_crop.size == 0:
        print("Warning: Empty crop detected.")
        return None

    return face_crop


def extract_face(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        landmarks = detect_landmarks(image, face_mesh)
        if landmarks is None:
            print("\n\nNo face detected.")
            return None

        # Align the face
        left_eye, right_eye = get_eye_centers(landmarks, image.shape)
        aligned_image = align_face(image, left_eye, right_eye)

        # Redetect landmarks on aligned image
        landmarks = detect_landmarks(aligned_image, face_mesh)
        if landmarks is None:
            print("\n\nNo face detected after alignment.")
            return None

        occludeEyes = occlude_eyes(aligned_image.copy(), landmarks, aligned_image.shape)
        occludeLips = occlude_lips(aligned_image.copy(), landmarks, aligned_image.shape)

        polygon_points = extract_polygon_points(landmarks, aligned_image.shape)


        face_crop = crop_face_with_polygon(aligned_image, polygon_points)
        occludeEyes = crop_face_with_polygon(occludeEyes, polygon_points)
        occludeLips = crop_face_with_polygon(occludeLips, polygon_points)


        return face_crop, occludeEyes, occludeLips
    
