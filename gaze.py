import cv2 as cv
import mediapipe as mp
import numpy as np


def get_head_pose_angles(frame, face_mesh):
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    head_landmarks = [1, 152, 33, 263, 61, 291]
    model_points = np.array([
        [0.0, 0.0, 0.0],        # Nose tip
        [0.0, -63.6, -12.5],    # Chin
        [-43.3, 32.7, -26.0],   # Left eye left corner
        [43.3, 32.7, -26.0],    # Right eye right corner
        [-28.9, -28.9, -24.1],  # Left mouth corner
        [28.9, -28.9, -24.1]    # Right mouth corner
    ])

    head_points = []

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]
        for idx in head_landmarks:
            landmark = face_landmarks.landmark[idx]
            head_points.append((landmark.x * w, landmark.y * h))
        head_points_np = np.array(head_points, dtype='double')

        for (x, y) in head_points:
            cv.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv.solvePnP(
            model_points, head_points_np, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
        )
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0,0] ** 2 + rotation_matrix[1,0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        return (pitch, yaw, roll)
    return None

