import cv2 as cv
import mediapipe as mp
import numpy as np


def get_eye_landmarks(frame, face_mesh):
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    left_eye = []
    right_eye = []
    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            for idx in left_eye_indices:
                left_eye.append((face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h))
            for idx in right_eye_indices:
                right_eye.append((face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h))

    return left_eye, right_eye
