import cv2 as cv
import mediapipe as mp


def get_eyes(frame, eyes):
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    eye_results = eyes.process(image_rgb)
    return eye_results