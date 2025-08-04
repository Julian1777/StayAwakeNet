import cv2 as cv
import mediapipe as mp


def get_hand(frame, hands):
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)
    return hand_results