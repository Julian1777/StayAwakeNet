import cv2 as cv
import mediapipe as mp
import numpy as np

from ear import get_eye_landmarks
from gaze import get_head_pose_angles
from hand import get_hand

#video_source = './vids/self_eye.mov'
video_source = './vids/driving1.mp4'
hand_video_source = './vids/hands1.mp4'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv.VideoCapture(video_source)
hand_cap = cv.VideoCapture(hand_video_source)

EYES_CLOSED_FRAMES = 0
EYES_CLOSED_THRESHOLD = 30

NO_HAND=0
NO_HAND_THRESHOLD = 20

def compute_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye

    def ecuclidean(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    ear = (ecuclidean(p2, p6) + ecuclidean(p3, p5)) / (2 * ecuclidean(p1, p4))
    return ear


while True:
    ret, frame = cap.read()
    ret_hand, hand_frame = hand_cap.read()
    if not ret:
        break

    # Face/eye/head pose detection on main video
    left_eye, right_eye = get_eye_landmarks(frame, face_mesh)

    if len(left_eye) == 6 and len(right_eye) == 6:
        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < 0.2:
            EYES_CLOSED_FRAMES += 1
        else:
            EYES_CLOSED_FRAMES = 0

        if EYES_CLOSED_FRAMES >= EYES_CLOSED_THRESHOLD:
            cv.putText(frame, 'WARNING: Eyes Closed!', (30, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.putText(frame, f'Left EAR: {left_ear:.2f}', (30, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f'Right EAR: {right_ear:.2f}', (30, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (x, y) in left_eye + right_eye:
            cv.circle(frame, (int(x), int(y)), 2, (255,0,0), -1)

    head_points = get_head_pose_angles(frame, face_mesh)
    if head_points is not None and len(head_points) == 6:
        for (x, y) in head_points:
            cv.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

    pose = get_head_pose_angles(frame, face_mesh)
    if pose is not None and len(pose) == 3:
        pitch, yaw, roll = pose

        cv.putText(frame, f'Pitch: {pitch:.1f}', (30, 180), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv.putText(frame, f'Yaw: {yaw:.1f}', (30, 210), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv.putText(frame, f'Roll: {roll:.1f}', (30, 240), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if yaw < -25 or yaw > 25:
            cv.putText(frame, 'WARNING: Look at the road!', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
        
        if pitch > 190 or pitch < -180:
            cv.putText(frame, 'WARNING: Head not upright!', (30, 270), cv.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

        if roll < -15 or roll > 15:
            cv.putText(frame, 'WARNING: Head tilting!', (30, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if ret_hand:
        hand_results = get_hand(hand_frame, hands)
        if hand_results.multi_hand_landmarks:
            NO_HAND = 0
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    hand_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            NO_HAND += 1
            if NO_HAND >= NO_HAND_THRESHOLD:
                cv.putText(hand_frame, 'WARNING: Hands not detected!', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv.imshow('Hand Detection', hand_frame)

    cv.imshow('EAR Detection & Head Pose', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hand_cap.release()
cv.destroyAllWindows()