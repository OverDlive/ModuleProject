import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# 얼굴 랜드마크 추출 함수
def extract_landmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    landmarks_array = []  # 얼굴 랜드마크 저장 리스트

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks_array.append((landmark.x, landmark.y, landmark.z))
    return landmarks_array

# 손동작 제스처 추출 함수
def extract_gesture_landmarks(image):
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_landmarks = []  # 손동작 제스처 랜드마크 저장 리스트

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            single_hand = []
            for landmark in hand_landmarks.landmark:
                single_hand.append((landmark.x, landmark.y, landmark.z))
            gesture_landmarks.append(single_hand)
    hands.close()
    return gesture_landmarks
