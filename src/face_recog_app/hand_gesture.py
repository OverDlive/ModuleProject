import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'asl_hand_gesture_model.h5')

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 모델 로드
model = load_model(model_path)

# 레이블 디코딩 설정 (학습 시 사용한 LabelEncoder)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))  # 알파벳 A-Z

# 손 랜드마크 추출 함수
def extract_landmarks_hand(image):
    # image가 이미 OpenCV 이미지 객체(NumPy 배열)인 경우 처리
    if isinstance(image, np.ndarray):
        # 이미지 처리를 바로 진행
        #image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
    else:
        raise ValueError("Input should be a valid image object (NumPy array)")
    #image = cv2.imread(image_path)
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    else:
        return None

# 테스트 함수
def test_hand_gesture(image):
    landmarks = extract_landmarks_hand(image)
    if landmarks is not None:
        landmarks = landmarks.reshape(1, -1)  # 모델 입력 형식에 맞게 변환
        prediction = model.predict(landmarks)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]#, np.max(prediction)  # 예측 레이블과 확률 반환
    else:
        return "No hand detected", 0.0

