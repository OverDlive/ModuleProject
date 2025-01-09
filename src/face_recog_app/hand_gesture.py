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
model_path = os.path.join(script_dir, 'gesture_recognizer.task')
model_file = open(model_path, "rb")
model_data = model_file.read()
model_file.close()

# STEP 1: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_buffer=model_data)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# MediaPipe 이미지 처리 초기화
mp_image = mp.Image

def gesture_detect(frame):
    # STEP 2: OpenCV에서 캡처한 프레임을 MediaPipe Image로 변환
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)  # uint8로 변환
    mp_image_instance = mp_image(image_format=mp.ImageFormat.SRGB, data=frame)

    # 이후 MediaPipe 모델에 mp_image_instance 전달
    # STEP 3: Load the input image.
    #image = mp.Image.create_from_file(mp_image_instance)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(mp_image_instance)

    # STEP 5: 제스처의 종류 반환
    # top_gesture의 형태 : Category(index=-1, score=0.8197612762451172, display_name='', category_name='Pointing_Up')
    if recognition_result.gestures:  # gestures 리스트가 비어 있지 않으면
        top_gesture = recognition_result.gestures[0][0]
        return top_gesture.category_name  # 제스처의 종류 반환
    else:
        return "No gesture recognized"  # 제스처가 인식되지 않았을 경우 처리

'''
# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

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
        #image_resized = cv2.resize(image, (256, 256))  # 예: 256x256으로 크기 조정
        #image_normalized = image_resized / 255.0  # 픽셀 정규화
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB변환
        result = hands.process(image_rgb)
    else:
        raise ValueError("Input should be a valid image object (NumPy array)")
    
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        return normalize_landmarks(landmarks)
    else:
        return None

def normalize_landmarks(landmarks):
    """
    손 랜드마크를 손목 기준으로 정규화하고 스케일 조정.
    
    Args:
        landmarks (np.ndarray): 손 랜드마크 좌표 (21, 3).

    Returns:
        np.ndarray: 정규화된 손 랜드마크 좌표 (63,).
    """
    wrist = landmarks[0]  # 손목 랜드마크
    normalized = landmarks - wrist  # 상대 좌표 계산
    max_value = np.max(np.abs(normalized))  # 최대 절대값
    scaled = normalized / max_value  # 최대값으로 정규화
    return scaled.flatten()

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
'''