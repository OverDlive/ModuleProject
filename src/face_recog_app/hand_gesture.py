import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')
joblib_path = os.path.join(script_dir, 'sign_language_model.joblib')

def convert_landmarks_to_dataframe(detection_result):
    """
    MediaPipe의 손 랜드마크 감지 결과를 판다스 데이터프레임으로 변환합니다.

    Args:
        detection_result: MediaPipe HandLandmarker의 감지 결과

    Returns:
        pandas.DataFrame: 각 손가락 랜드마크의 x, y, z 좌표가 포함된 데이터프레임
    """
    all_hands_data = []

    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hand_data = []

        # 각 랜드마크의 x, y, z 좌표를 추출
        for idx, landmark in enumerate(hand_landmarks):
            hand_data.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # 손의 번호와 좌표 데이터를 딕셔너리로 변환
        hand_dict = {
            'hand_index': hand_idx,
            **{f'landmark_{i}_{coord}': val
               for i in range(21)  # MediaPipe는 각 손마다 21개의 랜드마크를 감지
               for coord, val in zip(['x', 'y', 'z'], hand_data[i*3:(i+1)*3])}
        }
        all_hands_data.append(hand_dict)

    # 데이터프레임 생성
    df = pd.DataFrame(all_hands_data)
    
    # hand_index 열이 존재할 경우에만 삭제
    if 'hand_index' in df.columns:
        df.drop('hand_index', axis=1, inplace=True)

    return df


def predict_sign(frame) -> str:
    # MediaPipe HandLandmarker 초기화
    model_file = open(model_path, "rb")
    model_data = model_file.read()
    model_file.close()

    base_options = python.BaseOptions(model_asset_buffer = model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # OpenCV 프레임을 MediaPipe Image로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # 손 랜드마크 감지
    detection_result = detector.detect(mp_image)
    # 랜드마크 데이터 프레임 생성
    landmarks_df = convert_landmarks_to_dataframe(detection_result)
    if landmarks_df.empty:
        return "NO_HAND_DETECTED"
    # 사전 학습된 모델로 손동작 예측
    model = joblib.load(joblib_path)
    y_pred = model.predict(landmarks_df)        # [숫자]
    # 숫자를 알파벳 문자로 변환
    result = chr(y_pred[0] + ord('A'))          # 대문자 알파벳

    return result




import cv2
import streamlit as st

def finger_landmark_visualization(image_dir: str):
    # MediaPipe HandLandmarker 초기화
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    image1 = mp.Image.create_from_file(image_dir)
    detection_result = detector.detect(image1)
    landmarks_df = convert_landmarks_to_dataframe(detection_result)

    image2 = cv2.imread(image_dir)
    landmark_index = dict()
    tmp = [[0, 1, 5, 9, 13, 17],
           [2, 3, 4],
           [6, 7, 8],
           [10, 11, 12],
           [14, 15, 16],
           [18, 19, 20]]

    for i, v in enumerate(tmp):
        for j in v:
            landmark_index[j] = i
    
    color = [(48, 48, 255), (180, 229, 255), (128, 64, 128),
             (0, 204, 255), (48, 255, 48), (192, 101, 21)]

    h, w, _ = image2.shape

    for i in range(0, len(landmarks_df.columns), 3):
        group = landmarks_df.iloc[0, i:i+3].tolist()  # Series를 리스트로 변환
        index = i // 3
        j = landmark_index[index]
        lx = group[0]
        ly = group[1]

        px = int(lx * w)
        py = int(ly * h)

        cv2.circle(image2, (px, py), 2, color[j], -1)

    st.image(image2)
