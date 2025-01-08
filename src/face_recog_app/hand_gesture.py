import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os
import numpy as np

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
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2, # 탐지 가능한 최대 손 수
                                           min_hand_detection_confidence=0.3, # 탐지 신뢰도 설정
                                           min_tracking_confidence=0.3 # 추적 신뢰도 설정
                                           )
    detector = vision.HandLandmarker.create_from_options(options)

    # frame이 numpy.ndarray일 경우, 이를 uint8로 변환한 후 mp.Image로 변환
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

     # OpenCV 프레임을 MediaPipe Image로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # 손 랜드마크 감지
    detection_result = detector.detect(mp_image)

    # 손 영역 잘라내기
    #frame = crop_to_hand_area(frame, detection_result)

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
