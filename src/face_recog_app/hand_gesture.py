import os
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')
joblib_path = os.path.join(script_dir, 'sign_language_model.joblib')

def convert_landmarks_to_dataframe(detection_result):
    """ Mediapipe HandLandmarker → pandas DataFrame 변환 """
    all_hands_data = []
    if detection_result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            hand_data = []
            for lm in hand_landmarks:
                hand_data.extend([lm.x, lm.y, lm.z])
            # 딕셔너리 생성
            hand_dict = {
                'hand_index': hand_idx,
                **{
                    f'landmark_{i}_{coord}': val
                    for i in range(21)
                    for coord, val in zip(['x','y','z'], hand_data[i*3:(i+1)*3])
                }
            }
            all_hands_data.append(hand_dict)
    df = pd.DataFrame(all_hands_data)
    if 'hand_index' in df.columns:
        df.drop('hand_index', axis=1, inplace=True)
    return df

def predict_sign(frame):
    """
    단일 OpenCV 이미지(frame)에서 손 랜드마크를 추출한 뒤,
    사전학습된 sign_language_model.joblib을 이용해 알파벳 예측
    """
    if not os.path.exists(model_path):
        print("[predict_sign] hand_landmarker.task 파일을 찾을 수 없습니다.")
        return "NO_HAND_MODEL"
    if not os.path.exists(joblib_path):
        print("[predict_sign] sign_language_model.joblib 파일을 찾을 수 없습니다.")
        return "NO_SIGN_MODEL"

    # 1) Mediapipe HandLandmarker 초기화
    with open(model_path, "rb") as f:
        model_data = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # 2) Mediapipe Image 변환 후 감지
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)

    # 3) 랜드마크 → DataFrame 변환
    landmarks_df = convert_landmarks_to_dataframe(detection_result)
    if landmarks_df.empty:
        return "NO_HAND_DETECTED"

    # 4) 모델 로드 및 예측
    model = joblib.load(joblib_path)
    y_pred = model.predict(landmarks_df)
    result = chr(y_pred[0] + ord('A'))  # 예: 0->'A', 1->'B', ...
    return result

def finger_landmark_visualization(image_path: str):
    """ (기존과 동일) """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        st.error("이미지를 로드할 수 없습니다.")
        return

    # 손 랜드마크 감지
    # => 예: predict_sign와 유사하지만, 단순 좌표만 표시
    if not os.path.exists(model_path):
        st.error("hand_landmarker.task 파일이 없습니다.")
        return
    with open(model_path, "rb") as f:
        model_data = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_cv)
    detection_result = detector.detect(mp_image)

    # 좌표 시각화
    df = convert_landmarks_to_dataframe(detection_result)
    if df.empty:
        st.error("손이 검출되지 않았습니다.")
        return

    h, w, _ = image_cv.shape
    for i in range(0, len(df.columns), 3):
        group = df.iloc[0, i:i+3].tolist()
        lx, ly, lz = group
        px = int(lx * w)
        py = int(ly * h)
        cv2.circle(image_cv, (px, py), 3, (0,255,0), -1)

    st.image(image_cv, caption="손 랜드마크 시각화 결과")