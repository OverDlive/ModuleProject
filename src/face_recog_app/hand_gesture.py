import os
import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import streamlit as st

# 현재 파일 기준 경로 처리
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

    # detection_result.hand_landmarks: 각 손의 랜드마크 목록
    # 예: List of [NormalizedLandmark(x=..., y=..., z=...), ...]
    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hand_data = []
        for idx, landmark in enumerate(hand_landmarks):
            # (x, y, z) 하나씩 추출하여 리스트에 담음
            hand_data.extend([landmark.x, landmark.y, landmark.z])

        hand_dict = {
            'hand_index': hand_idx,
            **{
                f'landmark_{i}_{coord}': val
                for i in range(21)  # Mediapipe는 각 손마다 21개의 랜드마크를 감지
                for coord, val in zip(['x', 'y', 'z'], hand_data[i*3:(i+1)*3])
            }
        }
        all_hands_data.append(hand_dict)

    df = pd.DataFrame(all_hands_data)
    # hand_index 열이 존재할 경우에만 삭제
    if 'hand_index' in df.columns:
        df.drop('hand_index', axis=1, inplace=True)

    return df


def predict_sign(frame) -> str:
    """
    OpenCV frame(이미지)에서 손 랜드마크를 검출한 뒤, 
    사전 학습된 수어(sign language) 모델로 알파벳을 예측.
    """
    # 1) hand_landmarker.task 파일을 버퍼로 로드
    with open(model_path, "rb") as f:
        model_data = f.read()

    # 2) Mediapipe HandLandmarker 초기화
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # 3) OpenCV BGR 이미지를 Mediapipe Image(SRGB)로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # 4) 손 랜드마크 감지
    detection_result = detector.detect(mp_image)

    # 5) 랜드마크 데이터프레임 생성
    landmarks_df = convert_landmarks_to_dataframe(detection_result)
    if landmarks_df.empty:
        return "NO_HAND_DETECTED"

    # 6) 사전 학습된 모델(joblib)로 예측
    model = joblib.load(joblib_path)
    y_pred = model.predict(landmarks_df)  # [숫자]
    result = chr(y_pred[0] + ord('A'))    # 대문자 알파벳으로 변환
    return result


def finger_landmark_visualization(image_dir: str):
    """
    이미지 파일을 입력받아, 손 랜드마크를 시각화한 이미지를 Streamlit으로 표시.
    """
    # 1) Mediapipe model buffer 로드
    with open(model_path, "rb") as f:
        model_data = f.read()

    # 2) HandLandmarker 초기화 (buffer 활용)
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # 3) 이미지 로드 (Mediapipe Image)
    image_mp = mp.Image.create_from_file(image_dir)
    detection_result = detector.detect(image_mp)

    # 4) 랜드마크 -> DataFrame
    landmarks_df = convert_landmarks_to_dataframe(detection_result)
    if landmarks_df.empty:
        st.error("손이 검출되지 않았습니다.")
        return

    # 5) OpenCV 이미지 로드
    image_cv = cv2.imread(image_dir)
    if image_cv is None:
        st.error("이미지를 로드할 수 없습니다.")
        return

    # 6) 손가락 라인 / 랜드마크 표시를 위한 인덱스 구성
    #    (Mediapipe에서 추천하는 손가락 연결 구조 등)
    #    예: tmp는 특정 관절 인덱스 리스트
    tmp = [
        [0, 1, 5, 9, 13, 17],
        [2, 3, 4],
        [6, 7, 8],
        [10, 11, 12],
        [14, 15, 16],
        [18, 19, 20]
    ]
    landmark_index = {}
    for i, v in enumerate(tmp):
        for j in v:
            landmark_index[j] = i

    color = [
        (48, 48, 255),
        (180, 229, 255),
        (128, 64, 128),
        (0, 204, 255),
        (48, 255, 48),
        (192, 101, 21)
    ]

    h, w, _ = image_cv.shape

    # 7) 각 랜드마크 좌표를 픽셀 단위로 변환하여 원 그리기
    for i in range(0, len(landmarks_df.columns), 3):
        group = landmarks_df.iloc[0, i:i+3].tolist()  # Series -> list
        index = i // 3  # landmark 인덱스 (0~20)

        # tmp/landmark_index를 통해 어떤 손가락 그룹에 속하는지 결정
        group_id = landmark_index.get(index, 0)

        lx, ly, lz = group  # (x, y, z)
        px = int(lx * w)
        py = int(ly * h)

        cv2.circle(image_cv, (px, py), 2, color[group_id], -1)

    st.image(image_cv, caption="손 랜드마크 시각화 결과")
