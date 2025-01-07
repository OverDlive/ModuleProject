import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib




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
    df.drop('hand_index', axis=1, inplace=True)
    return df



def predict_sign(image_dir: str) -> str:
   # MediaPipe HandLandmarker 초기화
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_dir)
    detection_result = detector.detect(image)

    landmarks_df = convert_landmarks_to_dataframe(detection_result)
    model = joblib.load('sign_language_model.joblib')
    y_pred = model.predict(landmarks_df)        # [숫자]

    result = chr(y_pred[0] + ord('A'))          # 대문자 알파벳

    return result