# 테스트 못해봄

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pandas as pd
import joblib

# MediaPipe HandLandmarker 초기화
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)  # 손 하나만 감지하도록 수정
detector = vision.HandLandmarker.create_from_options(options)

# joblib 모델 로드
model = joblib.load('sign_language_model.joblib') #학습시킨 joblib 파일 이름으로 변경해주세요.

def convert_landmarks_to_dataframe(detection_result):
    """랜드마크 결과를 DataFrame으로 변환 (실시간 처리에 최적화)"""
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0] # 첫 번째 손만 사용
        hand_data = [coord for landmark in hand_landmarks for coord in [landmark.x, landmark.y, landmark.z]]
        df = pd.DataFrame([hand_data], columns=[f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']])
        return df
    else:
        return pd.DataFrame() # 빈 DataFrame 반환


# 웹캠 캡처
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        print("프레임 읽기 실패.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # 손 랜드마크 감지
    detection_result = detector.detect(mp_image)

    # 랜드마크를 DataFrame으로 변환하고 예측 수행
    landmarks_df = convert_landmarks_to_dataframe(detection_result)
    if not landmarks_df.empty:
      try:
          prediction = model.predict(landmarks_df)
          predicted_label = prediction[0]
          print(f"Predicted Label: {predicted_label}")

          # 예측 결과 화면에 표시
          cv2.putText(frame, str(predicted_label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      except Exception as e:
          print(f"예측 중 오류 발생: {e}")

    # 랜드마크 시각화 (선택적)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close()
video_capture.release()
cv2.destroyAllWindows()