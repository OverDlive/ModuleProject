import cv2
import streamlit as st
from face_recog_app.authentication import extract_landmarks

def capture_images_from_webcam(target_count=1, interval=2, key="capture_button"):
    cap = cv2.VideoCapture(0)  # 웹캡 열기

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return []

    st.write("웹캠에서 이미지를 캡처 중...")
    captured_frames = []

    for i in range(target_count):
        ret, frame = cap.read()
        if not ret:
            st.warning("이미지 캡처 실패. 다시 시도합니다...")
            continue
        
        original_frame = frame.copy()  # 원본 이미지 보존

        # -------------------------------
        # 3D 랜드마크 (x, y, z) 가정
        # -------------------------------
        landmarks = extract_landmarks(frame)  # 예: [(0.1, 0.2, 0.03), (0.4, 0.3, 0.01), ...]

        if not landmarks:
            print(f"{i+1}번째 캡처: 랜드마크를 찾지 못했습니다.")
        else:
            # (x, y, z) 형태로 언패킹
            # 실제 화면 픽셀 좌표로 찍고 싶다면, 이미지 크기 곱해서 int 변환 필요
            for (lx, ly, lz) in landmarks:
                # 예: 웹캠 이미지 크기를 고려해 픽셀 좌표화
                h, w, _ = frame.shape
                px = int(lx * w)  # Mediapipe가 normalized coords 라면
                py = int(ly * h)
                # Z 좌표(lz)는 시각화에 따라 활용
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
        
        st.image(frame, caption=f"{i+1}번째 캡처 결과")
        captured_frames.append(original_frame)

    cap.release()
    return captured_frames
