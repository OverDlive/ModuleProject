import cv2
import streamlit as st
from face_recog_app.authentication import extract_landmarks

# 웹캡처 함수
def capture_images_from_webcam(target_count=1, interval=2, key="capture_button"):
    cap = cv2.VideoCapture(0)  # 웹캡 열기
    captured_images = []

    if not cap.isOpened():
        st.error("웹캡을 열 수 없습니다.")
        return []

    st.write("웹캡에서 이미지를 캡처 중...")
    for _ in range(target_count):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 랜드마크 추출
            landmarks = extract_landmarks(frame)  # 얼굴 랜드마크 추출
            
            # 랜드마크 그리기
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 초록색 원으로 랜드마크 그리기
            
            captured_images.append(frame)
            st.image(frame, caption=f"캡처된 이미지", use_container_width=True)
        else:
            st.warning(f"이미지 캡처 실패. 다시 시도합니다...")

    cap.release()
    return captured_images