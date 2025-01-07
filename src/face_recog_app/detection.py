import cv2
import time
import streamlit as st

# 웹캡처 함수
def capture_images_from_webcam(target_count=6):
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    captured_images = []
    capture_attempts = 0  # 캡처 시도 횟수

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return []

    st.write("웹캡처를 시작하려면 버튼을 클릭하세요.")
    
    if st.button('캡처 시작'):  # 버튼 클릭 시 캡처 시작
        st.write("웹캠에서 이미지를 캡처 중입니다...")
        while len(captured_images) < target_count:
            capture_attempts += 1
            st.write(f"캡처 진행: {len(captured_images)}/{target_count} (시도 횟수: {capture_attempts})")
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                captured_images.append(frame)
                st.image(frame, caption=f"캡처된 이미지 {len(captured_images)}", use_container_width=True)
            else:
                st.warning(f"이미지 캡처 실패. 다시 시도합니다...")

    cap.release()
    return captured_images