import cv2
import time
import streamlit as st

# 웹캠 캡처 함수
def capture_images_from_webcam(target_count=6, interval=2):
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    captured_images = []
    capture_attempts = 0  # 캡처 시도 횟수

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return []

    st.write("웹캠에서 이미지를 캡처 중입니다...")
    while len(captured_images) < target_count:
        capture_attempts += 1
        st.write(f"캡처 진행: {len(captured_images)}/{target_count} (시도 횟수: {capture_attempts})")
        countdown(interval)  # 캡처 전 카운트다운
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_images.append(frame)
            st.image(frame, caption=f"캡처된 이미지 {len(captured_images)}", use_container_width=True)
        else:
            st.warning(f"이미지 캡처 실패. 다시 시도합니다...")

    cap.release()
    return captured_images

# 카운트다운 함수
def countdown(seconds=3):
    for i in range(seconds, 0, -1):
        st.write(f"캡처 시작까지 {i}초 남았습니다...")
        time.sleep(1)
