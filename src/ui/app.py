import streamlit as st
import mediapipe as mp
import cv2
import time
import numpy as np

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Streamlit 설정
st.title("얼굴 랜드마크 좌표 추출 - 웹캡 캡처")
st.write("웹캠에서 캡처한 이미지의 얼굴 랜드마크 좌표를 배열에 저장합니다.")

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

# 랜드마크 좌표 추출 및 저장 함수
def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(img_rgb)

    landmarks_array = []  # 랜드마크 좌표 저장 리스트

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks_array.append((landmark.x, landmark.y, landmark.z))
    return landmarks_array

# 카운트다운 함수
def countdown(seconds=3):
    for i in range(seconds, 0, -1):
        st.write(f"캡처 시작까지 {i}초 남았습니다...")
        time.sleep(1)

# 버튼 클릭 시 웹캠 캡처 실행
if st.button("웹캠 캡처 시작"):
    countdown(3)  # 캡처 시작 전 3초 카운트다운
    images = capture_images_from_webcam(target_count=6)
    all_landmarks = []  # 모든 이미지의 랜드마크 저장 리스트

    if images:
        st.write("캡처된 이미지를 처리 중입니다...")
        for idx, image_np in enumerate(images):
            retry_count = 0
            while retry_count < 3:  # 최대 3번 재시도
                landmarks = extract_landmarks(image_np)
                if landmarks:
                    all_landmarks.append(landmarks)
                    break
                else:
                    retry_count += 1
                    st.warning(f"이미지 {idx + 1}에서 얼굴 감지 실패. 다시 시도합니다...")

        # 결과 저장 알림
        if len(all_landmarks) == len(images):
            st.success(f"{len(all_landmarks)}개의 이미지에 대해 랜드마크 좌표를 저장했습니다.")
        else:
            st.error("일부 이미지에서 얼굴 감지에 실패했습니다.")