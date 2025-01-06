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
def capture_images_from_webcam(num_images=6, interval=2):
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    captured_images = []

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return []

    st.write("웹캠에서 이미지를 캡처 중입니다...")
    for i in range(num_images):
        st.write(f"{i + 1}/{num_images}번째 캡처 시작")
        countdown(interval)  # 캡처 전 카운트다운 추가
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_images.append(frame)
            st.image(frame, caption=f"캡처된 이미지 {i + 1}", use_container_width=True)
        else:
            st.warning("이미지 캡처 실패.")
    
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
    images = capture_images_from_webcam()
    all_landmarks = []  # 모든 이미지의 랜드마크 저장 리스트

    if images:
        st.write("캡처된 이미지를 처리 중입니다...")
        for image_np in images:
            landmarks = extract_landmarks(image_np)
            if landmarks:
                all_landmarks.append(landmarks)

        # 결과 저장 알림
        st.write(f"{len(all_landmarks)}개의 이미지에 대해 랜드마크 좌표를 저장했습니다.")