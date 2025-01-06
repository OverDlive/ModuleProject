import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Streamlit 설정
st.title("얼굴 랜드마크 좌표 추출 - 웹캠 캡처")
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
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_images.append(frame)
            st.image(frame, caption=f"캡처된 이미지 {i + 1}", use_column_width=True)
        else:
            st.warning("이미지 캡처 실패.")
        cv2.waitKey(interval * 1000)  # 2초 대기

    cap.release()
    return captured_images

# 랜드마크 좌표 추출 및 저장 함수
def extract_landmarks(image, idx):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(img_rgb)

    landmarks_array = []  # 랜드마크 좌표 저장 리스트

    if results.multi_face_landmarks:
        st.write(f"이미지 {idx + 1}의 랜드마크 좌표:")
        for face_landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                landmarks_array.append((landmark.x, landmark.y, landmark.z))
                st.write(f"랜드마크 {i}: (x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f})")
    else:
        st.write(f"이미지 {idx + 1}에서 얼굴을 감지하지 못했습니다.")

    return landmarks_array

# 버튼 클릭 시 웹캠 캡처 실행
if st.button("웹캠 캡처 시작"):
    images = capture_images_from_webcam()
    all_landmarks = []  # 모든 이미지의 랜드마크 저장 리스트

    if images:
        st.write("캡처된 이미지를 처리 중입니다...")
        for idx, image_np in enumerate(images):
            landmarks = extract_landmarks(image_np, idx)
            if landmarks:
                all_landmarks.append(landmarks)

        # 모든 랜드마크 출력
        st.write("모든 랜드마크 좌표 배열:")
        st.json(all_landmarks)  # JSON 형식으로 랜드마크 출력
