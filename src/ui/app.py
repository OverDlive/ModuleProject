import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Streamlit 설정
st.title("얼굴 랜드마크 추출 및 그리기")
st.write("캡처된 이미지에 얼굴 랜드마크를 그려 출력하며, 랜드마크 좌표를 배열로 저장합니다.")

# 웹캠 캡처 함수
def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        st.error("이미지 캡처에 실패했습니다.")
        return None

# 랜드마크 추출 및 그리기 함수
def process_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(img_rgb)

    landmarks_array = []  # 랜드마크 좌표 저장 리스트

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks_array.append((landmark.x, landmark.y, landmark.z))

            # 랜드마크를 이미지에 그리기
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

    return image, landmarks_array

# 상태 저장
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []
if "landmark_data" not in st.session_state:
    st.session_state.landmark_data = []

# 캡처 진행
if st.button("캡처"):
    image = capture_image_from_webcam()
    if image is not None:
        processed_image, landmarks = process_image(image)
        st.session_state.captured_images.append(processed_image)
        st.session_state.landmark_data.append(landmarks)
        st.image(processed_image, caption=f"캡처된 이미지 {len(st.session_state.captured_images)}", use_container_width=True)

        if landmarks:
            st.success(f"이미지 {len(st.session_state.captured_images)}의 랜드마크 추출 완료!")
        else:
            st.warning(f"이미지 {len(st.session_state.captured_images)}에서 얼굴을 감지하지 못했습니다.")

# 현재 상태 출력
st.write(f"현재 캡처된 이미지 수: {len(st.session_state.captured_images)}/6")

# 결과 출력
if len(st.session_state.captured_images) == 6:
    st.write("6개의 이미지를 모두 캡처했습니다.")
    st.write("랜드마크 좌표 배열:")
    st.write(st.session_state.landmark_data)