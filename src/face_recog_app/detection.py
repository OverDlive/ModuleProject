import cv2
import streamlit as st
import mediapipe as mp

# 얼굴 랜드마크 추출 함수
def extract_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    
    # 이미지에서 얼굴 랜드마크 추출
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks
    else:
        return None

def capture_images_from_webcam(target_count=1, key="capture"):
    """
    웹캠에서 이미지를 캡처하고 지정된 횟수만큼 이미지를 반환합니다.

    Args:
        target_count (int): 캡처할 이미지 수.
        key (str): 각 캡처에 고유한 키.

    Returns:
        list: 캡처된 이미지 목록.
    """
    images = []
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")  # Streamlit에 캠 오류 메시지 표시
        return []

    st.write("웹캠이 정상적으로 열렸습니다. 이미지 캡처를 시작합니다...")

    #for i in range(target_count):
        #st.write(f"이미지 {i+1} 캡처 중...")
    ret, frame = cap.read()
    if not ret:
        st.error(f"이미지 캡처에 실패했습니다.")
        #break
    images.append(frame)
    #st.progress((i + 1) / target_count)  # 캡처 진행 상황 표시

    cap.release()
    st.write("웹캠이 종료되었습니다.")  # 웹캠 종료 메시지
    return images

def draw_landmarks(image, landmarks):
    """
    랜드마크를 이미지 위에 그립니다.

    Args:
        image (numpy.ndarray): 랜드마크를 그릴 이미지.
        landmarks (list): [(x, y, z), ...] 형태의 랜드마크 좌표 리스트.

    Returns:
        numpy.ndarray: 랜드마크가 그려진 이미지.
    """
    for landmark in landmarks:
        x = int(landmark[0] * image.shape[1])  # 이미지 크기에 맞게 x 좌표 스케일링
        y = int(landmark[1] * image.shape[0])  # 이미지 크기에 맞게 y 좌표 스케일링
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 초록색 점
    return image