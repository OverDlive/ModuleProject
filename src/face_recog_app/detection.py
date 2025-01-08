import cv2
import time
import streamlit as st
import mediapipe as mp

# Mediapipe FaceMesh 초기화 (얼굴 랜드마크)
mp_face_mesh = mp.solutions.face_mesh
# Mediapipe Hands 초기화 (손 랜드마크)
mp_hands = mp.solutions.hands

def capture_images_from_webcam(target_count=1, interval=2, key="capture_button"):
    """
    웹캠을 열고 target_count만큼 이미지를 캡처하여 리스트로 반환.
    interval(초) 동안 대기 후 캡처할 수 있음(필요 시 수정).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return []

    st.write("웹캠에서 이미지를 캡처 중...")
    captured_frames = []

    for i in range(target_count):
        # interval 동안 대기
        time.sleep(interval)

        ret, frame = cap.read()
        if not ret:
            st.warning(f"{i+1}번째 이미지 캡처에 실패했습니다.")
            continue

        captured_frames.append(frame)
        st.write(f"{i+1}번째 이미지를 캡처했습니다.")

    cap.release()
    return captured_frames


def extract_landmarks(frame):
    """
    단일 OpenCV 프레임에서 얼굴 랜드마크를 추출.
    Mediapipe FaceMesh 이용 (Solutions API 예시).
    반환 형태: [(x, y), (x, y), ...] or None
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,       # 단일 이미지
        max_num_faces=1,             # 인식할 얼굴 수
        refine_landmarks=True,       # 눈/입 등 정교한 랜드마크
        min_detection_confidence=0.5
    ) as face_mesh:
        # BGR -> RGB 변환
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            return None

        # 한 명의 얼굴만 처리한다고 가정(max_num_faces=1)
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        landmarks_2d = []
        for lm in face_landmarks.landmark:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            landmarks_2d.append((x_px, y_px))

        return landmarks_2d


def extract_hand_landmarks(frame):
    """
    단일 OpenCV 프레임에서 손 랜드마크를 추출.
    Mediapipe Hands 이용 (Solutions API 예시).
    반환 형태: [(x, y), (x, y), ...] (여러 손이면 전부 합침) or None
    """
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if not results.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape
        all_hand_lms = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                all_hand_lms.append((x_px, y_px))

        return all_hand_lms if all_hand_lms else None


def draw_landmarks(frame, landmarks, is_face=True):
    """
    얼굴 혹은 손 랜드마크를 이미지 위에 시각화.
    landmarks: [(x, y), (x, y), ...] 형태 (픽셀 좌표)
    is_face: True이면 얼굴 랜드마크, False면 손 랜드마크(색상 구분용)
    """
    # 색상 예시
    color_face = (0, 255, 0)  # 초록
    color_hand = (255, 0, 0)  # 파랑

    color = color_face if is_face else color_hand

    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 2, color, -1)

    return frame