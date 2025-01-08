import streamlit as st
from face_recog_app.detection import capture_images_from_webcam, draw_landmarks, extract_landmarks
from face_recog_app.authentication import extract_landmarks, extract_gesture_landmarks
from ui.app import run_app
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
from database.db_control import (
    initialize_database,
    add_user,
    get_all_users,
    find_user_by_name,
    delete_user_by_name,
    log_access
)
import numpy as np
import cv2

# 얼굴 인증을 위한 유사도 계산 함수
def calculate_similarity(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return float('inf')  # 길이가 다르면 비교 불가
    
    diff = np.linalg.norm(np.array(landmarks1) - np.array(landmarks2), axis=1)
    return np.mean(diff)

# 손동작 제스처 유사도 계산 함수 (간단한 예시)
def calculate_gesture_similarity(gestures1, gestures2):
    if len(gestures1) != len(gestures2):
        return float('inf')  # 제스처 수가 다르면 비교 불가
    
    total_similarity = 0
    for g1, g2 in zip(gestures1, gestures2):
        if len(g1) != len(g2):
            return float('inf')
        diff = np.linalg.norm(np.array(g1) - np.array(g2), axis=1)
        total_similarity += np.mean(diff)
    
    return total_similarity / len(gestures1)

# Streamlit 앱 실행
def run_app():
    st.title("얼굴 인식 및 손동작 제스처 관리 시스템")
    st.sidebar.title("메뉴")
    menu = st.sidebar.selectbox("선택하세요", ["캡처 및 저장", "사용자 조회", "사용자 삭제", "얼굴 및 손동작 인증"])

    # 데이터베이스 초기화 (최초 한 번)
    if "db_initialized" not in st.session_state:
        initialize_database()
        st.session_state.db_initialized = True

    if menu == "캡처 및 저장":
        st.header("캡처 및 저장")
        name = st.text_input("사용자 이름 입력")
        capture_count = st.number_input("캡처할 이미지 수", min_value=1, max_value=10, value=6, step=1)

        # 세션 상태 초기화
        if "all_landmarks" not in st.session_state:
            # 여러 이미지 => List[List[(x, y, z), ...]]
            st.session_state.all_landmarks = []
        if "all_gesture_landmarks" not in st.session_state:
            # 여러 이미지의 손동작 제스처 => List[List[List[(x, y, z), ...]]]
            st.session_state.all_gesture_landmarks = []
        if "capture_count_so_far" not in st.session_state:
            st.session_state.capture_count_so_far = 0

        if st.session_state.capture_count_so_far < capture_count:
            button_text = f"이미지 {st.session_state.capture_count_so_far + 1} 촬영"
            if st.button(button_text):
                with st.spinner(f"이미지 {st.session_state.capture_count_so_far + 1} 촬영 중..."):
                    images = capture_images_from_webcam(target_count=1, key=f"capture_{st.session_state.capture_count_so_far}")
                if images:
                    landmarks = extract_landmarks(images[0])  # 얼굴 랜드마크 추출
                    gestures = extract_gesture_landmarks(images[0])  # 손동작 제스처 추출
                    if landmarks:
                        st.session_state.all_landmarks.append(landmarks)
                        st.session_state.all_gesture_landmarks.append(gestures)
                        st.session_state.capture_count_so_far += 1
                        st.success(f"이미지 {st.session_state.capture_count_so_far}번째 캡처 성공.")
                    else:
                        st.error("얼굴 랜드마크를 추출할 수 없습니다.")
                else:
                    st.error("이미지 캡처에 실패했습니다.")

        if st.session_state.capture_count_so_far == capture_count:
            if st.button("데이터베이스에 저장"):
                if not name.strip():
                    st.error("사용자 이름을 입력하세요.")
                else:
                    user_id = add_user(name, st.session_state.all_landmarks, st.session_state.all_gesture_landmarks)
                    st.success(f"{name}의 얼굴 및 손동작 데이터가 성공적으로 저장되었습니다. (User ID: {user_id})")

                    # 저장 후 세션 상태 초기화
                    st.session_state.all_landmarks = []
                    st.session_state.all_gesture_landmarks = []
                    st.session_state.capture_count_so_far = 0

    elif menu == "사용자 조회":
        st.header("사용자 조회")
        users = get_all_users()
        if users:
            st.write("저장된 사용자 목록:")
            for user in users:
                user_id, user_name, face_data, gesture_data, role = user
                gesture_count = len(gesture_data) if gesture_data else 0
                st.write(f"- **ID**: {user_id}, **이름**: {user_name}, **역할**: {role}, 얼굴 랜드마크 개수: {len(face_data) if face_data else 0}, 손동작 제스처 개수: {gesture_count}")
        else:
            st.info("등록된 사용자가 없습니다.")

    elif menu == "사용자 삭제":
        st.header("사용자 삭제")
        name = st.text_input("삭제할 사용자 이름 입력")
        if st.button("삭제"):
            user = find_user_by_name(name)
            if user:
                delete_user_by_name(name)
                st.success(f"{name}의 데이터가 삭제되었습니다.")
            else:
                st.error(f"{name} 사용자를 찾을 수 없습니다.")

    elif menu == "얼굴 및 손동작 인증":
        st.header("얼굴 및 손동작 인증")
        name = st.text_input("인증할 사용자 이름 입력")
        if st.button("인증 시작"):
            if not name.strip():
                st.error("사용자 이름을 입력하세요.")
            else:
                user = find_user_by_name(name)
                if not user:
                    st.error("해당 이름으로 저장된 얼굴 랜드마크 또는 손동작 제스처가 없습니다.")
                else:
                    user_id, user_name, saved_landmarks, saved_gestures, role = user
                    st.info("얼굴 및 손동작을 캡처하고 있습니다...")

                    # 웹캠 열기
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("웹캠을 열 수 없습니다.")
                        return

                    # 이미지 캡처
                    ret, frame = cap.read()
                    if not ret:
                        st.error("이미지를 캡처할 수 없습니다.")
                        cap.release()
                        return

                    # 얼굴 랜드마크 추출
                    landmarks = extract_landmarks(frame)
                    if not landmarks:
                        st.error("얼굴 랜드마크를 추출할 수 없습니다.")
                        cap.release()
                        return

                    # 손동작 제스처 추출
                    gestures = extract_gesture_landmarks(frame)

                    # 유사도 계산
                    similarity = calculate_similarity(saved_landmarks, landmarks)
                    gesture_similarity = calculate_gesture_similarity(saved_gestures, gestures) if saved_gestures and gestures else float('inf')
                    
                    st.write(f"얼굴 유사도: {similarity}")
                    st.write(f"손동작 유사도: {gesture_similarity}")

                    # 인증 결과
                    if similarity < 120 and gesture_similarity < 120:  # 임계값은 조정 필요
                        st.success(f"인증 성공: {name}")
                        log_access(user_id, "success", "얼굴 및 손동작 인증 성공")
                    else:
                        st.error("인증 실패: 얼굴 또는 손동작 유사도가 낮습니다.")
                        log_access(user_id, "failure", "얼굴 또는 손동작 유사도 낮음")

                    # 웹캠 자원 해제
                    cap.release()
if __name__ == "__main__":
    run_app()