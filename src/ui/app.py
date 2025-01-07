import streamlit as st
from face_recog_app.detection import capture_images_from_webcam
from face_recog_app.authentication import extract_landmarks
from database.db_control import initialize_database, add_user

# Streamlit 앱 실행
def run_app():
    st.title("얼굴 인식 데이터 관리 시스템")
    st.sidebar.title("메뉴")
    menu = st.sidebar.selectbox("선택하세요", ["캡처 및 저장", "사용자 조회", "사용자 삭제"])

    # 데이터베이스 초기화
    if "db_initialized" not in st.session_state:
        initialize_database()
        st.session_state.db_initialized = True

    if menu == "캡처 및 저장":
        st.header("캡처 및 저장")
        name = st.text_input("사용자 이름 입력")
        capture_count = st.number_input("캡처할 이미지 수", min_value=1, max_value=10, value=6, step=1)  # 캡처할 이미지 수 입력

        if st.button("캡처 시작"):
            # 웹캡처 진행
            captured_images = []
            all_landmarks = []
            capture_attempts = 0

            while len(captured_images) < capture_count:
                st.write(f"현재 캡처된 이미지 수: {len(captured_images)} / {capture_count}")
                images = capture_images_from_webcam(target_count=capture_count - len(captured_images))

                if images:
                    for img in images:
                        landmarks = None
                        while landmarks is None:  # 랜드마크 추출이 실패할 때까지 반복
                            landmarks = extract_landmarks(img)
                            if landmarks is None:
                                st.warning("얼굴 랜드마크를 추출할 수 없습니다. 다시 시도합니다...")
                        all_landmarks.append(landmarks)
                        captured_images.append(img)
                        st.success(f"이미지 {len(captured_images)}가 성공적으로 캡처되었습니다.")

                capture_attempts += 1
                if len(captured_images) < capture_count:
                    st.write(f"캡처 시도 횟수: {capture_attempts}")

            if all_landmarks:
                # 여러 이미지에서 추출된 랜드마크를 저장
                for landmarks in all_landmarks:
                    add_user(name, landmarks)
                st.success(f"{name}의 얼굴 데이터가 저장되었습니다.")
            else:
                st.error("얼굴 랜드마크를 추출할 수 없습니다.")
    elif menu == "사용자 조회":
        # 사용자 조회 로직
        pass
    elif menu == "사용자 삭제":
        # 사용자 삭제 로직
        pass