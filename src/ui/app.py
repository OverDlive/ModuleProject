import streamlit as st
from face_recog_app.detection import capture_images_from_webcam
from face_recog_app.authentication import extract_landmarks
from database.db_control import (
    initialize_database,
    add_user,
    get_all_users,
    find_user_by_name,
    delete_user_by_name
)

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

        # 이미지 저장용 리스트
        captured_images = []
        all_landmarks = []

        # 캡처 횟수 관리
        if "capture_count_so_far" not in st.session_state:
            st.session_state.capture_count_so_far = 0  # 시작 시 클릭된 횟수 초기화

        # 캡처된 이미지 수가 목표치에 도달할 때까지 반복
        if st.session_state.capture_count_so_far < capture_count:
            if st.button(f"캡처 이미지 {st.session_state.capture_count_so_far + 1} 촬영"):
                # 이미지를 캡처하고 랜드마크 추출
                images = capture_images_from_webcam(target_count=1, key="capture_button")
                if images:
                    landmarks = extract_landmarks(images[0])  # 첫 번째 이미지에서 랜드마크 추출
                    if landmarks:
                        captured_images.append(images[0])
                        all_landmarks.append(landmarks)
                        st.success(f"이미지 {st.session_state.capture_count_so_far + 1}가 성공적으로 캡처되었습니다.")
                        st.session_state.capture_count_so_far += 1  # 클릭 횟수 증가
                    else:
                        st.error(f"이미지 {st.session_state.capture_count_so_far + 1}에서 얼굴 랜드마크를 추출할 수 없습니다.")
                else:
                    st.error(f"이미지 {st.session_state.capture_count_so_far + 1} 캡처에 실패했습니다.")

        # 모든 캡처가 완료되면 랜드마크 저장
        if st.session_state.capture_count_so_far == capture_count:
            if all_landmarks:
                for landmarks in all_landmarks:
                    add_user(name, landmarks)
                st.success(f"{name}의 얼굴 데이터가 저장되었습니다.")
            else:
                st.error("캡처 및 랜드마크 추출에 실패했습니다.")
    
    elif menu == "사용자 조회":
        st.header("사용자 조회")
        users = get_all_users()
        if users:
            st.write("저장된 사용자 목록:")
            for user in users:
                st.write(f"ID: {user[0]}, 이름: {user[1]}")
        else:
            st.write("등록된 사용자가 없습니다.")

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