import streamlit as st
from face_recog_app.detection import capture_images_from_webcam
from face_recog_app.authentication import extract_landmarks
from database.db_control import initialize_database, add_user, get_all_users, find_user_by_name, delete_user_by_name

# Streamlit 앱 실행
def run_app():
    st.title("얼굴 인식 데이터 관리 시스템")
    st.sidebar.title("메뉴")
    menu = st.sidebar.selectbox("선택하세요", ["캡처 및 저장", "사용자 조회", "사용자 삭제"])

    # 데이터베이스 초기화
    initialize_database()

    if menu == "캡처 및 저장":
        st.header("캡처 및 저장")
        name = st.text_input("사용자 이름 입력")
        if st.button("캡처 시작"):
            images = capture_images_from_webcam(target_count=1)
            if images:
                landmarks = extract_landmarks(images[0])
                if landmarks:
                    add_user(name, landmarks)
                    st.success(f"{name}의 얼굴 데이터가 저장되었습니다.")
                else:
                    st.error("얼굴 랜드마크를 추출할 수 없습니다.")

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