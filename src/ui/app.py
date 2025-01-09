import streamlit as st
from face_recog_app.detection import capture_images_from_webcam, draw_landmarks, extract_landmarks
from face_recog_app.authentication import authenticate_face_and_gesture
from database.db_control import (
    initialize_database,
    add_user,
    get_all_users,
    find_user_by_name,
    delete_user_by_name,
)
import numpy as np
import cv2

# Streamlit 앱 실행
def run_app():
    st.title('얼굴 인식 및 손동작 제스처 관리 시스템')

    st.sidebar.title("메뉴")
    menu = st.sidebar.selectbox("선택하세요", ["캡처 및 저장", "사용자 조회", "사용자 삭제", "얼굴 및 손동작 인증"])

    # 데이터베이스 초기화 (최초 한 번)
    if "db_initialized" not in st.session_state:
        initialize_database()
        st.session_state.db_initialized = True

    # 선택지 목록
    #gesture_options = ["Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
    # 선택 박스 생성
    #st.sidebar.title('인증 제스처')
    #selected_option = st.sidebar.selectbox("선택", gesture_options)
    # 선택된 옵션 표시
    #st.write(f"선택된 옵션: {selected_option}")

    # 캠처 및 저장장
    if menu == "캡처 및 저장":
        #st.header("캡처 및 저장 - 얼굴을 중앙에 위치해 주세요")
        st.markdown(
    '<h3>캡처 및 저장 - 얼굴을 <span style="color:red; font-weight:bold;">중앙에 위치</span>해 주세요</h3>',
    unsafe_allow_html=True
)
        name = st.text_input("사용자 이름 입력")
        # 선택지 목록
        gesture_options = ["Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
        # 선택 박스 생성
        selected_option = st.selectbox('선택', gesture_options)
        capture_count = st.number_input("캡처할 이미지 수", min_value=1, max_value=10, value=6, step=1)

        # 세션 상태 초기화
        if "all_landmarks" not in st.session_state:
            # 여러 이미지 => List[List[(x, y, z), ...]]
            st.session_state.all_landmarks = []
        if "capture_count_so_far" not in st.session_state:
            st.session_state.capture_count_so_far = 0

        if st.session_state.capture_count_so_far < capture_count:
            button_text = f"이미지 {st.session_state.capture_count_so_far + 1} 촬영"
            if st.button(button_text):
                with st.spinner(f"이미지 {st.session_state.capture_count_so_far + 1} 촬영 중..."):
                    images = capture_images_from_webcam(target_count=1, key=f"capture_{st.session_state.capture_count_so_far}")
                if images:
                    landmarks = extract_landmarks(images[0])  # 얼굴 랜드마크 추출
                    if landmarks:
                        st.session_state.all_landmarks.append(landmarks)
                        # 랜드마크를 이미지에 표시
                        annotated_image = draw_landmarks(images[0].copy(), landmarks)
                        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption=f"캡처 {st.session_state.capture_count_so_far}", use_container_width=True)
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
                    user_id = add_user(name, st.session_state.all_landmarks, selected_option)
                    st.success(f"{name}의 얼굴 데이터가 성공적으로 저장되었습니다. (User ID: {user_id})")

                    # 저장 후 세션 상태 초기화
                    st.session_state.all_landmarks = []
                    st.session_state.capture_count_so_far = 0

    # 사용자 조회
    elif menu == "사용자 조회":
        st.header("사용자 조회")
        users = get_all_users()
        if users:
            st.write("저장된 사용자 목록:")
            for user in users:
                user_id, user_name, face_data, gesture_options, role = user
                st.write(f"- **ID**: {user_id}, **이름**: {user_name}, **제스처**: {gesture_options}, **역할**: {role}, 얼굴 랜드마크 개수: {len(face_data)}")
        else:
            st.info("등록된 사용자가 없습니다.")

    # 사용자 삭제제
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

    # 얼굴 및 손동작 인증증
    elif menu == "얼굴 및 손동작 인증":
        st.header("얼굴 및 손동작 인증")
        name = st.text_input("인증할 사용자 이름 입력")
        if st.button("인증 시작"):
            if not name.strip():
                st.error("사용자 이름을 입력하세요.")
            else:
                # 얼굴 및 손동작 인증 함수 호출
                result, frame = authenticate_face_and_gesture(name)

                # 캡처된 이미지 출력
                if frame is not None:
                    # OpenCV 이미지를 RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="캡처된 이미지", use_container_width=True)

                # 인증 결과 출력
                if "인증 성공" in result:
                    st.success(result)
                else:
                    st.error(result)