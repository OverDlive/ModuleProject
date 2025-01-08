import streamlit as st
from face_recog_app.detection import (
    capture_images_from_webcam,
    draw_landmarks,
    extract_landmarks,
    extract_hand_landmarks
)
from face_recog_app.authentication import (
    authenticate_face_and_gesture,  # 얼굴+손동작 비교 로직
    calculate_similarity
)
from database.db_control import (
    initialize_database,
    add_user,            # add_user(name, face_data, gesture_data)
    get_all_users,
    find_user_by_name,
    delete_user_by_name,
    log_access
)
import numpy as np
import cv2

def run_app():
    st.markdown("<h1>얼굴 및 손동작 제스처 관리 시스템</h1>", unsafe_allow_html=True)

    st.sidebar.title("메뉴")
    menu = st.sidebar.selectbox(
        "선택하세요",
        ["캡처 및 저장", "사용자 조회", "사용자 삭제", "얼굴 및 손동작 인증"]
    )

    # 데이터베이스 초기화 (최초 한 번)
    if "db_initialized" not in st.session_state:
        initialize_database()
        st.session_state.db_initialized = True

    # ─────────────────────────────────────────
    # 1) "캡처 및 저장": 얼굴 + 손이 나온 사진 한 번 캡처
    # ─────────────────────────────────────────
    if menu == "캡처 및 저장":
        st.header("캡처 및 저장")
        name = st.text_input("사용자 이름 입력")

        if st.button("얼굴+손 캡처"):
            with st.spinner("이미지 캡처 중..."):
                images = capture_images_from_webcam(target_count=1, key="face_and_hand_capture")
            
            if images:
                # 하나의 이미지에서 얼굴 랜드마크 추출
                face_landmarks = extract_landmarks(images[0])
                # 같은 이미지에서 손 랜드마크 추출
                hand_landmarks = extract_hand_landmarks(images[0])

                if face_landmarks and hand_landmarks:
                    # 얼굴 및 손 랜드마크 시각화
                    annotated_frame = images[0].copy()
                    annotated_frame = draw_landmarks(annotated_frame, face_landmarks, is_face=True)
                    annotated_frame = draw_landmarks(annotated_frame, hand_landmarks, is_face=False)  # is_face=False 가정

                    st.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption="캡처된 얼굴+손 이미지",
                        use_container_width=True
                    )

                    # DB 저장 버튼
                    if st.button("데이터베이스에 저장"):
                        if not name.strip():
                            st.error("사용자 이름을 입력하세요.")
                        else:
                            user_id = add_user(name, face_landmarks, hand_landmarks)
                            st.success(f"사용자 '{name}'의 얼굴/손 데이터가 저장되었습니다. (User ID: {user_id})")
                else:
                    st.error("얼굴 또는 손동작 랜드마크를 추출할 수 없습니다.")
            else:
                st.error("이미지 캡처에 실패했습니다.")

    # ─────────────────────────────────────────
    # 2) "사용자 조회"
    # ─────────────────────────────────────────
    elif menu == "사용자 조회":
        st.header("사용자 조회")
        users = get_all_users()
        if users:
            st.write("저장된 사용자 목록:")
            for user in users:
                # DB에서 face_data, gesture_data를 모두 꺼낸다고 가정
                user_id, user_name, face_data, gesture_data, role = user  
                fcount = len(face_data) if face_data else 0
                gcount = len(gesture_data) if gesture_data else 0

                st.write(f"- **ID**: {user_id}, **이름**: {user_name}, "
                         f"**역할**: {role}, 얼굴 데이터 수: {fcount}, 손 데이터 수: {gcount}")
        else:
            st.info("등록된 사용자가 없습니다.")

    # ─────────────────────────────────────────
    # 3) "사용자 삭제"
    # ─────────────────────────────────────────
    elif menu == "사용자 삭제":
        st.header("사용자 삭제")
        name = st.text_input("삭제할 사용자 이름 입력")
        if st.button("삭제"):
            user = find_user_by_name(name)
            if user:
                delete_user_by_name(name)
                st.success(f"'{name}'의 데이터가 삭제되었습니다.")
            else:
                st.error(f"'{name}' 사용자를 찾을 수 없습니다.")

    # ─────────────────────────────────────────
    # 4) "얼굴 및 손동작 인증"
    # ─────────────────────────────────────────
    elif menu == "얼굴 및 손동작 인증":
        st.header("얼굴 및 손동작 인증")
        name = st.text_input("인증할 사용자 이름 입력")

        if st.button("인증 시작"):
            if not name.strip():
                st.error("사용자 이름을 입력하세요.")
            else:
                # 얼굴 및 손동작 인증: 단 한 번의 캡처
                result, frame = authenticate_face_and_gesture(name)
                # authenticate_face_and_gesture(name) 내부에서:
                #  - capture_images_from_webcam(1장)
                #  - extract_landmarks(얼굴), extract_hand_landmarks(손)
                #  - DB의 face_data, gesture_data와 비교
                #  - 결과 문자열, 사용된 frame 반환

                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="인증에 사용된 이미지", use_container_width=True)

                if "인증 성공" in result:
                    st.success(result)
                else:
                    st.error(result)