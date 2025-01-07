import cv2
import streamlit as st
from face_recog_app.authentication import extract_landmarks

# 웹캡처 함수
def capture_images_from_webcam(target_count=1, interval=2, key="capture_button"):
    cap = cv2.VideoCapture(0)  # 웹캡 열기

    if not cap.isOpened():
        st.error("웹캡을 열 수 없습니다.")
        return []

    st.write("웹캡에서 이미지를 캡처 중...")
    for _ in range(target_count):
        ret, frame = cap.read()
        if ret:
            original_frame = frame.copy()  # 원본 이미지를 보존하기 위해 복사
            
            # 랜드마크 추출
            landmarks = extract_landmarks(frame)  # 얼굴 랜드마크 추출
            
            if not landmarks:
                # landmarks가 None 이거나 빈 리스트인 경우 처리
                print("랜드마크를 찾지 못했습니다.")
            else:
                # 랜드마크 그리기
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 초록색 원으로 랜드마크 그리기
            
            st.image(frame, caption=f"캡처된 이미지", use_container_width=True)  # 랜드마크가 그려진 이미지 출력
        else:
            st.warning(f"이미지 캡처 실패. 다시 시도합니다...")

    cap.release()
    return [original_frame]  # 원본 이미지 하나만 반환