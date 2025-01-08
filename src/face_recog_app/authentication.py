import cv2
import numpy as np
from face_recog_app.detection import extract_landmarks
from database.db_control import find_user_by_name, log_access
from face_recog_app.hand_gesture import predict_sign, convert_landmarks_to_dataframe
import streamlit as st  # Streamlit을 사용하여 디버그 메시지 출력

# 얼굴 인증을 위한 유사도 계산 함수 (유클리드 거리 기반)
def calculate_similarity(landmarks1, landmarks2):
    """
    두 랜드마크 세트 간의 유사도를 계산합니다.

    Args:
        landmarks1 (List[Tuple[float, float, float]]): 첫 번째 랜드마크 세트.
        landmarks2 (List[Tuple[float, float, float]]): 두 번째 랜드마크 세트.

    Returns:
        float: 유사도 값 (유클리드 거리의 평균).
    """
    if len(landmarks1) != len(landmarks2):
        return float('inf')  # 길이가 다르면 비교 불가
    
    # 유클리드 거리 계산
    diff = np.linalg.norm(np.array(landmarks1) - np.array(landmarks2), axis=1)
    
    # 평균 차이를 구하여 유사도 반환
    return np.mean(diff)

# 얼굴 및 손동작 인증 함수
def authenticate_face_and_gesture(name, today_alphabet=None):
    """
    주어진 사용자 이름을 기반으로 얼굴 및 손동작 인증을 수행합니다.

    Args:
        name (str): 인증할 사용자 이름.
        today_alphabet (str, optional): 오늘의 알파벳. 기본값은 None.

    Returns:
        tuple: 인증 결과 메시지와 인증에 사용된 이미지 프레임.
    """
    st.write(f"인증 시도 중: 사용자 이름 = {name}, 오늘의 알파벳 = {today_alphabet}")
    
    # 데이터베이스에서 사용자 정보 가져오기
    user = find_user_by_name(name)
    if not user:
        st.error("해당 이름으로 저장된 얼굴 랜드마크가 없습니다.")
        return "해당 이름으로 저장된 얼굴 랜드마크가 없습니다.", None

    # 사용자 정보 언패킹 (5개 값)
    user_id, user_name, saved_face_landmarks, saved_gesture_landmarks, role = user
    st.write(f"사용자 정보: ID = {user_id}, 이름 = {user_name}, 역할 = {role}")

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return "웹캠을 열 수 없습니다.", None

    # 이미지 캡처
    ret, frame = cap.read()
    st.write(f"이미지 캡처 성공 여부: {ret}")
    if not ret:
        cap.release()
        st.error("이미지를 캡처할 수 없습니다.")
        return "이미지를 캡처할 수 없습니다.", None

    # 얼굴 랜드마크 추출
    face_landmarks = extract_landmarks(frame)  # 얼굴 랜드마크 추출
    st.write(f"추출된 얼굴 랜드마크: {face_landmarks}")
    if not face_landmarks:
        cap.release()
        st.error("얼굴 랜드마크를 추출할 수 없습니다.")
        return "얼굴 랜드마크를 추출할 수 없습니다.", frame

    # 유사도 계산 (저장된 얼굴 랜드마크와 비교)
    similarity_results = []
    for idx, saved_landmark in enumerate(saved_face_landmarks):
        similarity = calculate_similarity(saved_landmark, face_landmarks)
        similarity_results.append(similarity)
        st.write(f"저장된 랜드마크 {idx+1}과의 유사도: {similarity}")

    # 유사도가 120 미만인 경우 카운트
    successful_matches = sum(1 for similarity in similarity_results if similarity < 120)
    st.write(f"유사도가 120 미만인 랜드마크 수: {successful_matches} / {len(saved_face_landmarks)}")

    # 절반 이상 유사도가 120 미만이면 얼굴 인증 성공
    if successful_matches >= len(saved_face_landmarks) / 2:
        log_access(user_id, "success", "얼굴 인증 성공")
        st.success("얼굴 인증 성공")

        # 손동작 제스처 추출
        gesture = predict_sign(frame)  # 손동작 제스처 추출 및 알파벳 예측
        st.write(f"예측된 손동작 제스처: {gesture}")

        # 손동작이 오늘의 알파벳과 일치하는지 확인
        if today_alphabet is not None and gesture == today_alphabet:
            log_access(user_id, "success", "얼굴 및 손동작 인증 성공")
            st.success(f"인증 성공: {name}")
            cap.release()
            return f"인증 성공: {name}", frame
        else:
            log_access(user_id, "failure", "손동작 인증 실패")
            cap.release()
            if today_alphabet is not None:
                st.error("손동작 인증 실패: 오늘의 알파벳과 일치하지 않음.")
                return "손동작 인증 실패: 오늘의 알파벳과 일치하지 않음.", frame
            else:
                st.error("손동작 인증 실패: 오늘의 알파벳이 제공되지 않았습니다.")
                return "손동작 인증 실패: 오늘의 알파벳이 제공되지 않았습니다.", frame
    else:
        log_access(user_id, "failure", "얼굴 유사도 낮음")
        cap.release()
        st.error("인증 실패: 얼굴 유사도가 낮습니다.")
        return "인증 실패: 얼굴 유사도가 낮습니다.", frame