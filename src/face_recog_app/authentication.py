import cv2
import numpy as np
from face_recog_app.detection import extract_landmarks
from database.db_control import find_user_by_name, log_access
from face_recog_app.hand_gesture import predict_sign, convert_landmarks_to_dataframe

# 얼굴 인증을 위한 유사도 계산 함수 (유클리드 거리 기반)
def calculate_similarity(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return float('inf')  # 길이가 다르면 비교 불가
    
    # 유클리드 거리 계산
    diff = np.linalg.norm(np.array(landmarks1) - np.array(landmarks2), axis=1)
    
    # 평균 차이를 구하여 유사도 반환
    return np.mean(diff)

# 얼굴 인증 함수
def authenticate_face_and_gesture(name, today_alphabet):
    # 데이터베이스에서 사용자 정보 가져오기
    user = find_user_by_name(name)
    if not user:
        return "해당 이름으로 저장된 얼굴 랜드마크가 없습니다.", None

    user_id, user_name, saved_landmarks, role = user

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "웹캠을 열 수 없습니다.", None

    # 이미지 캡처
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "이미지를 캡처할 수 없습니다.", None

    # 얼굴 랜드마크 추출
    landmarks = extract_landmarks(frame)  # 이제 detection.py에서 호출
    if not landmarks:
        cap.release()
        return "얼굴 랜드마크를 추출할 수 없습니다.", frame

    # 유사도 계산 (저장된 랜드마크와 비교)
    similarity_results = []
    for saved_landmark in saved_landmarks:
        similarity = calculate_similarity(saved_landmark, landmarks)
        similarity_results.append(similarity)

    # 유사도가 120 미만인 경우 카운트
    successful_matches = sum(1 for similarity in similarity_results if similarity < 120)
    
    # 절반 이상 유사도가 120 미만이면 인증 성공
    if successful_matches >= len(saved_landmarks) / 2:
        log_access(user_id, "success", "얼굴 인증 성공")
    
        # 손동작 제스처 추출
        gesture = predict_sign(frame)  # 손동작 제스처 추출 및 알파벳 예측

        # 손동작이 오늘의 알파벳과 일치하는지 확인
        if gesture == today_alphabet:
            log_access(user_id, "success", "얼굴 및 손동작 인증 성공")
            cap.release()
            return f"인증 성공: {name}", frame
        else:
            log_access(user_id, "failure", "손동작 인증 실패")
            cap.release()
            return "손동작 인증 실패: 오늘의 알파벳과 일치하지 않음.", frame
    else:
        log_access(user_id, "failure", "얼굴 유사도 낮음")
        cap.release()
        return "인증 실패: 얼굴 유사도가 낮습니다.", frame
