import cv2
import numpy as np
from face_recog_app.detection import extract_landmarks
from database.db_control import find_user_by_name, log_access
from face_recog_app.hand_gesture import gesture_detect

# 얼굴 인증을 위한 유사도 계산 함수 (유클리드 거리 기반)
def calculate_similarity(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return float('inf')  # 길이가 다르면 비교 불가
    
    # 유클리드 거리 계산
    diff = np.linalg.norm(np.array(landmarks1) - np.array(landmarks2), axis=1)
    
    # 평균 차이를 구하여 유사도 반환
    return np.mean(diff)

# 손동작 인증을 위한 이미지 전처리
'''def preprocess_image(frame):
    # 이미지 크기 조정
    resized = cv2.resize(frame, (640, 480))  # 원하는 크기로 조정
    
    # 밝기와 대비 조정
    alpha = 1.5  # 대비
    beta = 50    # 밝기
    adjusted = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)
    
    return adjusted'''

# 얼굴 인증 함수
def authenticate_face_and_gesture(name):
    # 데이터베이스에서 사용자 정보 가져오기
    user = find_user_by_name(name)
    if not user:
        return "해당 이름으로 저장된 얼굴 랜드마크가 없습니다.", None

    user_id, user_name, saved_landmarks, selected_gesture, role = user

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "웹캠을 열 수 없습니다.", None

    # 이미지 캡처
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "이미지를 캡처할 수 없습니다.", None
    
    # 전처리 적용
    # frame = preprocess_image(frame)

    # 얼굴 랜드마크 추출
    landmarks = extract_landmarks(frame)  # 이제 detection.py에서 호출
    if not landmarks:
        cap.release()
        return "얼굴 랜드마크를 추출할 수 없습니다.", frame
    
    # 유사도 계산 (저장된 랜드마크와 비교)
    similarity_results = []
    for saved_landmark in saved_landmarks:
        # 인증부분에서 캡처한 얼굴의 랜드마크를 상대적인 위치로 변환
        # 중심 랜드마크 선택 (예: 코 끝 부분, id=1로 가정)
        central_landmark_id = 1
        central_landmark_1 = saved_landmark[central_landmark_id]
        central_landmark_2 = landmarks[central_landmark_id]
        
        # 중심 기준으로 상대적인 값으로 변환 -> 얼굴의 위치가 달라도 제대로된 비교를 하기위함
        relative_landmarks_1 = [
            (x - central_landmark_1[0], y - central_landmark_1[1], z - central_landmark_1[2])
            for x, y, z in saved_landmark
        ]
        relative_landmarks_2 = [
            (x - central_landmark_2[0], y - central_landmark_2[1], z - central_landmark_2[2])
            for x, y, z in landmarks
        ]
        similarity = calculate_similarity(relative_landmarks_1, relative_landmarks_2)
        #similarity = calculate_similarity(saved_landmark, relative_landmarks)
        similarity_results.append(similarity)

    # 유사도가 임계값 미만인 경우 카운트
    successful_matches = sum(1 for similarity in similarity_results if similarity < 0.03)
    
    # 절반 이상 유사도가 120 미만이면 인증 성공
    if successful_matches >= len(saved_landmarks) / 2:
        log_access(user_id, "success", "얼굴 인증 성공")
    
        # 손동작 제스처 추출
        gesture = gesture_detect(frame)  # 손동작 제스처 추출

        # 저장된 제스처와 비교
        if gesture == selected_gesture:
            log_access(user_id, "success", "얼굴 및 손동작 인증 성공")
            cap.release()
            return f"인증 성공: {name}", frame
        else:
            log_access(user_id, "failure", "손동작 인증 실패")
            cap.release()
            return f"손동작 인증 실패: {gesture} - 저장된 {selected_gesture}제스처와 일치하지 않음.", frame
    else:
        log_access(user_id, "failure", "얼굴 유사도 낮음")
        cap.release()
        return f"인증 실패: 얼굴 유사도가 낮습니다. 유사도 : {similarity_results}", frame