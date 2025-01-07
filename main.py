import cv2
from project.ModuleProject.image_down import download_datset, get_image_folder_file
import mediapipe as mp
import numpy as np
import cv2
import math
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# mediapipe초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 랜드마크 dataframe으로 변환하는 함수
def convert_landmarks_to_dataframe(detection_result):
    all_hand = []
    for hand_idx, hand_landmarks in enumerate(detection_result.multi_hand_landmarks):
        hand_data = []
        # 각 손의 랜드마크 좌표 x,y,z추가
        for idx, landmark in enumerate(hand_landmarks.landmark):
            hand_data.extend([landmark.x, landmark.y, landmark.z])
        hand_dict = {
            'hand_index': hand_idx,            
            **{f'landmark_{i}_{coord}': val
               # 각 손 랜드마크 감지
               for i in range(21)
               for coord, val in zip(['x','y','z'], hand_data[i*3:(i+1)*3])}
        }
        all_hand.append(hand_dict)
        # 데이터프레임으로 변환
        hands = pd.DataFrame(all_hand)
        return hands

# 웹캠에서 랜드마크 추출 및 반환
def extract_landmark_image(frame):
    rgb = cv2.imread(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        landmarks_df = convert_landmarks_to_dataframe(result)
        return landmarks_df
    else:
        return None
# 엄지 손가락과 새끼 손가락의 랜드마크만 추출
def extract_thumb_success_landmarks(landmarks_df):
    thumb_landmarks = [f'landmark_{i}_x' for i in range(5)] + [f'landmark_{i}_y'for i in range(5)]
    pinkie_landmarks = [f'landmark_{i}_x' for i in range(16, 21)] + [f'landmark_{i}_y' for i in range(16, 21)]
    thumb_coords = landmarks_df[thumb_landmarks].values.flatten()
    pinkie_coords = landmarks_df[pinkie_landmarks].values.flatten()
    return np.concatenate([thumb_coords, pinkie_coords])
# 엄지손가락 새끼손가락 일정 이상시 펼친걸로 인식하도록 설정
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1
    x2, y2 = landmark2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
# 손 펼쳐져있는지 확인 - 거리계산
def hand_finger(landmarks_df):
    thumb_coords = [
        (landmarks_df['landmark_0_x'], landmarks_df['landmark_0_y']),
        (landmarks_df['landmark_4_x'], landmarks_df['landmark_4_y'])
    ]
    pinkie_coords = [
        (landmarks_df['landmark_16_x'], landmarks_df['landmark_16_y']),
        (landmarks_df['landmark_20_x'], landmarks_df['landmark_20_y']),  
    ]
    # 엄지손가락 새끼손가락 끝의 두 점 사이 거리 계산
    thumb_distance = calculate_distance(thumb_coords[0], thumb_coords[1])
    pinkie_distance = calculate_distance(pinkie_coords[0], pinkie_coords[1])
    # 손가락 펼침 기준
    thumb_threshold = 0.2
    pinkie_threshold = 0.3
    return thumb_distance > thumb_threshold and pinkie_distance > pinkie_threshold
#웹캠과 랜드마크 비교
def compare_landmarks(save_landmarks, webcam_landmarks):
    save_thumb_and_pinkie = extract_thumb_success_landmarks(save_landmarks)
    webcam_thumb_and_pinkie = extract_thumb_success_landmarks(webcam_landmarks)
    #코사인 유사도 계산산
    similarity = cosine_similarity([save_thumb_and_pinkie], [webcam_thumb_and_pinkie])
    return similarity[0][0]

# 실시간 웹캠 랜드마크 추출출
def webcam_tracking(save_landmarks_df):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        webcam_landmarks_df = extract_landmark_image(frame)
        if webcam_landmarks_df is not None:
            if hand_finger(webcam_landmarks_df):
                similary_score = compare_landmarks(save_landmarks_df, webcam_landmarks_df)
                if similary_score > 0.9:
                    cv2.putText(frame, '로그인 성공', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(frame, '로그인 실패', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame, 'hand not spread correctly', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('손 추적중', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()