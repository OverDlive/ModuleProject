import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh

# 랜드마크 좌표 추출 및 저장 함수
def extract_landmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    landmarks_array = []  # 랜드마크 좌표 저장 리스트

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks_array.append((landmark.x, landmark.y, landmark.z))
    return landmarks_array
