import mediapipe as mp
import numpy as np
import cv2
import tensorflow as tf

import mediapipe as mp
import numpy as np
import cv2
import tensorflow as tf

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 얼굴 랜드마크 추출 함수
def extract_landmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    landmarks_array = []  # 얼굴 랜드마크 저장 리스트

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks_array.append((landmark.x, landmark.y, landmark.z))
    return landmarks_array

# 손동작 제스처 추출 클래스
class HandGestureRecognition:
    def __init__(self, model_path, actions, seq_length=30):
        self.actions = actions
        self.seq_length = seq_length
        # Keras 모델 로드
        self.model = tf.keras.models.load_model(model_path)

        # MediaPipe Hands 초기화
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 초기화 변수
        self.seq = []
        self.action_seq = []

    def preprocess_frame(self, frame):
        """이미지를 RGB로 변환하고, 좌우 반전하여 전처리하는 함수"""
        flipped_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def extract_gesture_landmarks(self, image):
        """손동작의 랜드마크를 추출하는 함수"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        gesture_landmarks = []  # 손동작 제스처 랜드마크 저장 리스트

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand = []
                for landmark in hand_landmarks.landmark:
                    single_hand.append((landmark.x, landmark.y, landmark.z))
                gesture_landmarks.append(single_hand)
        return gesture_landmarks

    def extract_joint_angles(self, hand_landmarks):
    
        joint = np.zeros((21, 4))  # 21개의 관절, 각 관절에 x, y, z, visibility 값 설정
        for j, lm in enumerate(hand_landmarks):
            joint[j] = [lm.x, lm.y, lm.z, 1.0]  # lm.x, lm.y, lm.z로 접근

        # Parent joints와 Child joints 계산
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joints
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joints
        v = v2 - v1  # 벡터 차이
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # 벡터 정규화

        # 각도 계산
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
        angle = np.degrees(angle)
        return np.concatenate([joint.flatten(), angle])

    def predict_action(self, input_data):
        """손동작을 예측하는 함수"""
        input_data = np.expand_dims(np.array(input_data, dtype=np.float32), axis=0)
        y_pred = self.model.predict(input_data)  # 모델 예측
        i_pred = int(np.argmax(y_pred))  # 예측된 액션 인덱스
        conf = y_pred[0][i_pred]  # 신뢰도 값
        return i_pred, conf

    def draw_landmarks(self, frame, hand_landmarks):
        """손 랜드마크를 이미지에 그리는 함수"""
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    def run(self):
        """웹캠을 통해 실시간으로 손동작을 인식하는 함수"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img_rgb = self.preprocess_frame(img)
            landmarks = self.extract_gesture_landmarks(img)

            # 손동작 랜드마크 추출 및 예측
            if landmarks:
                for hand in landmarks:
                    d = self.extract_joint_angles(hand)
                    self.seq.append(d)

                    if len(self.seq) < self.seq_length:
                        continue

                    input_data = self.seq[-self.seq_length:]
                    i_pred, conf = self.predict_action(input_data)

                    if conf < 0.3:
                        continue

                    action = self.actions[i_pred]
                    self.action_seq.append(action)

                    if len(self.action_seq) < 3:
                        continue

                    # 일관된 동작이 있는 경우
                    this_action = '?' if self.action_seq[-1] != self.action_seq[-2] or self.action_seq[-1] != self.action_seq[-3] else action

                    # 동작 결과를 화면에 표시
                    cv2.putText(img, f'{this_action.upper()}', 
                                org=(10, 50), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=1, 
                                color=(255, 255, 255), 
                                thickness=2)

            # 결과 화면에 표시
            cv2.imshow('Hand Gesture Recognition', img)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    actions = ['good', 'five', 'fit', 'V', 'piece']  # 사용할 손동작 예시
    model_path = r'C:\Users\user\Desktop\python\ModuleProject-main1\src\face_recog_app\model.keras'  # 모델 파일 경로
    recognizer = HandGestureRecognition(model_path, actions)
    recognizer.run()




