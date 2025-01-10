import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class HandGestureRecognition:
    def __init__(self, model_path, actions, seq_length=30):
        self.actions = actions
        self.seq_length = seq_length
        
        # 모델 로드
        self.model = load_model(model_path)

        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 변수 초기화
        self.seq = []
        self.action_seq = []

    def preprocess_frame(self, frame):
        """프레임을 전처리하는 함수 (이미지 뒤집기 및 RGB 변환)"""
        flipped_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def process_frame(self, frame):
        """프레임을 처리하여 손 랜드마크를 추출하는 함수"""
        result = self.hands.process(frame)
        return result

    def extract_joint_angles(self, hand_landmarks):
        """손 랜드마크로부터 각 관절의 각도를 추출하는 함수"""
        joint = np.zeros((21, 4))
        for j, lm in enumerate(hand_landmarks):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joints
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joints
        v = v2 - v1
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
        angle = np.degrees(angle)
        return np.concatenate([joint.flatten(), angle])

    def predict_action(self, input_data):
        """예측을 수행하는 함수"""
        input_data = np.expand_dims(np.array(input_data, dtype=np.float32), axis=0)
        y_pred = self.model.predict(input_data).squeeze()  # 예측
        i_pred = int(np.argmax(y_pred))  # 예측된 액션 인덱스
        conf = y_pred[i_pred]  # 예측된 액션의 신뢰도
        return i_pred, conf

    def draw_landmarks(self, frame, hand_landmarks):
        """손 랜드마크를 이미지에 그리는 함수"""
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def run(self):
        """웹캠을 통해 실시간으로 손동작을 인식하는 함수"""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            # 이미지 전처리: 뒤집기 및 RGB 변환
            img_rgb = self.preprocess_frame(img)

            # 손 인식
            result = self.process_frame(img_rgb)

            # 손 랜드마크가 있는 경우
            if result.multi_hand_landmarks:
                for res in result.multi_hand_landmarks:
                    # 관절 각도 추출
                    d = self.extract_joint_angles(res.landmark)
                    self.seq.append(d)

                    if len(self.seq) < self.seq_length:
                        continue

                    input_data = self.seq[-self.seq_length:]
                    i_pred, conf = self.predict_action(input_data)

                    if conf < 0.9:  # 신뢰도가 낮으면 무시
                        continue

                    action = self.actions[i_pred]
                    self.action_seq.append(action)

                    if len(self.action_seq) < 3:
                        continue

                    # 일관된 동작이 있는 경우
                    this_action = '?' if self.action_seq[-1] != self.action_seq[-2] or self.action_seq[-1] != self.action_seq[-3] else action

                    # 동작 결과를 화면에 표시
                    cv2.putText(img, f'{this_action.upper()}', 
                                org=(int(res.landmark[0].x * img.shape[1]), 
                                     int(res.landmark[0].y * img.shape[0] + 20)), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=1, 
                                color=(255, 255, 255), 
                                thickness=2)

                    # 손 랜드마크 그리기
                    self.draw_landmarks(img, res)

            # 결과 화면에 표시
            cv2.imshow('Hand Gesture Recognition', img)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 사용 예시
if __name__ == "__main__":
    actions = ['good', 'five', 'fit', 'V', 'piece']  # 제스처 리스트
    model_path = r'C:\Users\user\Desktop\python\ModuleProject-main1\src\face_recog_app\model.keras'  # 모델 경로

    recognizer = HandGestureRecognition(model_path, actions)
    recognizer.run()


