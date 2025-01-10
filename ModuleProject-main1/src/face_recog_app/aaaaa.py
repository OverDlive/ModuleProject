import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os
import cv2
import streamlit as st
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')
joblib_path = os.path.join(script_dir, 'sign_language_model.joblib')

def convert_landmarks_to_dataframe(detection_result):
    all_hands_data = []

    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hand_data = []

        for idx, landmark in enumerate(hand_landmarks):
            hand_data.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        hand_dict = {
            'hand_index': hand_idx,
            **{f'landmark_{i}_{coord}': val
               for i in range(21)
               for coord, val in zip(['x', 'y', 'z'], hand_data[i*3:(i+1)*3])}
        }
        all_hands_data.append(hand_dict)

    df = pd.DataFrame(all_hands_data)
    
    if 'hand_index' in df.columns:
        df.drop('hand_index', axis=1, inplace=True)

    return df

def resize_image(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized

def is_hand_in_center(landmarks_df, image_shape, threshold=0.2):
    h, w, _ = image_shape
    center_x, center_y = w / 2, h / 2

    for i in range(0, len(landmarks_df.columns), 3):
        x = landmarks_df.iloc[0, i]
        y = landmarks_df.iloc[0, i + 1]
        if abs(x - center_x / w) < threshold and abs(y - center_y / h) < threshold:
            return True
    return False

def predict_sign(frame) -> str:
    frame = resize_image(frame, scale_percent=150)  # 150%로 크기 조정

    model_file = open(model_path, "rb")
    model_data = model_file.read()
    model_file.close()

    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    landmarks_df = convert_landmarks_to_dataframe(detection_result)

    if landmarks_df.empty or not is_hand_in_center(landmarks_df, frame.shape):
        return "NO_HAND_DETECTED"

    model = joblib.load(joblib_path)
    y_pred = model.predict(landmarks_df)
    result = chr(y_pred[0] + ord('A'))

    return result

def finger_landmark_visualization(image_dir: str):
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    image1 = mp.Image.create_from_file(image_dir)
    detection_result = detector.detect(image1)
    landmarks_df = convert_landmarks_to_dataframe(detection_result)

    image2 = cv2.imread(image_dir)
    landmark_index = dict()
    tmp = [[0, 1, 5, 9, 13, 17],
           [2, 3, 4],
           [6, 7, 8],
           [10, 11, 12],
           [14, 15, 16],
           [18, 19, 20]]

    for i, v in enumerate(tmp):
        for j in v:
            landmark_index[j] = i
    
    color = [(48, 48, 255), (180, 229, 255), (128, 64, 128),
             (0, 204, 255), (48, 255, 48), (192, 101, 21)]

    h, w, _ = image2.shape

    for i in range(0, len(landmarks_df.columns), 3):
        group = landmarks_df.iloc[0, i:i+3].tolist()
        index = i // 3
        j = landmark_index[index]
        lx = group[0]
        ly = group[1]

        px = int(lx * w)
        py = int(ly * h)

        cv2.circle(image2, (px, py), 2, color[j], -1)

    st.image(image2)
