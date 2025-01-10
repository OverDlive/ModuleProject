import tensorflow as tf

# HDF5 파일 로드
model = tf.keras.models.load_model('C:/Users/user/Desktop/python/ModuleProject-main1/src/face_recog_app/model.keras')

# SavedModel 형식으로 저장
model.save('C:/Users/user/Desktop/python/ModuleProject-main1/src/face_recog_app/saved_model_format')

