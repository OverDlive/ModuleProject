import kagglehub
import os
import mediapipe as mp

def download_datset(datset_name):
    path = kagglehub.dataset_download(dataset_name)
    print(f'dataset downlaod {path}')
    return path

def get_image_folder_file(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg','.jpeg','.png')):
                image_files.append(os.path.join(root, file))
    return image_files

def access_files(folder_path):
    success_files = []
    for root, dirs, files in os.walk(folder_path):
        if target_folder_name in os.path.basename(root):
            for file in files:
                if file.lower().endswith(('.jpg','.jpeg','.png')):
                    file_path = os.path.join(root,file)
                    success_files.append(file_path)
                    print(f'로그인 이미지들{target_folder_name}')
        return success_files
# 데이터 셋 이름
dataset_name = "chaitanyakakade77/american-sign-language-dataset"
#데이터 셋 다운로드
dataset_path = download_datset(dataset_name)
# 다운로드된 데이터셋 경로
folder_path = 'C:\\Users\\user\\.cache\\kagglehub\\datasets\\chaitanyakakade77\\american-sign-language-dataset\\versions\\1\\ASL_Data'
target_folder_name = 'Y-samples'
image_files = get_image_folder_file(folder_path)
sccuess_files = get_image_folder_file(folder_path)
print(image_files)
