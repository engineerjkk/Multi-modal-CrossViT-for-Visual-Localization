import torch
from tqdm import tqdm
from PIL import Image
# Day-to-night generators
cyclegan = torch.hub.load('mohwald/gandtr', 'cyclegan')

import cv2  # OpenCV 라이브러리 임포트
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as transforms
input_dir = '/home/cvnar/JKK/datasets/aachen/images/images_upright/db/'
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
output_dir = '/home/cvnar/JKK/datasets/aachen/images/images_upright/db_cyclegan/'
transform = transforms.Compose([
    transforms.ToTensor()#,
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  # 정규화 해제
    return tensor.clamp(0, 1)  # 값의 범위를 0과 1 사이로 제한

def save_image_opencv(tensor, filename):
    tensor = unnormalize(tensor)  # 정규화 해제
    image_np = tensor.permute(1, 2, 0).numpy()  # (높이, 너비, 채널) 형식으로 변경
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
    image_np = (image_np * 255).astype(np.uint8)  # [0, 1] 범위를 [0, 255] 범위로 변경
    cv2.imwrite(filename, image_np)

# 각 이미지에 대해 처리 수행
for image_file in tqdm(image_files):
    image_path = os.path.join(input_dir, image_file)
    input_image = load_and_preprocess_image(image_path).unsqueeze(0)
    output = cyclegan(input_image)
    output_image = output.squeeze().detach()

    if output_image.shape[0] == 3:  # RGB 이미지라고 가정
        # 결과 이미지 저장
        img_name = image_file.split('.')
        name, ext = img_name[0], img_name[1]
        image_file = f"{name}_cyclegan.{ext}"
        output_path = os.path.join(output_dir, image_file)

        # OpenCV를 사용하여 시각화
        output_image_for_cv2 = output_image.permute(1, 2, 0).numpy()  # (높이, 너비, 채널) 형식으로 변경
        output_image_for_cv2 = cv2.cvtColor(output_image_for_cv2, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
        normalized_image_data = ((output_image_for_cv2 - np.min(output_image_for_cv2)) / (np.max(output_image_for_cv2) - np.min(output_image_for_cv2)) * 255).astype(np.uint8)
        cv2.imwrite(output_path, normalized_image_data)