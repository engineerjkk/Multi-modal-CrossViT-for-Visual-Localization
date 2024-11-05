import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import griddata
import pickle
from hloc.utils.read_write_model import (
    read_images_binary,
    read_points3D_binary,
    read_cameras_binary,
    qvec2rotmat
)
from segment_anything import sam_model_registry, SamPredictor

# Global constants
PATCH_SIZE = 14  # Patch size for grid
DEVICE = "cuda"
SAM_CHECKPOINT = "segment-anything-main/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

def initialize_sam():
    """Initialize Segment Anything Model (SAM)"""
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    return SamPredictor(sam)

def extract_patch_2D_points(xys, img_width, img_height, size):
    """
    Extract 2D points for each patch in the image grid
    Returns dictionary with (row, col) keys and lists of 2D points
    """
    patch_width = img_width // size
    patch_height = img_height // size
    patches_2D_points = {}
    
    for i in range(size):
        for j in range(size):
            x_start, x_end = i * patch_width, (i + 1) * patch_width
            y_start, y_end = j * patch_height, (j + 1) * patch_height
            
            mask = (xys[:, 0] >= x_start) & (xys[:, 0] < x_end) & \
                  (xys[:, 1] >= y_start) & (xys[:, 1] < y_end)
            
            patches_2D_points[(j, i)] = xys[mask].tolist()
    
    return patches_2D_points

def get_camera_matrix(camera):
    """Create camera matrix from camera parameters"""
    return np.array([
        [camera.params[0], 0, camera.params[1]],
        [0, camera.params[0], camera.params[2]],
        [0, 0, 1]
    ])

def convert_2D_to_3D(point_2d, K, RT):
    """Convert 2D point to 3D world coordinates"""
    point_2d = np.array([int(point_2d[0]), int(point_2d[1]), 1]).reshape(3, 1)
    point_3d_cam = np.linalg.inv(K).dot(point_2d)
    point_3d_cam = point_3d_cam / np.linalg.norm(point_3d_cam)
    
    point_3d_world = np.linalg.inv(RT).dot(np.vstack([point_3d_cam * 10, 1]))
    return point_3d_world[:3].reshape(-1)

def process_image(image_id, image, cameras, predictor, size=PATCH_SIZE):
    """Process single image and extract patch information"""
    xys = image.xys
    img = Image.open(f'datasets/aachen/images/images_upright/{image.name}')
    img_width, img_height = img.size
    
    # Extract patches and process with SAM
    patches_2D_ids = extract_patch_2D_points(xys, img_width, img_height, size)
    cv_image = cv2.cvtColor(cv2.imread(str(f'datasets/aachen/images/images_upright/{image.name}')), 
                           cv2.COLOR_BGR2RGB)
    
    predictor.set_image(cv_image)
    
    # Process patches and create mappings
    image_key = {}
    image_key_3d = {'Token': image.tvec}
    
    # Create RT matrix
    camera = cameras[image.camera_id]
    K = get_camera_matrix(camera)
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    RT_4x4 = np.eye(4)
    RT_4x4[:3, :3] = R
    RT_4x4[:3, 3] = t.reshape(-1)
    
    # Process each patch
    for key, patches in patches_2D_ids.items():
        patch_center_x = key[1] * (img_width // size) + (img_width // (2*size))
        patch_center_y = key[0] * (img_height // size) + (img_height // (2*size))
        
        if not patches and key[0] == 0:  # Sky patch
            image_key[key] = [0, 0]
            image_key_3d[key] = np.array([100000, 100000, 100000])
        else:
            image_key[key] = [patch_center_x, patch_center_y]
            image_key_3d[key] = convert_2D_to_3D([patch_center_x, patch_center_y], K, RT_4x4)
    
    return {
        'image_key': dict(sorted(image_key.items())),
        'image_key_3d': image_key_3d
    }

def rotary_position_embeddings(coords, dim=192):
    """
    Compute Rotary Position Embeddings for coordinates
    Extends input coordinates to higher dimensions using sin/cos embeddings
    """
    coords = 2.0 * (coords - coords.min()) / (coords.max() - coords.min()) - 1.0
    angles = coords * torch.pi
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    
    embeddings = torch.zeros(*coords.shape[:-1], dim)
    full_blocks = dim // 14
    
    for i in range(full_blocks):
        embeddings[..., 14*i:14*i+7] = sin
        embeddings[..., 14*i+7:14*i+14] = cos
    
    remaining_dims = dim % 14
    if remaining_dims > 0:
        extended_sin_cos = torch.cat((sin, cos), dim=-1).flatten()
        embeddings[..., -remaining_dims:] = extended_sin_cos[:remaining_dims]
    
    return embeddings

def main():
    """Main processing pipeline"""
    # Load data
    images = read_images_binary('outputs/aachen/sfm_superpoint+superglue/images.bin')
    cameras = read_cameras_binary('outputs/aachen/sfm_superpoint+superglue/cameras.bin')
    predictor = initialize_sam()
    
    # Process all images
    total_image_data = {}
    for image_id, image in tqdm(images.items(), desc="Processing images"):
        total_image_data[image_id] = process_image(image_id, image, cameras, predictor)
    
    # Save intermediate results
    with open("DataBase/TotalImageKey3D_14x14.pickle", "wb") as f:
        pickle.dump({k: v['image_key_3d'] for k, v in total_image_data.items()}, f)
    
    # Process embeddings
    encoded_points = {}
    for id, data in tqdm(total_image_data.items(), desc="Computing embeddings"):
        points_3d = data['image_key_3d']
        encoded_points[id] = {}
        for key, value in points_3d.items():
            if value is not None:
                input_tensor = torch.tensor(value).unsqueeze(0)
                output = rotary_position_embeddings(input_tensor)
                encoded_points[id][key] = output.squeeze(0)
    
    # Create final tensor dictionary
    tensor_dict = {}
    for key in tqdm(encoded_points, desc="Creating final tensors"):
        token_value = encoded_points[key]['Token']
        patch_values = [encoded_points[key][patch_key] for patch_key in encoded_points[key] 
                       if patch_key != 'Token']
        tensor_array = torch.stack([torch.tensor(token_value)] + 
                                 [torch.tensor(patch_value) for patch_value in patch_values])
        tensor_dict[key] = tensor_array
    
    # Save final results
    with open('DataBase/Large_Patch_14x14_RT_RoPE_Tensor.pickle', 'wb') as f:
        pickle.dump(tensor_dict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()