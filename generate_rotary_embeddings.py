import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from hloc.utils.read_write_model import (
    read_images_binary, read_points3D_binary,
    read_cameras_binary, qvec2rotmat
)
from segment_anything import sam_model_registry, SamPredictor

class PositionEmbeddingGenerator:
    """Generate rotary position embeddings for image patches"""
    
    def __init__(self, image_size=14, sam_checkpoint="segment-anything-main/sam_vit_h_4b8939.pth"):
        self.image_size = image_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize SAM model
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam.to(self.device))
    
    def extract_patch_points(self, xys, img_width, img_height):
        """Extract 2D points for each image patch"""
        patch_width = img_width // self.image_size
        patch_height = img_height // self.image_size
        
        patches = {}
        for i in range(self.image_size):
            for j in range(self.image_size):
                x_start, x_end = i * patch_width, (i + 1) * patch_width
                y_start, y_end = j * patch_height, (j + 1) * patch_height
                
                mask = (xys[:, 0] >= x_start) & (xys[:, 0] < x_end) & \
                       (xys[:, 1] >= y_start) & (xys[:, 1] < y_end)
                
                patches[(j, i)] = xys[mask].tolist()
        
        return patches
    
    def compute_3d_positions(self, patches, image, camera):
        """Compute 3D positions for patches"""
        positions = {'Token': image.tvec}
        
        # Camera parameters
        K = np.array([
            [camera.params[0], 0, camera.params[1]],
            [0, camera.params[0], camera.params[2]],
            [0, 0, 1]
        ])
        
        R = qvec2rotmat(image.qvec)
        t = image.tvec.reshape(3, 1)
        RT = np.eye(4)
        RT[:3, :3] = R
        RT[:3, 3] = t.reshape(-1)
        
        for key, patch_center in patches.items():
            if patch_center == [0, 0]:
                positions[key] = np.array([100000, 100000, 100000])
            else:
                point_2d = np.array([int(patch_center[0]), int(patch_center[1]), 1]).reshape(3, 1)
                point_3d_cam = np.linalg.inv(K).dot(point_2d)
                point_3d_cam = point_3d_cam / np.linalg.norm(point_3d_cam)
                point_3d_world = np.linalg.inv(RT).dot(np.vstack([point_3d_cam * 10, 1]))
                positions[key] = point_3d_world[:3].reshape(-1)
        
        return positions

    @staticmethod
    def compute_rotary_embeddings(coords, dim=192):
        """Compute rotary position embeddings"""
        # Normalize coordinates
        coords = 2.0 * (coords - coords.min()) / (coords.max() - coords.min()) - 1.0
        angles = coords * torch.pi
        
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        embeddings = torch.zeros(*coords.shape[:-1], dim)
        full_blocks = dim // 14
        
        # Fill embeddings with sin/cos blocks
        for i in range(full_blocks):
            embeddings[..., 14*i:14*i+7] = sin
            embeddings[..., 14*i+7:14*i+14] = cos
        
        # Handle remaining dimensions
        remaining = dim % 14
        if remaining > 0:
            extended = torch.cat((sin, cos), dim=-1).flatten()
            embeddings[..., -remaining:] = extended[:remaining]
            
        return embeddings
    
    def process_images(self, images_path, points3d_path, cameras_path):
        """Process all images and generate embeddings"""
        # Load data
        images = read_images_binary(images_path)
        points3d = read_points3D_binary(points3d_path)
        cameras = read_cameras_binary(cameras_path)
        
        # Process each image
        positions = {}
        for image_id, image in tqdm(images.items(), desc="Processing images"):
            if image_id in {1167, 3266, 3302, 3297}:  # Skip problematic images
                continue
                
            img_path = Path('datasets/aachen/images/images_upright') / image.name
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Extract patches and compute 3D positions
            patches = self.extract_patch_points(image.xys, img_width, img_height)
            positions[image_id] = self.compute_3d_positions(
                patches, image, cameras[image.camera_id]
            )
        
        return positions
    
    def generate_embeddings(self, positions):
        """Generate final embeddings for all positions"""
        embeddings = {}
        
        for img_id, pos in tqdm(positions.items(), desc="Generating embeddings"):
            img_embeddings = {}
            for key, value in pos.items():
                if value is not None:
                    input_tensor = torch.tensor(value).unsqueeze(0)
                    output = self.compute_rotary_embeddings(input_tensor)
                    img_embeddings[key] = output.squeeze(0)
            
            # Stack token and patch embeddings
            token = img_embeddings['Token']
            patches = [img_embeddings[k] for k in img_embeddings if k != 'Token']
            embeddings[img_id] = torch.stack([token] + patches)
        
        return embeddings

def main():
    # Configuration
    IMAGE_SIZE = 14
    PATHS = {
        'images': 'outputs/aachen/sfm_superpoint+superglue/images.bin',
        'points3d': 'outputs/aachen/sfm_superpoint+superglue/points3D.bin',
        'cameras': 'outputs/aachen/sfm_superpoint+superglue/cameras.bin',
        'sam': 'segment-anything-main/sam_vit_h_4b8939.pth',
        'output': 'DataBase/Large_Patch_14x14_RT_RoPE_Tensor.pickle'
    }
    
    # Generate embeddings
    generator = PositionEmbeddingGenerator(
        image_size=IMAGE_SIZE,
        sam_checkpoint=PATHS['sam']
    )
    
    positions = generator.process_images(
        PATHS['images'],
        PATHS['points3d'],
        PATHS['cameras']
    )
    
    embeddings = generator.generate_embeddings(positions)
    
    # Save results
    with open(PATHS['output'], 'wb') as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Results saved to {PATHS['output']}")

if __name__ == "__main__":
    main()