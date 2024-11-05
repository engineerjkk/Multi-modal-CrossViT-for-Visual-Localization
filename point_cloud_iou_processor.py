from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import pickle
from collections import defaultdict
from hloc.utils.read_write_model import read_images_binary, read_points3D_binary

@dataclass
class BBox3D:
    """Class representing a 3D bounding box"""
    min_coords: np.ndarray
    max_coords: np.ndarray
    
    @property
    def volume(self) -> float:
        dims = np.maximum(0, self.max_coords - self.min_coords)
        return float(np.prod(dims))

class PointCloudProcessor:
    """Class for processing 3D point clouds and computing IoU values"""
    
    def __init__(
        self,
        min_common_points: int,
        images_path: str,
        points3d_path: str,
        outlier_threshold: float = 3.0
    ):
        self.min_common_points = min_common_points
        self.outlier_threshold = outlier_threshold
        self.hloc_images = read_images_binary(images_path)
        self.hloc_points3d = read_points3D_binary(points3d_path)
        
        # Initialize image and point mappings
        self.image_to_points = self._build_image_point_mapping()
        self.point_to_images = self._build_point_image_mapping()
        self.database = None

    def _build_image_point_mapping(self) -> Dict[int, Set[int]]:
        """Create mapping from image IDs to their 3D points"""
        return {
            img_id: {pid for pid in img.point3D_ids if pid != -1}
            for img_id, img in self.hloc_images.items()
        }
    
    def _build_point_image_mapping(self) -> Dict[int, List[int]]:
        """Create mapping from 3D points to images containing them"""
        mapping = defaultdict(list)
        for img_id, points in self.image_to_points.items():
            for point in points:
                mapping[point].append(img_id)
        return dict(mapping)

    def _get_point_cloud(self, img_id: int) -> np.ndarray:
        """Get 3D point cloud for an image"""
        points = [
            self.hloc_points3d[pid].xyz
            for pid in self.hloc_images[img_id].point3D_ids
            if pid != -1
        ]
        return np.array(points) if points else np.array([])

    def _remove_outliers(self, points: np.ndarray) -> np.ndarray:
        """Remove outliers using Z-score method"""
        if len(points) < 4:  # Minimum required points
            return points
        z_scores = np.abs(stats.zscore(points))
        mask = ~(z_scores > self.outlier_threshold).any(axis=1)
        return points[mask]

    def _compute_bbox(self, points: np.ndarray) -> Optional[BBox3D]:
        """Compute 3D bounding box from point cloud"""
        if len(points) < 3:
            return None
        return BBox3D(
            min_coords=np.min(points, axis=0),
            max_coords=np.max(points, axis=0)
        )

    def _compute_iou(self, bbox1: BBox3D, bbox2: BBox3D) -> float:
        """Compute IoU between two 3D bounding boxes"""
        intersection_min = np.maximum(bbox1.min_coords, bbox2.min_coords)
        intersection_max = np.minimum(bbox1.max_coords, bbox2.max_coords)
        
        intersection_bbox = BBox3D(intersection_min, intersection_max)
        union_volume = bbox1.volume + bbox2.volume - intersection_bbox.volume
        
        return intersection_bbox.volume / union_volume if union_volume > 0 else 0.0

    def compute_image_pair_iou(self, img1_id: int, img2_id: int) -> float:
        """Compute IoU between point clouds of two images"""
        points1 = self._get_point_cloud(img1_id)
        points2 = self._get_point_cloud(img2_id)
        
        if len(points1) < 3 or len(points2) < 3:
            return 0.0
            
        points1 = self._remove_outliers(points1)
        points2 = self._remove_outliers(points2)
        
        bbox1 = self._compute_bbox(points1)
        bbox2 = self._compute_bbox(points2)
        
        if not bbox1 or not bbox2:
            return 0.0
            
        return self._compute_iou(bbox1, bbox2)

    def build_database(self) -> List[Dict]:
        """Build database of anchor-positive-negative image relationships"""
        database = []
        exclude_img_ids = {1167, 3266, 3302, 3297}  # Images with insufficient 3D points
        
        for anchor_id, anchor_points in tqdm(self.image_to_points.items(), desc="Building database"):
            if len(anchor_points) <= 1:
                continue
                
            entry = {
                'Anchor': [self.hloc_images[anchor_id].id],
                'Positive': [],
                'Negative': [],
                'Positive IoU': []
            }
            
            checked_images = {anchor_id}
            
            # Find positive pairs
            for point in anchor_points:
                for other_id in self.point_to_images[point]:
                    if other_id in checked_images:
                        continue
                        
                    checked_images.add(other_id)
                    common_points = anchor_points & self.image_to_points[other_id]
                    
                    if len(common_points) >= self.min_common_points:
                        entry['Positive'].append(self.hloc_images[other_id].id)
                        iou = self.compute_image_pair_iou(anchor_id, other_id)
                        entry['Positive IoU'].append(iou)
            
            # Add negative pairs
            entry['Negative'] = [
                self.hloc_images[other_id].id
                for other_id in self.hloc_images
                if other_id not in checked_images and 
                self.hloc_images[other_id].id not in exclude_img_ids
            ]
            
            if entry['Positive']:
                database.append(entry)
                
        self.database = database
        return database

    def save_database(self, output_path: str):
        """Save database to a file"""
        if not self.database:
            raise ValueError("Database hasn't been built yet. Call build_database() first.")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.database, f)

def main():
    """Main execution function"""
    # Configuration
    BASE_PATH = Path('outputs/aachen/sfm_superpoint+superglue')
    IMAGES_PATH = BASE_PATH / 'images.bin'
    POINTS3D_PATH = BASE_PATH / 'points3D.bin'
    OUTPUT_PATH = Path('DataBase/output_iou_database.pkl')
    MIN_COMMON_POINTS = 8

    # Check if paths exist
    if not all(p.exists() for p in [IMAGES_PATH, POINTS3D_PATH]):
        raise FileNotFoundError("Required input files not found")

    try:
        print("Initializing Point Cloud Processor...")
        processor = PointCloudProcessor(
            MIN_COMMON_POINTS,
            str(IMAGES_PATH),
            str(POINTS3D_PATH)
        )
        
        print("Building and computing database...")
        processor.build_database()
        
        print("Saving results...")
        processor.save_database(str(OUTPUT_PATH))
        
        print(f"Successfully saved results to {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()