from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from models.crossvit_official import crossvit_tiny_224

# Constants
DATA_PATH = Path('datasets/aachen')
OUTPUT_PATH = Path('DataBase')
QUERIES_PATH = DATA_PATH / 'queries'
SFM_PATH = Path('outputs/aachen/sfm_superpoint+superglue')
MIN_COMMON_POINTS = 1

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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
        if len(points) < 4:
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
        exclude_img_ids = {1167, 3266, 3302, 3297}
        
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

class ImageDataset(Dataset):
    """Dataset for processing anchor images"""
    def __init__(self, database, images, base_path='datasets/aachen/images/images_upright/'):
        self.database = database
        self.images = images
        self.base_path = Path(base_path)
        
    def __len__(self):
        return len(self.database)
    
    def __getitem__(self, idx):
        data = self.database[idx]
        image_id = data['Anchor'][0]
        image_path = self.base_path / self.images[image_id].name
        image = Image.open(image_path)
        return transform(image), self.images[image_id].id

def normalize_distances(distances):
    """Normalize distances to sum to 1"""
    total = sum(distances)
    if total == 0:
        raise ValueError("Sum of distances cannot be zero")
    return [d / total for d in distances]

def compute_features(model, database, images, batch_size=1, num_workers=8):
    """Extract features from images using the model"""
    model.eval()
    features = {}
    dataset = ImageDataset(database, images)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    
    print("Computing features...")
    with torch.no_grad():
        for images, keys in tqdm(dataloader):
            images = images.cuda()
            output = model(images)
            output = F.normalize(output, p=2, dim=1)
            for key, feature in zip(keys, output):
                features[int(key)] = feature
    return features

def compute_negative_distances(database, features):
    """Compute normalized distances for negative samples"""
    distance_fn = nn.PairwiseDistance(p=2)
    
    print("Computing negative distances...")
    for entry in tqdm(database):
        anchor_feature = torch.tensor(features[entry['Anchor'][0]])
        distances = []
        for neg_id in entry['Negative']:
            neg_feature = torch.tensor(features[neg_id])
            dist = distance_fn(anchor_feature, neg_feature).float().squeeze()
            distances.append(dist.item())
        entry['Negative_Sampling'] = normalize_distances(distances)
    
    return database

def merge_query_files(day_file: Path, night_file: Path, output_file: Path) -> dict:
    """Merge day and night query files into a single output file"""
    unique_entries = {}
    
    for file in [day_file, night_file]:
        with open(file, 'r') as f:
            for line in f.readlines():
                if line := line.strip():
                    key = line.split()[0]
                    unique_entries[key] = line
    
    with open(output_file, 'w') as f:
        for key in sorted(unique_entries.keys()):
            f.write(f"{unique_entries[key]}\n")
    
    print(f"Merged files successfully. Total unique entries: {len(unique_entries)}")
    return unique_entries

def create_csv_files():
    """Create CSV files for queries and database"""
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Merge query files
    day_queries = QUERIES_PATH / 'day_time_queries_with_intrinsics.txt'
    night_queries = QUERIES_PATH / 'night_time_queries_with_intrinsics.txt'
    merged_queries = QUERIES_PATH / 'all_queries_merged.txt'
    
    unique_entries = merge_query_files(day_queries, night_queries, merged_queries)
    
    # Create query CSV
    with open(merged_queries, 'r') as f:
        image_paths = [line.split(' ', 1)[0] for line in f.readlines()]
    
    query_df = pd.DataFrame(image_paths, columns=['Query'])
    query_df.to_csv(OUTPUT_PATH / 'Query922.csv', index=False)
    print(f"Created query CSV with {len(query_df)} entries")
    
    # Create database CSV
    with open(OUTPUT_PATH / 'DataBase3Column.pkl', 'rb') as f:
        database = pickle.load(f)
    images = read_images_binary(str(SFM_PATH / 'images.bin'))
    
    # Create DataBase4324.csv (excluding specific images)
    exclude_img_ids = {1167, 3266, 3302, 3297}
    image_names_filtered = [
        images[entry['Anchor'][0]].name for entry in tqdm(database, desc="Processing database entries (filtered)") 
        if images[entry['Anchor'][0]].id not in exclude_img_ids
    ]
    db_df_filtered = pd.DataFrame(image_names_filtered, columns=['Anchor'])
    db_df_filtered.to_csv(OUTPUT_PATH / 'DataBase4324.csv', index=False)
    print(f"Created filtered database CSV with {len(db_df_filtered)} entries")
    
    # Create DataBase4328.csv (including all images)
    image_names_all = [
        images[entry['Anchor'][0]].name for entry in tqdm(database, desc="Processing database entries (all)")
    ]
    db_df_all = pd.DataFrame(image_names_all, columns=['Anchor'])
    db_df_all.to_csv(OUTPUT_PATH / 'DataBase4328.csv', index=False)
    print(f"Created complete database CSV with {len(db_df_all)} entries")

def prepare_validation_data(input_path: Path, output_path: Path):
    """Prepare validation dataset from database"""
    print(f"Loading database from {input_path}")
    with open(input_path, 'rb') as f:
        database = pickle.load(f)
    
    validation_data = {
        'Anchor': [],
        'Positive': []
    }
    
    for entry in tqdm(database, desc="Processing validation data"):
        validation_data['Anchor'].append([entry['Anchor'][0]])
        validation_data['Positive'].append(entry['Positive'])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(validation_data, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Validation data preparation completed. Total entries: {len(validation_data['Anchor'])}")

def process_point_cloud():
    """Process point cloud and generate output_iou_database.pkl"""
    images_path = str(SFM_PATH / 'images.bin')
    points3d_path = str(SFM_PATH / 'points3D.bin')
    output_path = OUTPUT_PATH / 'output_iou_database.pkl'
    min_common_points = 8

    if not all(Path(p).exists() for p in [images_path, points3d_path]):
        raise FileNotFoundError("Required input files not found")

    try:
        print("\nInitializing Point Cloud Processor...")
        processor = PointCloudProcessor(
            min_common_points,
            images_path,
            points3d_path
        )
        
        print("Building and computing database...")
        processor.build_database()
        
        print("Saving results...")
        processor.save_database(str(output_path))
        
        print(f"Successfully saved results to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error occurred in point cloud processing: {str(e)}")
        return False

def compute_and_save_features():
    """Compute and save feature distances"""
    # Load data
    with open(OUTPUT_PATH / 'output_iou_database.pkl', 'rb') as f:
        database = pickle.load(f)
    images = read_images_binary(str(SFM_PATH / 'images.bin'))
    
    # Initialize model
    model = crossvit_tiny_224(pretrained=True)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
    
    # Compute features and distances
    features = compute_features(model, database, images)
    database = compute_negative_distances(database, features)
    
    # Save results
    output_path = OUTPUT_PATH / 'DataBase_Norm_Tiny.pickle'
    with open(output_path, 'wb') as f:
        pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)
    
    print("Feature computation completed")

def prepare_point_cloud_database():
    """Prepare and save point cloud database"""
    output_path = OUTPUT_PATH / 'DataBase3Column.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processor = PointCloudProcessor(
        MIN_COMMON_POINTS,
        str(SFM_PATH / 'images.bin'),
        str(SFM_PATH / 'points3D.bin')
    )
    
    print("Building database...")
    database = processor.build_database()
    
    with open(output_path, 'wb') as f:
        pickle.dump(database, f)
    
    print("Point cloud database preparation completed")

def main():
    """Main execution function with integrated processing"""
    print("Starting integrated processing pipeline...")
    
    # Step 1: Process point cloud and generate output_iou_database.pkl
    print("\nExecuting point cloud processing to generate output_iou_database.pkl...")
    if not process_point_cloud():
        raise RuntimeError("Point cloud processing failed")
        
    # Check if required file exists
    if not (OUTPUT_PATH / 'output_iou_database.pkl').exists():
        raise FileNotFoundError("output_iou_database.pkl not found. Point cloud processing may have failed.")
    
    # Step 2: Prepare point cloud database
    print("\nPreparing point cloud database...")
    prepare_point_cloud_database()
    
    # Step 3: Create CSV files
    print("\nCreating CSV files...")
    create_csv_files()
    
    # Step 4: Compute features and distances
    print("\nComputing features and distances...")
    compute_and_save_features()
    
    print("\nStarting validation data preparation...")
    
    # Step 5: Prepare validation datasets
    prepare_validation_data(
        OUTPUT_PATH / 'output_iou_database.pkl',
        Path('ValidationSet/TeacherVal.pickle')
    )
    prepare_validation_data(
        OUTPUT_PATH / 'DataBase3Column.pkl',
        Path('ValidationSet/ValidationAll_Teacher_Key_Real.pickle')
    )
    
    print("All processing steps completed successfully!")

if __name__ == "__main__":
    main()