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
from hloc.utils.read_write_model import read_images_binary
from models.crossvit_official import crossvit_tiny_224
from point_cloud_iou_processor import PointCloudProcessor

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
    # Create output directory
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
    
    image_names = [images[entry['Anchor'][0]].name for entry in tqdm(database, 
                                                                    desc="Processing database entries")]
    db_df = pd.DataFrame(image_names, columns=['Anchor'])
    db_df.to_csv(OUTPUT_PATH / 'DataBase4324.csv', index=False)
    print(f"Created database CSV with {len(db_df)} entries")

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

def main():
    """Main execution function"""
    # Step 1: Prepare point cloud database
    prepare_point_cloud_database()
    
    # Step 2: Create CSV files
    create_csv_files()
    
    # Step 3: Compute features and distances
    compute_and_save_features()
    
    # Step 4: Prepare validation datasets
    prepare_validation_data(
        OUTPUT_PATH / 'output_iou_database.pkl',
        Path('ValidationSet/TeacherVal.pickle')
    )
    prepare_validation_data(
        OUTPUT_PATH / 'DataBase3Column.pkl',
        Path('ValidationSet/ValidationAll_Teacher_Key_Real.pickle')
    )

if __name__ == "__main__":
    main()