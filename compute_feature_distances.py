import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pickle
from tqdm import tqdm
from hloc.utils.read_write_model import read_images_binary
from models.crossvit_official import crossvit_tiny_224

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
        
        # Load and transform image
        image_path = self.base_path / self.images[image_id].name
        image = Image.open(image_path)
        image = transform(image)
        
        return image, self.images[image_id].id

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
        
        # Compute distances to negative samples
        distances = []
        for neg_id in entry['Negative']:
            neg_feature = torch.tensor(features[neg_id])
            dist = distance_fn(anchor_feature, neg_feature).float().squeeze()
            distances.append(dist.item())
            
        # Normalize distances
        entry['Negative_Sampling'] = normalize_distances(distances)
    
    return database

def main():
    # Paths
    DATABASE_PATH = 'DataBase/output_iou_database.pkl'
    IMAGES_PATH = 'outputs/aachen/sfm_superpoint+superglue/images.bin'
    OUTPUT_PATH = 'DataBase/DataBase_Norm_Tiny.pickle'
    
    # Load data
    print("Loading database and images...")
    with open(DATABASE_PATH, 'rb') as f:
        database = pickle.load(f)
    images = read_images_binary(IMAGES_PATH)
    
    # Initialize model
    print("Initializing model...")
    model = crossvit_tiny_224(pretrained=True)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
    
    # Compute features and distances
    features = compute_features(model, database, images)
    database = compute_negative_distances(database, features)
    
    # Save results
    print(f"Saving results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)
    
    print("Processing completed")

if __name__ == "__main__":
    main()