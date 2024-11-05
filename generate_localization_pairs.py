import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from models.crossvit_official import crossvit_tiny_224

class Config:
    """Configuration settings"""
    # Paths
    BASE_PATH = Path('datasets/aachen/images/images_upright')
    OUTPUT_PATH = Path('outputs/aachen')
    MODEL_PATH = Path('model/Student_Model/Best_Model.pth')
    
    # Image preprocessing
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Batch settings
    BATCH_SIZE = 32
    NUM_WORKERS = 8

class ImageDataset(Dataset):
    """Dataset for loading images from CSV"""
    def __init__(self, csv_path: str, image_key: str = 'Anchor'):
        self.data_frame = pd.read_csv(csv_path)
        self.image_key = image_key

    def __getitem__(self, idx):
        img_path = Config.BASE_PATH / self.data_frame[self.image_key][idx]
        return Config.TRANSFORM(Image.open(img_path))

    def __len__(self):
        return len(self.data_frame)

class FeatureExtractor:
    """Extract features from images using CrossViT model"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        """Initialize and load pretrained model"""
        model = crossvit_tiny_224(pretrained=True)
        checkpoint = torch.load(Config.MODEL_PATH)
        state_dict = {k.replace('module.', ''): v 
                     for k, v in checkpoint['crossvit_state_dict'].items()}
        model.load_state_dict(state_dict)
        return nn.DataParallel(model.to(self.device))
        
    def extract_features(self, dataset: Dataset) -> np.ndarray:
        """Extract features from dataset"""
        loader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            shuffle=False
        )
        
        features = []
        self.model.eval()
        
        with torch.no_grad():
            for images in tqdm(loader, desc='Extracting features'):
                output = self.model(images.to(self.device))
                features.append(F.normalize(output, p=2, dim=1).cpu())
                
        return torch.cat(features).numpy()

def generate_pairs(query_features: np.ndarray, db_features: np.ndarray, 
                  query_csv: Path, db_csv: Path, output_file: Path):
    """Generate and save image pairs based on nearest neighbors"""
    # Find nearest neighbors
    print("Finding nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(db_features)
    distances, indices = nbrs.kneighbors(query_features)
    
    # Load CSV data
    query_df = pd.read_csv(query_csv)
    db_df = pd.read_csv(db_csv)
    
    # Write pairs to file
    print(f"Saving pairs to {output_file}")
    with open(output_file, 'w') as f:
        pairs = []
        for i in range(len(indices)):
            for idx in indices[i]:
                pairs.append(f"{query_df['Query'][i]} {db_df['Anchor'][idx]}\n")
        
        # Sort and write pairs
        f.writelines(sorted(pairs))

def main():
    """Main execution function"""
    print("Initializing feature extractor...")
    extractor = FeatureExtractor()
    
    # Process database images
    print("\nProcessing database images...")
    db_dataset = ImageDataset('Table/DataBase4328.csv')
    db_features = extractor.extract_features(db_dataset)
    
    # Process query images
    print("\nProcessing query images...")
    query_dataset = ImageDataset('Table/Query922.csv', 'Query')
    query_features = extractor.extract_features(query_dataset)
    
    # Save features for visualization
    np.save('VisualizingEmbedding/np_train_features_student_4th.npy', db_features)
    
    # Generate pairs file
    print("\nGenerating image pairs...")
    generate_pairs(
        query_features,
        db_features,
        Path('Table/Query922.csv'),
        Path('Table/DataBase4328.csv'),
        Config.OUTPUT_PATH / 'Retrieved_Images.txt'
    )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()