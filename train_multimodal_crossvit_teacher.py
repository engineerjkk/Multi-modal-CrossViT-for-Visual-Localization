import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pickle
import random
import multiprocessing
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from hloc.utils.read_write_model import read_images_binary
from models.crossvit_PE_RT_official_MultiModal import crossvit_tiny_224

class Config:
    """Training configuration"""
    # Paths
    IMAGE_PATH = Path('datasets/aachen/images/images_upright')
    MODEL_PATH = Path('model/Train_Teacher')
    LOG_PATH = Path('runs/Train_Teacher')
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 10001
    VALIDATION_START = 3000
    SAVE_INTERVAL = 100
    MARGIN = 0.5
    
    # Image preprocessing
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ImageDataset(Dataset):
    """Dataset for training with positive/negative pairs"""
    def __init__(self, database, images, embeddings):
        self.database = database
        self.images = images
        self.embeddings = embeddings
        
    def get_image_path(self, idx, is_night=False):
        """Get image path with night image handling"""
        img_name = self.images[idx].name
        if is_night:
            name = Path(img_name)
            img_name = f"{name.parent}_cyclegan/{name.stem}_cyclegan{name.suffix}"
        return Config.IMAGE_PATH / img_name
    
    def __getitem__(self, idx):
        data = self.database[idx]
        
        # Get anchor data
        anchor_path = self.get_image_path(data['Anchor'][0])
        anchor_img = Config.TRANSFORM(Image.open(anchor_path))
        anchor_embed = self.embeddings[data['Anchor'][0]].clone().detach().float()
        
        # Get positive/negative sample
        is_positive = random.choice([0, 1])
        samples = data['Positive'] if is_positive else data['Negative']
        iou = data['Positive IoU'] if is_positive else [0] * len(samples)
        
        # Get chosen sample
        chosen_idx = random.randrange(len(samples))
        is_night = random.choice([0, 1])
        chosen_path = self.get_image_path(samples[chosen_idx], is_night)
        chosen_img = Config.TRANSFORM(Image.open(chosen_path))
        chosen_embed = self.embeddings[samples[chosen_idx]].clone().detach().float()
        
        return anchor_img, chosen_img, torch.tensor(iou[chosen_idx], dtype=torch.float32), \
               anchor_embed, chosen_embed
    
    def __len__(self):
        return len(self.database)

class ValidationDataset(Dataset):
    """Dataset for validation"""
    def __init__(self, keys, images, embeddings):
        self.keys = keys
        self.images = images
        self.embeddings = embeddings
        
    def __getitem__(self, idx):
        key = self.keys[idx]
        img_path = Config.IMAGE_PATH / self.images[key].name
        image = Config.TRANSFORM(Image.open(img_path).convert('RGB'))
        return image, self.embeddings[key]
    
    def __len__(self):
        return len(self.keys)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""
    def __init__(self, margin=Config.MARGIN):
        super().__init__()
        self.margin = margin
        self.dist_fn = nn.PairwiseDistance(p=2)

    def forward(self, out1, out2, label):
        dist = self.dist_fn(out1, out2).squeeze()
        return torch.sum(label * dist.pow(2) * 0.5 + 
                        (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2) * 0.5)

class Trainer:
    """Model trainer class"""
    def __init__(self):
        self.setup_paths()
        self.load_data()
        self.setup_model()
        self.writer = SummaryWriter(Config.LOG_PATH)
        
    def setup_paths(self):
        """Ensure necessary directories exist"""
        Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        Config.LOG_PATH.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load necessary data files"""
        print("Loading data...")
        self.images = read_images_binary('outputs/aachen/sfm_superpoint+superglue/images.bin')
        
        with open('DataBase/DataBase_Norm_Tiny.pickle', 'rb') as f:
            self.database = pickle.load(f)
        with open('DataBase/Large_Patch_14x14_RT_RoPE_Tensor.pickle', 'rb') as f:
            self.embeddings = pickle.load(f)
        with open('ValidationSet/TeacherVal.pickle', 'rb') as f:
            self.val_data = pickle.load(f)
            
    def setup_model(self):
        """Initialize model and training components"""
        self.model = nn.DataParallel(
            crossvit_tiny_224(pretrained=True).cuda(),
            device_ids=[0, 1]
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = ContrastiveLoss()
        
    def train(self):
        """Main training loop"""
        # Setup data loaders
        train_loader = DataLoader(
            ImageDataset(self.database, self.images, self.embeddings),
            batch_size=Config.BATCH_SIZE, shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )
        
        unique_keys = list(set([val for sublist in self.val_data['Anchor'] + 
                              self.val_data['Positive'] for val in sublist]))
        val_loader = DataLoader(
            ValidationDataset(unique_keys, self.images, self.embeddings),
            batch_size=Config.BATCH_SIZE,
            num_workers=multiprocessing.cpu_count()
        )
        
        best_recall = 0
        for epoch in tqdm(range(Config.EPOCHS), desc="Training"):
            # Training phase
            self.model.train()
            for batch in train_loader:
                loss = self.train_step(batch)
                self.writer.add_scalar('training_loss', loss, epoch)
            
            # Validation phase
            if epoch >= Config.VALIDATION_START or (epoch % Config.SAVE_INTERVAL == 0):
                recall = self.validate(val_loader, unique_keys)
                self.writer.add_scalar('validation_recall', recall, epoch)
                print(f"Epoch {epoch}, Recall@3: {recall:.4f}")
                
                # Save checkpoints
                if recall > best_recall:
                    best_recall = recall
                    self.save_checkpoint(epoch, recall, is_best=True)
                if epoch % Config.SAVE_INTERVAL == 0:
                    self.save_checkpoint(epoch, recall)
    
    def train_step(self, batch):
        """Single training step"""
        anchor_img, chosen_img, iou, anchor_embed, chosen_embed = [x.cuda() for x in batch]
        
        anchor_out = self.model(anchor_img, anchor_embed)
        chosen_out = self.model(chosen_img, chosen_embed)
        loss = self.criterion(anchor_out, chosen_out, iou)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader, unique_keys):
        """Compute validation metrics"""
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for imgs, embeds in val_loader:
                output = self.model(imgs.cuda(), embeds.cuda())
                features.extend(output.cpu().numpy())
        
        nn_model = NearestNeighbors(n_neighbors=3).fit(features)
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}
        
        recalls = []
        for anchor, positives in zip(self.val_data['Anchor'], self.val_data['Positive']):
            neighbors = nn_model.kneighbors(features[key_to_idx[anchor[0]]].reshape(1, -1))[1][0]
            recalls.append(any(key_to_idx[p] in neighbors for p in positives))
            
        return sum(recalls) / len(recalls)
    
    def save_checkpoint(self, epoch, recall, is_best=False):
        """Save model checkpoint"""
        prefix = 'best_' if is_best else ''
        path = Config.MODEL_PATH / f'{prefix}model_{epoch}_{recall:.4f}.pth'
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'recall': recall
        }, path)

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()