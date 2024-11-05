import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MARGIN = 0.5
    BASE_PATH = Path('datasets/aachen/images/images_upright')
    MODEL_PATH = Path('model/Train_Teacher')
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ImageDataset(Dataset):
    """Training dataset with positive/negative pairs"""
    def __init__(self, database, images, embeddings):
        self.database = database
        self.images = images
        self.embeddings = embeddings
        
    def __getitem__(self, idx):
        data = self.database[idx]
        
        # Process anchor
        anchor_img = Config.TRANSFORM(Image.open(Config.BASE_PATH / self.images[data['Anchor'][0]].name))
        anchor_embed = self.embeddings[data['Anchor'][0]].clone().detach().float()
        
        # Select positive/negative sample
        is_positive = random.choice([0, 1])
        is_night = random.choice([0, 1])
        sample_type = 'Positive' if is_positive else 'Negative'
        
        # Get chosen sample
        sample_idx = random.choice(range(len(data[sample_type]))) if is_positive else \
                    random.choices(range(len(data[sample_type])), weights=data['Negative_Sampling'], k=1)[0]
        
        # Get image path
        img_name = self.images[data[sample_type][sample_idx]].name
        if is_night:
            parts = Path(img_name)
            img_name = f"{parts.parent}_cyclegan/{parts.stem}_cyclegan{parts.suffix}"
            
        chosen_img = Config.TRANSFORM(Image.open(Config.BASE_PATH / img_name))
        chosen_embed = self.embeddings[data[sample_type][sample_idx]].clone().detach().float()
        iou = data[f'{sample_type} IoU'][sample_idx] if is_positive else 0
        
        return anchor_img, chosen_img, torch.tensor(iou, dtype=torch.float32), \
               anchor_embed, chosen_embed
               
    def __len__(self):
        return len(self.database)

class Trainer:
    """Model trainer class"""
    def __init__(self):
        self.setup()
        self.model = self.load_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.PairwiseDistance(p=2)
        self.writer = SummaryWriter('runs/KnowledgeDistillation_Student')
        
    def setup(self):
        """Load necessary data"""
        self.images = read_images_binary('outputs/aachen/sfm_superpoint+superglue/images.bin')
        with open('DataBase/DataBase_Norm_Tiny.pickle', 'rb') as f:
            self.database = pickle.load(f)
        with open('DataBase/Large_Patch_14x14_RT_RoPE_Tensor.pickle', 'rb') as f:
            self.embeddings = pickle.load(f)
        with open('ValidationSet/ValidationAll_Teacher_Key_Real.pickle', 'rb') as f:
            self.val_data = pickle.load(f)
            
    def load_model(self):
        """Initialize and load pretrained model"""
        model = crossvit_tiny_224(pretrained=True)
        checkpoint = torch.load(Config.MODEL_PATH / 'best_model1867_0.8984736355226642.pth')
        model.load_state_dict({k.replace('module.', ''): v 
                             for k, v in checkpoint['crossvit_state_dict'].items()})
        return nn.DataParallel(model.cuda(), device_ids=[0,1])
        
    def train(self):
        """Main training loop"""
        train_loader = self.get_dataloader(is_train=True)
        val_loader = self.get_dataloader(is_train=False)
        best_recall = 0
        
        for epoch in tqdm(range(10001)):
            # Training
            self.model.train()
            for batch in train_loader:
                loss = self.train_step(batch)
                self.writer.add_scalar('train_loss', loss, epoch)
                
            # Validation
            if epoch >= 3000 or epoch % 100 == 0:
                recall = self.validate(val_loader)
                print(f"Epoch {epoch}, Recall@3: {recall:.4f}")
                self.writer.add_scalar('validation_recall', recall, epoch)
                
                if recall > best_recall:
                    best_recall = recall
                    self.save_checkpoint(epoch, recall, is_best=True)
                if epoch % 100 == 0:
                    self.save_checkpoint(epoch, recall)
                    
    def train_step(self, batch):
        """Single training step"""
        anchor_img, chosen_img, iou, anchor_embed, chosen_embed = [x.cuda() for x in batch]
        
        anchor_out = F.normalize(self.model(anchor_img, anchor_embed), p=2, dim=1)
        chosen_out = F.normalize(self.model(chosen_img, chosen_embed), p=2, dim=1)
        
        dist = self.criterion(anchor_out, chosen_out)
        loss = iou * dist.pow(2) * 0.5 + \
               (1 - iou) * torch.clamp(Config.MARGIN - dist, min=0).pow(2) * 0.5
        
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()
        
        return loss.mean().item()
    
    def validate(self, loader):
        """Compute validation recall"""
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for batch, embed in loader:
                output = self.model(batch.cuda(), embed.cuda())
                features.extend(F.normalize(output, p=2, dim=1).cpu())
        
        features = torch.stack(features).numpy()
        neighbors = NearestNeighbors(n_neighbors=3).fit(features)
        
        recalls = []
        key_to_idx = {k: i for i, k in enumerate(loader.dataset.keys)}
        
        for anchor, positives in zip(self.val_data['Anchor'], self.val_data['Positive']):
            idx = neighbors.kneighbors(features[key_to_idx[anchor[0]]].reshape(1, -1))[1][0]
            recalls.append(any(key_to_idx[p] in idx for p in positives))
            
        return sum(recalls) / len(recalls)
    
    def save_checkpoint(self, epoch, recall, is_best=False):
        """Save model checkpoint"""
        prefix = 'best_' if is_best else ''
        path = Config.MODEL_PATH / f'{prefix}model{epoch}_{recall:.4f}.pth'
        torch.save({
            'step': epoch,
            'crossvit_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def get_dataloader(self, is_train=True):
        """Get data loader for training or validation"""
        if is_train:
            dataset = ImageDataset(self.database, self.images, self.embeddings)
        else:
            keys = list(set([val for sublist in self.val_data['Anchor'] + 
                           self.val_data['Positive'] for val in sublist]))
            dataset = ImageDataset(keys, self.images, self.embeddings)
            
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE, 
                         shuffle=is_train, num_workers=multiprocessing.cpu_count())

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()