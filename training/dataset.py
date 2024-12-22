# training/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CarDamageDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load annotations
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        
        # Create image_id to filename mapping
        self.id_to_filename = {
            img['id']: img['file_name'] 
            for img in self.annotations['images']
        }
        
        # Create image_id to labels mapping
        self.image_to_labels = {}
        for img in self.annotations['images']:
            self.image_to_labels[img['id']] = torch.zeros(5, dtype=torch.float32)
            
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            category_id = ann['category_id']
            self.image_to_labels[img_id][category_id - 1] = 1.0
        
        # Store image ids for __getitem__
        self.image_ids = list(self.id_to_filename.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image_path = self.image_dir / self.id_to_filename[img_id]
        
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        
        # Get labels
        labels = self.image_to_labels[img_id]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']
        
        return image, labels

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')  # Using Python's built-in inf instead of numpy's

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            # Spatial transforms
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=0,
                p=0.5
            ),
            
            # Color transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0
                ),
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            
            # Cutout/dropout for robustness
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=5,
                min_height=8,
                min_width=8,
                p=0.3
            ),
            
            # Normalize and convert to tensor (always applied)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])