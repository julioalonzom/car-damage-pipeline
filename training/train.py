# training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import ViTForImageClassification
from pathlib import Path
import logging
from datetime import datetime

from .dataset import CarDamageDataset, get_transforms, EarlyStopping
from .config import TrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_refined.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MPSCompatibleViT(ViTForImageClassification):
    def forward(self, pixel_values, labels=None):
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        batch_size = pixel_values.shape[0]
        
        outputs = self.vit(
            pixel_values,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            interpolate_pos_encoding=None,
            return_dict=True,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0].reshape(batch_size, -1))

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else logits

def train_model(initial_lr=1e-4):
    try:
        # Setup device - using CPU as per our previous discussion
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        
        # Load model
        logger.info("Loading base model...")
        model = MPSCompatibleViT.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=5,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        
        # Initialize model structure
        hidden_size = model.config.hidden_size
        model.classifier = nn.Linear(hidden_size, 5)
        model = model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        criterion = nn.BCEWithLogitsLoss()
        
        # Load the best checkpoint from previous training
        logger.info("Loading previous best checkpoint...")
        checkpoint = torch.load(str(TrainingConfig.CHECKPOINT_DIR / 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_loss = checkpoint['loss']
        logger.info(f"Loaded checkpoint with validation loss: {prev_loss:.4f}")
        
        # Initialize datasets with enhanced augmentations
        train_dataset = CarDamageDataset(
            TrainingConfig.TRAIN_DIR,
            TrainingConfig.TRAIN_ANNO,
            transform=get_transforms('train')
        )
        
        val_dataset = CarDamageDataset(
            TrainingConfig.VAL_DIR,
            TrainingConfig.VAL_ANNO,
            transform=get_transforms('val')
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=TrainingConfig.BATCH_SIZE, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=TrainingConfig.BATCH_SIZE,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, verbose=True)
        best_val_loss = float('inf')
        
        # Training loop
        logger.info("Starting refined training...")
        
        for epoch in range(TrainingConfig.NUM_EPOCHS):
            model.train()
            train_loss = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["logits"]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs["logits"]
                    val_loss += criterion(outputs, labels).item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            logger.info(f'Epoch: {epoch}')
            logger.info(f'Average Training Loss: {avg_train_loss:.4f}')
            logger.info(f'Average Validation Loss: {avg_val_loss:.4f}')
            
            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f'Validation loss improved to {best_val_loss:.4f}. Saving checkpoint...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, str(TrainingConfig.CHECKPOINT_DIR / 'best_model_refined.pt'))
            
            # Early stopping check
            if early_stopping(avg_val_loss):
                logger.info("Early stopping triggered!")
                break
        
        logger.info("Refined training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_model()