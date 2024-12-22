# training/config.py
from pathlib import Path

class TrainingConfig:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "car_damage_dataset"
    
    # Dataset paths
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    
    # Annotation paths
    TRAIN_ANNO = TRAIN_DIR / "COCO_mul_train_annos.json"
    VAL_ANNO = VAL_DIR / "COCO_mul_val_annos.json"
    TEST_ANNO = TEST_DIR / "COCO_mul_test_annos.json"
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-5
    
    # Model checkpoints
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Create directories if they don't exist
TrainingConfig.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)