# tests/setup_test_data.py
import shutil
from pathlib import Path

def setup_test_data():
    """Copy a test image to the test data directory."""
    tests_dir = Path(__file__).parent
    test_data_dir = tests_dir / "data" / "test_images"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy an image from your data directory
    source_image = Path("../data/test_images/1.jpg")  # Adjust path as needed
    if source_image.exists():
        shutil.copy(source_image, test_data_dir / "1.jpg")
        print(f"Copied test image to {test_data_dir}")
    else:
        print(f"Source image not found at {source_image}")

if __name__ == "__main__":
    setup_test_data()