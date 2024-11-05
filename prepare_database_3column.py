from pathlib import Path
import pickle
from point_cloud_iou_processor import PointCloudProcessor

# Configuration
MIN_COMMON_POINTS = 1
BASE_PATH = Path('outputs/aachen/sfm_superpoint+superglue')
IMAGES_PATH = BASE_PATH / 'images.bin'
POINTS3D_PATH = BASE_PATH / 'points3D.bin'
OUTPUT_PATH = Path('DataBase/DataBase3Column.pkl')

def main():
    """Prepare and save point cloud database"""
    # Create output directory if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor and generate database
    print("Initializing point cloud processor...")
    processor = PointCloudProcessor(
        MIN_COMMON_POINTS,
        str(IMAGES_PATH),
        str(POINTS3D_PATH)
    )
    
    print("Building database...")
    database = processor.build_database()
    
    # Save database
    print(f"Saving database to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(database, f)
    
    print("Database preparation completed")

if __name__ == "__main__":
    main()