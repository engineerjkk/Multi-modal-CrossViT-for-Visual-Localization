import pandas as pd
import pickle
from tqdm import tqdm
from hloc.utils.read_write_model import read_images_binary
from pathlib import Path

# Constants
DATA_PATH = Path('datasets/aachen')
OUTPUT_PATH = Path('DataBase')
QUERIES_PATH = DATA_PATH / 'queries'
SFM_PATH = Path('outputs/aachen/sfm_superpoint+superglue')

def merge_query_files(day_file: Path, night_file: Path, output_file: Path) -> dict:
    """
    Merge day and night query files into a single output file.
    
    Args:
        day_file: Path to day queries file
        night_file: Path to night queries file
        output_file: Path to output merged file
        
    Returns:
        Dictionary containing unique entries
    """
    unique_entries = {}
    
    # Process both day and night files
    for file in [day_file, night_file]:
        with open(file, 'r') as f:
            for line in f.readlines():
                if line := line.strip():
                    key = line.split()[0]
                    unique_entries[key] = line
    
    # Save merged results
    with open(output_file, 'w') as f:
        for key in sorted(unique_entries.keys()):
            f.write(f"{unique_entries[key]}\n")
    
    print(f"Merged files successfully. Total unique entries: {len(unique_entries)}")
    return unique_entries

def create_query_csv(input_file: Path, output_file: Path):
    """Create CSV file containing query image paths"""
    # Extract image paths from input file
    with open(input_file, 'r') as f:
        image_paths = [line.split(' ', 1)[0] for line in f.readlines()]
    
    # Save to CSV
    df = pd.DataFrame(image_paths, columns=['Query'])
    df.to_csv(output_file, index=False)
    print(f"Created query CSV with {len(df)} entries")

def create_database_csv(database_file: Path, images_file: Path, output_file: Path):
    """Create CSV file containing database image paths"""
    # Load necessary files
    with open(database_file, 'rb') as f:
        database = pickle.load(f)
    images = read_images_binary(str(images_file))
    
    # Extract image names
    image_names = [images[entry['Anchor'][0]].name for entry in tqdm(database, 
                                                                    desc="Processing database entries")]
    
    # Save to CSV
    df = pd.DataFrame(image_names, columns=['Anchor'])
    df.to_csv(output_file, index=False)
    print(f"Created database CSV with {len(df)} entries")

def main():
    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Merge query files
    day_queries = QUERIES_PATH / 'day_time_queries_with_intrinsics.txt'
    night_queries = QUERIES_PATH / 'night_time_queries_with_intrinsics.txt'
    merged_queries = QUERIES_PATH / 'all_queries_merged.txt'
    
    unique_entries = merge_query_files(day_queries, night_queries, merged_queries)
    
    # Create query CSV
    create_query_csv(
        input_file=merged_queries,
        output_file=OUTPUT_PATH / 'Query922.csv'
    )
    
    # Create database CSV
    create_database_csv(
        database_file=OUTPUT_PATH / 'DataBase3Column.pkl',
        images_file=SFM_PATH / 'images.bin',
        output_file=OUTPUT_PATH / 'DataBase4324.csv'
    )

if __name__ == "__main__":
    main()