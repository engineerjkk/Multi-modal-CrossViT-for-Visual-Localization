from pathlib import Path
import pickle
from tqdm import tqdm

def prepare_validation_data(input_path: str, output_path: str):
    """
    Prepare validation dataset from IoU database
    
    Args:
        input_path: Path to input IoU database pickle file
        output_path: Path to save processed validation data
    """
    # Load database
    print(f"Loading database from {input_path}")
    with open(input_path, 'rb') as f:
        database = pickle.load(f)
    
    # Process data into required format
    print("Processing database entries")
    validation_data = {
        'Anchor': [],
        'Positive': []
    }
    
    for entry in tqdm(database):
        validation_data['Anchor'].append([entry['Anchor'][0]])
        validation_data['Positive'].append(entry['Positive'])
    
    # Save processed data
    print(f"Saving validation data to {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(validation_data, f, pickle.HIGHEST_PROTOCOL)
    
    print("Validation data preparation completed")
    print(f"Total entries: {len(validation_data['Anchor'])}")

def main():
    """Main execution function"""
    PATHS = {
        'input': 'DataBase/output_iou_database.pkl',
        'output': 'ValidationSet/TeacherVal.pickle'
    }
    
    prepare_validation_data(PATHS['input'], PATHS['output'])

if __name__ == "__main__":
    main()