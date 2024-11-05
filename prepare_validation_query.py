import pickle
from tqdm import tqdm

# Load database
print("Loading database...")
with open('DataBase/DataBase3Column.pkl', 'rb') as f:
    database = pickle.load(f)

# Transform data structure
print("Processing data...")
validation_data = {'Anchor': [], 'Positive': []}

for entry in tqdm(database):
    validation_data['Anchor'].append([entry['Anchor'][0]])
    validation_data['Positive'].append(entry['Positive'])

# Save results
print("Saving results...")
with open('ValidationSet/ValidationAll_Teacher_Key_Real.pickle', 'wb') as f:
    pickle.dump(validation_data, f, pickle.HIGHEST_PROTOCOL)

print(f"Done! Total entries: {len(validation_data['Anchor'])}")