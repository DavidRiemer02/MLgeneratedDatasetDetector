import os
import argparse
from RandomForestTraining import classify_new_dataset
from MultipleRandomForestTraining import classify_new_dataset_multiple_models
from Utils.benford_analysis import benford_deviation as bd
from Utils.zipf_analysis import zipf_correlation as zc

def main():
    parser = argparse.ArgumentParser(description='Classify a folder of datasets as Real or Fake.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing dataset CSV files')
    args = parser.parse_args()
    
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder '{args.folder_path}' not found.")
        return
    
    for file_name in os.listdir(args.folder_path):
        file_path = os.path.join(args.folder_path, file_name)
        if file_name.endswith('.csv') and os.path.isfile(file_path):
            print(f"Processing {file_name}...")
            classify_new_dataset(file_path)
            print("Now Processing with Multiple Models...")
            classify_new_dataset_multiple_models(file_path)

if __name__ == "__main__":
    main()