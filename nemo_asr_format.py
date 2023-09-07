import pandas as pd
import json
import argparse
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_nemo_asr_json_files(csv_filepath, audio_files_path, cer_threshold):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filepath)
    logging.info(f"Total input data count: {len(df)}")
    
    # Filter the DataFrame based on the provided CER threshold
    df_filtered = df[df['cer'] <= cer_threshold]
    logging.info(f"Data count after filtering by CER threshold ({cer_threshold}): {len(df_filtered)}")
    
    # Split the filtered data into 80% training and 20% testing
    train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
    logging.info(f"Data split - Training: {len(train_df)} (80%), Testing: {len(test_df)} (20%)")
    
    def write_entries_to_file(dataframe, filename):
        with open(filename, 'w') as file:
            for _, entry in dataframe.iterrows():
                data = {
                    "audio_filepath": audio_files_path + entry["audio_file_path"].split('/')[-1],
                    "duration": entry["clip_duration"],
                    "text": entry["original_sentence"]
                }
                file.write(json.dumps(data) + '\n')
    
    # Write entries for train and test
    write_entries_to_file(train_df, 'train.json')
    write_entries_to_file(test_df, 'test.json')
    
    logging.info(f"Data written to train.json and test.json successfully!")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Nemo ASR compatible JSON files from CSV data.")
    parser.add_argument("--csv_filepath",  type=str, default="./2_stage_preprocessed_data.csv",  help="Path to the CSV file.")
    parser.add_argument("--audio_files_path", type=str, default="/media/real/data/uzbekvoice/clips",  help="Base path for the audio files.")
    parser.add_argument("--cer_threshold", type=float, default=0.18, help="CER threshold for filtering data.")
    
    args = parser.parse_args()
    
    create_nemo_asr_json_files(args.csv_filepath, args.audio_files_path, args.cer_threshold)
