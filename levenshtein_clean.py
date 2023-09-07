import os
import sys
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import nemo.collections.asr as nemo_asr

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_inference_result(audio_filepath, model):
    if not os.path.isfile(audio_filepath):
        logging.error(f"Audio file {audio_filepath} does not exist.")
        return None    
    
    logging.info(f"Performing inference on {audio_filepath}...")
    return model.transcribe(paths2audio_files=[audio_filepath])[0]

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def replace_multi_chars(text):
    text = text.replace("o‘", "1")
    text = text.replace("g‘", "2")
    text = text.replace("sh", "3")
    text = text.replace("ng", "4")
    text = text.replace("ch", "5")
    text = text.replace("‘", "'")
    return text

def remove_punctuations(text):
    punctuations = ['.', ',', '!', '?', '-', '–', '—', '(', ')', '[', ']', '{', '}', ':', ';', '"', "'", '«', '»', '…', '‒', '‐', '‑', '‹', '›', '⁃', '⁻', '₋', '−', '–', '—', '―', '⁓', '〜', '〰', '–', '—', '―', '⁓', '〜']
    for punctuation in punctuations:
        text = text.replace(punctuation, "")
    return text


def calculate_error_rates(real_transcription, model_transcription):
    cer = levenshtein_distance(real_transcription, model_transcription) / len(real_transcription)
    ser = 0 if real_transcription == model_transcription else 1
    return cer, ser

def main(args):
    logger = setup_logger()
    logger.info("Loading model...")
    quartznet_saved = nemo_asr.models.EncDecCTCModel.restore_from(args.model_path)

    logger.info("Reading CSV file...")
    df = pd.read_csv(args.input_csv)
    df['audio_file_path'] = df.apply(lambda row: os.path.join(args.audio_files_dir, row['client_id'], f"{row['original_sentence_id']}.wav"), axis=1)
    
    logging.info("Replacing multi characters...")
    df['original_sentence'] = df['original_sentence'].apply(lambda x: replace_multi_chars(x))

    logging.info("Removing punctuations...")
    df['original_sentence'] = df['original_sentence'].apply(lambda x: remove_punctuations(x))

    logger.info("Performing inference on audio files...")
    df['inference_result'] = df['audio_file_path'].apply(lambda x: get_inference_result(x, quartznet_saved))

    logging.info("Removing NaN values for inference results or original sentences...")
    df = df.dropna(subset=['inference_result', 'original_sentence'])

    logger.info("Calculating error rates...")
    df['cer'] = df.apply(lambda row: calculate_error_rates(row['original_sentence'], row['inference_result'])[0], axis=1)

    logger.info(f"Saving results to {args.output_csv}...")
    df.to_csv(args.output_csv, index=False)
    logger.info("Process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files and calculate error rates.")
    parser.add_argument("--input_csv", default= './1_stage_preprocessed_data.csv', help="Path to the input CSV file.")
    parser.add_argument("--audio_files_dir", default= '/media/real/data/uzbekvoice/clips', help="Directory containing audio files.")
    parser.add_argument("--output_csv", default= './2_stage_preprocessed_data.csv', help="Path to save the output CSV file.")
    parser.add_argument("--model_path", default='saved_model/quartznet15x5.pt', help="Path to the saved model.")
    args = parser.parse_args()
    main(args)
