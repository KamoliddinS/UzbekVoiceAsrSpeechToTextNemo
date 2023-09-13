import os
import argparse
import logging
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file_path):
    name, ext = os.path.splitext(mp3_file_path)
    wav_file_path = "{0}.wav".format(name)
    mp3_sound = AudioSegment.from_mp3(mp3_file_path)
    mp3_sound.export(wav_file_path, format="wav")
    return wav_file_path

def list_files_in_folder(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith('.mp3'):
                mp3_file_path = os.path.join(root, file_name)
                wav_file_path = convert_mp3_to_wav(mp3_file_path)
                os.remove(mp3_file_path)
                count += 1
                logging.info("Converted %d mp3 files to wav files", count)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Argument parsing
    parser = argparse.ArgumentParser(description='Convert MP3 files to WAV format in a specified folder.')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing MP3 files.')
    args = parser.parse_args()

    list_files_in_folder(args.folder_path)