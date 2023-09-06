import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import argparse

def filter_out_numbers(text):
    if any(char.isdigit() for char in text):
        return None
    return text

def filter_out_invalid_characters(text):
    alphabet = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'x', 'y', 'z', '‘', '.', ' ']
    for char in text:
        if char not in alphabet:
            return None
    return text

def preprocess_data(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = json.loads(input_file.read())

    df = pd.DataFrame(data)
    df = df.dropna(subset=['clip_duration'])
    df = df[df.original_sentence.apply(filter_out_numbers).notnull()]
    df['original_sentence'] = df['original_sentence'].str.lower()
    df['original_sentence'] = df['original_sentence'].apply(filter_out_invalid_characters)
    df = df[df.original_sentence.notnull()]
    df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess voice dataset")
    parser.add_argument("--input", type=str, default='/media/real/data/uzbekvoice/uzbekvoice-dataset/voice_dataset.json', help="Path to the input JSON file")
    parser.add_argument("--output", type=str, default="./1_stage_preprocessed_data.csv", help="Path to save the preprocessed CSV file")
    args = parser.parse_args()

    preprocess_data(args.input, args.output)
