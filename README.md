

# Uzbek Speech-to-Text using Nvidia NeMo ASR

This repository contains code and resources to train a speech-to-text model using the Uzbek voice dataset with Nvidia NeMo's Automatic Speech Recognition (ASR) toolkit.

## Prerequisites

- A machine with an NVIDIA GPU.
- Conda environment manager.
- Python 3.10
- Pytorch 1.13.1 or above

you have to download dataset from [here](https://uzbekvoice.ai) 

you will get clips.zip file and voice_dataset.json file

voice_dataset.json file contains meta data about dataset
clips.zip file contains audio files

unzip clips.zip


you have to download pre_trained model from [here](https://drive.google.com/drive/folders/1dq_jXAJqyEeITShxEvrVQumE_3ix4TYW?usp=sharing)
and unzip it and put into the current directory


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KamoliddinS/UzbekvoiceAsrTextToSpeechNemo.git
   cd UzbekvoiceAsrTextToSpeechNemo
   ```
   > You have to download pre_trained model from [here](https://dri) and unzip it and put into the current directory. 

2. **Set Up a Conda Environment**:
   ```bash
   conda create --name nemo_asr_uzbek python==3.10.12
   conda activate nemo_asr_uzbek
   ```
3. **Install prerequisites**:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. **Install NeMo**:
   ```bash
    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']
   ```
   > Note: You might need to install additional dependencies based on your specific requirements.

5. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## Usage
The following steps are required to train a speech-to-text model using the Uzbek voice dataset.

### 1. Data Cleaning - Stage 1

**Script**: `clean_stage_1.py`

- **Input**: `voice_dataset.json`
- **Output**: `1_stage_preprocessed_data.csv`

**Usage**:
```bash
python clean_stage_1.py
```

### 2. Audio Preprocessing

**Script**: `pre_procecessing_auido.py`

- **Input**: Folder path containing the audio files from the uzbekvoice dataset.
- **Function**: Converts `.mp3` files to `.wav` format.

**Usage**:
```bash
python pre_procecessing_auido.py --folder_path /path/to/uzbekvoice/dataset
```
**Note**: Download the uzbekvoice dataset audio files and provide the path to the dataset.

### 3. Levenshtein Cleaning

**Script**: `levenshtain_clean.py`

**Usage**:
```bash
python levenshtain_clean.py --input_csv 1_stage_preprocessed_data.csv --audio_files_dir /path/to/preprocessed/wav/files --output_csv output.csv --model_path /path/to/pretrained/model
```
**Note**: 
- Download the pre-trained model from the provided link, unzip it, and place it in the repository's cloned directory. [DOWNLOAD](https://drive.google.com/drive/folders/1dq_jXAJqyEeITShxEvrVQumE_3ix4TYW?usp=sharing)
- Provide the path to the preprocessed `.wav` files folder.

### 4. NeMo ASR Format Conversion

**Script**: `nemo_asr_format.py`

**Usage**:
```bash
python nemo_asr_format.py --csv_filepath output.csv --audio_files_path /path/to/audio/files --cer_threshold 0.18
```
**Note**: Provide the path to the audio files that were downloaded and preprocessed.

### 5. Model Training

**Script**: `train.py`

**Usage**:
```bash
python train.py --train_json_path train.json --test_json_path test.json --model_name model_name --model_save_path /path/to/save/model --checkpoint True --num_epochs 10
```
**Note**: 
- By default, `nemo_asr_format.py` outputs `train.json` and `test.json`.
- Provide the desired model name and the path where you want to save the trained model.
- The `--checkpoint` flag determines whether to evaluate the model or not.


> By following the above steps, you can preprocess, clean, and train an Uzbek Speech-to-Text model using Nvidia NeMo ASR. Ensure that all the required datasets and pre-trained models are downloaded and placed in the appropriate directories before running the scripts.


## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
