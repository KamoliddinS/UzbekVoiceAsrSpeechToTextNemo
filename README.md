

# Uzbek Speech-to-Text using Nvidia NeMo ASR

This repository contains code and resources to train a speech-to-text model using the Uzbek voice dataset with Nvidia NeMo's Automatic Speech Recognition (ASR) toolkit.

## Prerequisites

- A machine with an NVIDIA GPU.
- Conda environment manager.
- Python 3.10
- Pytorch 1.13.1 or above

you have to download dataset from [here](https://drive.google.com/drive/folders/18N5i7GD0LmUnNQok6BP3EC8PYov7pZDW) 

you will get clips.zip file and voice_dataset.json file

voice_dataset.json file contains meta data about dataset
clips.zip file contains audio files

unzip clips.zip  and replace INPUT_FILE_PATH, AUDIO_DIR_PATH


you have to download pre_trained model from [here](
https://dri)
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

## Preprocessing Audio Dataset 
```bash
python pre_processing_auidio.py AUDIO_DIR_PATH
```



## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
