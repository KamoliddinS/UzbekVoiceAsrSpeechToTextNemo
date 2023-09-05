

# Uzbek Speech-to-Text using Nvidia NeMo ASR

This repository contains code and resources to train a speech-to-text model using the Uzbek voice dataset with Nvidia NeMo's Automatic Speech Recognition (ASR) toolkit.

## Prerequisites

- A machine with an NVIDIA GPU.
- Conda environment manager.
- Python 3.9.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KamoliddinS/UzbekvoiceAsrTextToSpeechNemo.git
   cd UzbekvoiceAsrTextToSpeechNemo
   ```

2. **Set Up a Conda Environment**:
   ```bash
   conda create --name nemo_asr_uzbek python=3.9
   conda activate nemo_asr_uzbek
   ```

3. **Install Dependencies**:
   ```bash
   pip install nemo_toolkit[asr]
   ```

   > Note: You might need to install additional dependencies based on your specific requirements.

<!-- ## Usage

1. **Prepare the Dataset**:
   - Place your Uzbek voice dataset in the `data/` directory.
   - Ensure the dataset is in the required format for NeMo ASR.

2. **Train the Model**:
   ```bash
   python train_asr.py
   ```

3. **Evaluate and Use the Model**:
   - Once training is complete, you can use the trained model for inference on new audio samples.
   - Use the provided `inference.py` script to transcribe audio files:
     ```bash
     python inference.py --audio_path path_to_audio.wav
     ``` -->

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
