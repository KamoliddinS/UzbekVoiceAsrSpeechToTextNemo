import nemo.collections.asr as nemo_asr
import soundfile as sf
import nemo.utils
import librosa
import os

nemo.utils.logging.set_verbosity(nemo.utils.logging.ERROR)  # this is for hiding logs/warnings/infos

# Load the pre-trained or fine-tuned NEMO ASR model
asr_model = nemo_asr.models.EncDecCTCModel.restore_from("asr_model.nemo")


# Function to perform transcription on a single .wav file
def transcribe_single_audio(wav_file):
    # Load audio file
    audio_data, sample_rate = sf.read(wav_file)

    # Ensure the sample rate matches the model's expected input by resampling if necessary
    expected_sample_rate = asr_model.preprocessor._cfg['sample_rate']
    if sample_rate != expected_sample_rate:
        # print(f"Resampling from {sample_rate} Hz to {expected_sample_rate} Hz")
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=expected_sample_rate)

    # Save resampled audio to a temporary file
    resampled_wav_file = "resampled_audio.wav"
    sf.write(resampled_wav_file, audio_data, expected_sample_rate)

    # Get transcription from the resampled audio
    transcription_ = asr_model.transcribe([resampled_wav_file])
    os.remove(resampled_wav_file)

    return transcription_[0]


# Path to your .wav file
wav_file_path = '/media/real/data/uzbekvoice/clips/9ba8dc48-7b33-4def-a889-caef56f818a5/07b465b0529801ea040313b64cce70233b7b94ad7bccbd4d6c57ca4c6584a8e9.wav'

# Get the transcription
transcription = transcribe_single_audio(wav_file_path)
print("Transcription:", transcription)
