
# pip install chardet librosa

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()   

filename = "./outputs/Never Gonna Give You Up.mp3"
waveform, sample_rate = librosa.load(filename, sr=16000)

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    processor = WhisperProcessor.from_pretrained("openai/whisper-small.en", token=huggingface_token)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en", token=huggingface_token)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])

