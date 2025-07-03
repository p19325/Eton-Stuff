import sounddevice as sd
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()   

SAMPLE_RATE = 16000 

def record_audio(duration):
    
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')

    for t in range(duration, 0, -1):
        print(f"{t}...", end=" ", flush=True)
        time.sleep(1)  

    sd.wait()

    print("Recording finished.")
    return audio.flatten()

def transcribe_with_whisper(audio):

    with open("./secrets/huggingface_token.txt", "r") as f:
        huggingface_token = f.read().strip()
        processor = WhisperProcessor.from_pretrained("openai/whisper-small.en", token=huggingface_token)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en", token=huggingface_token)
    
    print("Processing audio...")
    tensor = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features

    print("Transcribing audio...")
    tokens = model.generate(tensor)
    
    transcription = processor.batch_decode(tokens, skip_special_tokens=True)
    return transcription[0]

audio = record_audio(5)

text = transcribe_with_whisper(audio)

print("Transcribed Text:", text)
