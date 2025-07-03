import datetime
import sounddevice as sd
import soundfile as sf
import subprocess
import sys
import os
from openai import OpenAI

DURATION_SECONDS = 5         
SAMPLE_RATE = 44100          
CHANNELS = 1                 
VOICE = "fable"            
INSTRUCTIONS = "Speak in a jamaican accent and give a reply to my greeting."  

def get_timestamp():    
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def record_audio(duration, fs, channels):    
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    return recording

def save_wav(data, fs, filename):    
    sf.write(filename, data, fs)

def transcribe_audio(client, audio_path):    
    with open(audio_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
    return resp.text

def generate_tts(client, text, voice, instructions, out_path):    
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(out_path)

def play_mp3(path):
    if sys.platform.startswith('darwin'):        
        subprocess.run(['afplay', path], check=True)
    else:
        print("Sorry, playback only implemented for MacOS.")

if __name__ == "__main__":
    
    output_dir = "./outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open("./secrets/openai_api_key.txt", "r") as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key)
    
    ts = get_timestamp()
    wav_file = f"{output_dir}/{ts}.wav"
    txt_file = f"{output_dir}/{ts}.txt"
    mp3_file = f"{output_dir}/{ts}.mp3"
    
    audio_data = record_audio(DURATION_SECONDS, SAMPLE_RATE, CHANNELS)
    save_wav(audio_data, SAMPLE_RATE, wav_file)
    print(f"Recording saved to {wav_file}")
    
    print("Transcribing audio...")
    transcript = transcribe_audio(client, wav_file)
    with open(txt_file, "w") as f:
        f.write(transcript)
    print(f"Transcript saved to {txt_file}")

    print("Generating speech... 📢")
    generate_tts(client, transcript, VOICE, INSTRUCTIONS, mp3_file)
    print(f"TTS audio saved to {mp3_file}")
    
    print("Playing back your parroted voice... 🦜")
    play_mp3(mp3_file)