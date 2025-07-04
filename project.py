from kokoro import KPipeline
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np
import torch
import os
import time
import sys
import subprocess
import openai
import ast
from openai import OpenAI

with open("./secrets/openai_api_key.txt") as f:
    openai_api_key = f.read().strip()


SAMPLE_RATE = 16000
OUTPUT_DIR = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()

pipeline = KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M')
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en", token=huggingface_token)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en", token=huggingface_token)

client = OpenAI(api_key=openai_api_key)  

def generate_quiz_questions(topic="general knowledge", num_questions=5):
    prompt = (
        f"Create {num_questions} short quiz questions with one correct keyword answer each "
        f"on the topic of '{topic}'. Return them as a Python list of dictionaries in this format:\n"
        f'[{{"question": "What is...", "keywords": ["answer"]}}, ...]'
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        generated_text = response.choices[0].message.content
        questions = ast.literal_eval(generated_text)
        return questions

    except Exception as error:
        print("Error parsing generated questions:", error)
        return []


#play mp3
def play_mp3(file_path):
    if sys.platform.startswith('darwin'):        
        subprocess.run(['afplay', file_path], check=True)
    else:
        print("Sorry, playback only implemented for MacOS.")

#generate speech
def generate_speech(text, filename):
    audio_out = AudioSegment.empty()
    for _, _, audio in pipeline(text, voice="bm_george"):
        audio_np = audio.detach().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        segment = AudioSegment(data=audio_int16.tobytes(), sample_width=2, frame_rate=24000, channels=1)
        audio_out += segment
    path = os.path.join(OUTPUT_DIR, filename)
    audio_out.export(path, format="mp3")
    return path

#record audio
def record_audio(duration=3):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    for t in range(duration, 0, -1):
        print(f"{t}...", end=" ", flush=True)
        time.sleep(1)
    sd.wait()
    print("\nRecording complete.")
    return audio.flatten()

#transcribe
def transcribe_with_whisper(audio):
    print("Transcribing...")
    input_tensor = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_tensor)
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return result.strip()

#questions and acceptable keyword answers
"""quiz = [
    {
        "question": "Which country has the highest life expectancy? ",
        "keywords": ["Hong Kong"]
    },
    {
        "question": "What company was initially known as Blue Ribbon Sports?",
        "keywords": ["Nike"]
    },
    {
        "question": "What is the most common surname in the United States?.",
        "keywords": ["Smith"]
    },
    {
        "questions": ""
    }
]
"""
#main quiz loop

quiz = generate_quiz_questions(topic="computer science", num_questions=5)

score = 0

for i, q in enumerate(quiz, start=1):
    print(f"\n--- Question {i} ---")

    #speak question
    qfile = generate_speech(q["question"], f"question_{i}.mp3")
    play_mp3(qfile)

    #record + transcribe
    #audio = record_audio()
    #answer = transcribe_with_whisper(audio)
    #print(f"You said: {answer}")

    #check answer
    #if any(k.lower() in answer.lower() for k in q["keywords"]):
    #    score += 1
     #   response = f"That is correct! You have a score of {score}"
    #else:
      #  response = "Sorry, that is incorrect."

    #print(response)
    #rfile = generate_speech(response, f"response_{i}.mp3")
    #play_mp3(rfile)

    MAX_RETRIES = 2
    retries = 0
    user_answer =""

    while retries <= MAX_RETRIES:
        audio = record_audio(5)
        user_answer = transcribe_with_whisper(audio).strip()
        print("Transcribed Answer:", user_answer)

        if len(user_answer.split()) < 2:
            retries += 1
            if retries <= MAX_RETRIES:
                retry_prompt = "Sorry, I didn't catch that. Please try again."
                print(retry_prompt)

                retry_gen = pipeline(retry_prompt, voice="bm_george")
                retry_audio = AudioSegment.empty()
                
                for _, _, audio in retry_gen:
                    audio_np = audio.detach().cpu().numpy()
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    segment = AudioSegment(data=audio_int16.tobytes(), sample_width=2, frame_rate=24000, channels=1)
                    retry_audio += segment
                retry_file = f"{OUTPUT_DIR}/retry_{i}.mp3"
                retry_audio.export(retry_file, format="mp3")
                play_mp3(retry_file)
            
            else:
                user_answer = ""
                print("No valid answer detected after retries.")
                break
        else:
            break

    if user_answer:
        if any(k.lower() in user_answer.lower() for k in q["keywords"]):
            score += 1
            response = f"That is correct! You have a score of {score}"
        else:
            response = "Sorry, that is incorrect."

        print(response)
        rfile = generate_speech(response, f"response_{i}.mp3")
        play_mp3(rfile)


summary = f"You got {score} out of {len(quiz)} questions correct."
print("\nQuiz finished.")
print(summary)
final = generate_speech(summary, "final_score.mp3")
play_mp3(final)
