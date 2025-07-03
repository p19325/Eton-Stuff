from openai import OpenAI
import os

output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="fable",
    input="I just love learning about artificial intelligence!",
    instructions="Speak in a cheerful and positive tone.",
) as response:
    response.stream_to_file(f"{output_dir}/speech.mp3")