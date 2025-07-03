from openai import OpenAI
import base64
import os

output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

result = client.images.generate(
    model="gpt-image-1",
    prompt="",
    size="1024x1024",
    quality="medium"    
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open(f"{output_dir}/random.png", "wb") as f:
    f.write(image_bytes)