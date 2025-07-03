from openai import OpenAI
import base64

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

data_url = f"data:image/jpeg;base64,{encode_image("./datasets/input_image.jpg")}"

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": [
                { 
                    "type": "text", 
                    "text": "What's in this image?" 
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ],
)

print(completion.choices[0].message.content)