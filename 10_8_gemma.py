from transformers import pipeline

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    pipe = pipeline("text-generation", model="google/gemma-3-1b-it", token=huggingface_token)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem about chocolate cake"},]
        },
    ],
]

responses = pipe(messages, max_new_tokens=100)

for response in responses[0][0]['generated_text']:
    if response['role'] == 'assistant':        
        text = response['content'].replace('\\n', '\n')
        print("-" * 80)        
        print(text)        
        print("-" * 80)


