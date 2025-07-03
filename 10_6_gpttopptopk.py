from transformers import pipeline

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    text_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2", token=huggingface_token)

prompt = "Once upon a time there lived"

results = text_generator(
    prompt,
    max_length=50,
    do_sample=True,
    num_return_sequences=3,
    temperature=0.25,
    top_k=50,
    top_p=0.85,
    pad_token_id=text_generator.tokenizer.eos_token_id,
    truncation=True
)

print("-" * 80)     
for i, result in enumerate(results):
    text = result['generated_text'].replace("\n", "")
    print(f"{i}: {text}")
print("-" * 80)     
