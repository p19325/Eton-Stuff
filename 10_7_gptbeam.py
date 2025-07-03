from transformers import pipeline

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    text_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2", token=huggingface_token)

prompt = "Once upon a time there lived"

results = text_generator(
    prompt,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True,
    num_return_sequences=1,
    pad_token_id=text_generator.tokenizer.eos_token_id,
    truncation=True
)

generated_text = results[0]["generated_text"].replace("\n", "")
print("-" * 80)     
print(generated_text)
print("-" * 80)     
