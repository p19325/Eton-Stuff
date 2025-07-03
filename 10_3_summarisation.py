from transformers import pipeline

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", token=huggingface_token)

text = """Machine learning (ML) is a field of study in artificial intelligence concerned with 
the development and study of statistical algorithms that can learn from data and generalise to 
unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in 
machine learning, advances in the field of deep learning have allowed neural networks, a class 
of statistical algorithms, to surpass many previous machine learning approaches in performance."""

results = summarizer_pipeline(text, max_length=80, min_length=30)
result = results[0]['summary_text'].replace(' .', '.').strip()

print(result)
