from transformers import pipeline

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    sentiment_pipeline = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis", token=huggingface_token)

data = ["I love cake", "I hate rain"]

results = sentiment_pipeline(data)

for i, result in enumerate(results):
    print (f"{i} - {data[i]}: {result}") 
