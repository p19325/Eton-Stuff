from transformers import pipeline

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    ner_pipeline = pipeline('ner', model='dslim/distilbert-NER', token=huggingface_token)

text = "Apple is a technology company based in California. The CEO is called Tim Cook."

results = ner_pipeline(text)

for result in results:
    print(result)
