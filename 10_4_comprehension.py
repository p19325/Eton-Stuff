from transformers import pipeline

with open("./secrets/huggingface_token.txt", "r") as f:
    huggingface_token = f.read().strip()
    qa_pipeline = pipeline("question-answering", model='distilbert/distilbert-base-cased-distilled-squad', token=huggingface_token)

result = qa_pipeline(
    question = "Where did George Washington go?",
    context = "George Washington went to Paris in April."
)

answer = result['answer']

print(answer)


