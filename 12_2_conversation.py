from openai import OpenAI 

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

response = client.responses.create( 
    model="gpt-4.1-mini", 
    input=[
        {"role": "assistant", "content": "knock knock."},
        {"role": "user", "content": "Who's there?"}, 
        {"role": "assistant", "content": "Annie."}, 
        {"role": "user", "content": "Annie who?"}, 
    ],
) 

print(response.output_text)
