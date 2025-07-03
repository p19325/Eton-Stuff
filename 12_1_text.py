from openai import OpenAI 

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

response = client.responses.create( 
	model="gpt-4.1", 
	input="Write a one-sentence bedtime story about a unicorn." 
) 

print(response.output_text)
