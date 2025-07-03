from openai import OpenAI

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

prompt = "Write some Python code to calculate Pi using an infinite series."

response = client.chat.completions.create(
    model="o4-mini",
    reasoning_effort="medium",
    messages=[
        {
            "role": "user", 
            "content": prompt
        }
    ]
)

print(response.choices[0].message.content)



