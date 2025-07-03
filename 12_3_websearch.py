from openai import OpenAI

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{
        "type": "web_search_preview",
        "search_context_size": "low",
    }],
    input="Find a positive UK news story from today and sumarise in one sentence."
)

print(response.output_text)
