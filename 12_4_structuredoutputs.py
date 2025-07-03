from openai import OpenAI
from pydantic import BaseModel

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4.1-nano",
    input=[
        {
            "role": "system", 
            "content": "Extract the event information."
        },
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent,
)

print(response.output_parsed)
