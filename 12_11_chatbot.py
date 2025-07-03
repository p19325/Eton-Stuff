from openai import OpenAI 
from colorama import Fore

with open("./secrets/openai_api_key.txt", "r") as f:
	api_key = f.read().strip()
client = OpenAI(api_key=api_key)

intro_text = "How can I help?"

conversation_history = [{"role": "assistant", "content": intro_text}]

def get_next_response(user_text):
    conversation_history.append({"role": "user", "content": user_text})
    response = client.responses.create( 
        model="gpt-4.1-mini", 
        input=conversation_history
    ) 
    conversation_history.append({"role": "assistant", "content": response.output_text})
    return response.output_text

print("-" * 80)
print(Fore.RED +intro_text)
print("(Type a blank line to end)")

while True:
    print(Fore.GREEN)
    user_text = input()
    if user_text == "":
         break
    print()
    print(Fore.RED + get_next_response(user_text))