import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from time import sleep

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError ("api error")

client = Groq(api_key = my_api_key)
model="llama-3.3-70b-versatile"

prompt = "explain how internet works in 500 words."
message = {
    "role": "user",
    "content":prompt
}

messages=[message]

# response = client.chat.completions.create(model=model, messages=messages)
# answer = response.choices[0].message.content
# print(answer)

stream = client.chat.completions.create(model=model, messages=messages, stream=True)

for chunk in stream:
    answer = chunk.choices[0].delta.content
    if answer:
        print(answer, end="", flush=True)