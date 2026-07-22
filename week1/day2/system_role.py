import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)

model = "llama-3.3-70b-versatile"

role = "user"
prompt = "suggest only one food brand name for my company"
# SYSTEM
message_system = {
    "role" : "system",
    "content" : "You are a helpful assistant that suggests food brands based on indian cuisine."
}

message = {
    "role":role,
    "content": prompt
}

messages = [message_system, message]

# temperature by default value is 0 (safe) range[0,1,2] more number more randomness.
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=2
)

print("###########################################")

answers = response.choices[0].message.content
print(answers)