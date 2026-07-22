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
prompt1 = "Hi"
prompt2 = "Explain time travel in details in 50 word"
prompt3 = "write a 20 word essay on Machine learniing"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {
    "role":role,
    "content":prompt
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=50
    ) 
    usage = response.usage
    print(f"Prompt: {prompt} --> your tokens: {usage.prompt_tokens} completion_tokens: {usage.completion_tokens} total tokens: {usage.prompt_tokens + usage.completion_tokens} Finish Reasons: {response.choices[0].finish_reason}")
