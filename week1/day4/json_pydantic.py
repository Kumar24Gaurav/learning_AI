import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError ("api error")

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"

from pydantic import BaseModel
class Ticket(BaseModel):
    name:str
    email:str
    address:str
    phone:int
    issue:str

schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract the personal information from the ticket strictly based on this {schema} and give a json output. 
"""

message_system = {
    "role":"system",   
    "content":system_prompt
}

text = "Hello my name is Gaurav. Yesterday i brokeup with my girlfriend Sheetal. I have motorola phone which is not working at all. My address is Gaya ji Bihar. My email is abc@example.com. My contact number is 81236"
prompt = f"This is a customer ticket. Please extract the personal information from this {text}"

message = {
    "role":"user",
    "content":prompt
}

messages = [message_system, message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)

answer = response.choices[0].message.content
print(answer)


# how to read the json output from the assistant
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.address)