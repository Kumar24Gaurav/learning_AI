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

def llm_ans(prompt):
    message = {
        "role":"user",
        "content":prompt
    }
    messages = [message]

    response = client.chat.completions.create(model = model, messages=messages)
    ans = response.choices[0].message.content
    return ans

# bad_prompt="""
# This is a user complaint:
# My laptop is not working
# classify this
# """

# print(llm_ans(bad_prompt))

good_prompt="""
#Role:
You are a support assistant at a mobile/laptop company

#Task:
You have to classify the issue in a category

#Constraints:
You have to classify the issue in one of three categories namely billing, technical, return.

#Output Format:
your answer should be in one word only. The one word shoud be one of the categories given in constraints

#Example:
For instance if a user complain says he wants a refund thenthe category is Return

#FallBack:
If the issue is unrelated to any of the categories mentioned in constraints, then the answer should be OTHER

This is user complaint:
i purchased a laptop of acer but in the bill their is some miss calculation
"""

print(llm_ans(good_prompt))