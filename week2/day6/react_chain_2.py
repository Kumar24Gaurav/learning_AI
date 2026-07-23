import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import re
from time import sleep

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError ("api error")

client = Groq(api_key = my_api_key)
model = model = "llama-3.3-70b-versatile"

#tools
def get_hotel_price(hotel):
    if hotel == "three star":
        return 35000
    if hotel == "five star":
        return 65000

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calculation error"

tools = {
    "get_hotel_price": get_hotel_price,
    "calculator": calculator
}

system_prompt = """
you are a travel assitant

you have following tools:
get_hotel_price(hotel)
calculator(expression)

IMPORTANT:
Call tools exactly like these examples:
Action: get_hotel_price("three star")
Action: calculator("5000 - 2500")

Never write:
calculator(expression="5000-2500")
get_hotel_price(hotel="five star")

Follow these rules:
1. Decide what you need to do next
2. Call only one tool at a time
3. After writing an action.
4. Never guess or invert a tool result.
5. wait untill you receive an observation.
6. Then decide your next action.
7. when the task is complete, give the final answer.

format:
Thought: what you need to do
Action: tool_name(argument)

When finished:
Final Answer: your answer
"""

def run_agent(question):
    messages = [
        {
            "role":"system",
            "content":system_prompt
        },
        {
            "role":"user",
            "content":question
        }
    ]

    for step in range(5):
        print("\n-------------------")
        print("STEP", step + 1)
        print("\n-------------------")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content

        print(answer)

        if "Final Answer:" in answer:
            break

        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)
            else:
                observation = "Tool not found"

            print(
                "Observation:", observation
            )

            # add llm response to memory
            messages.append({
                "role": "assistant",
                "content": answer
            })

            # send tool result to llm
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
            sleep(5)  # wait for 5 seconds before next step



prompt = """
I am going to dubai i want to stay in three star hotel
my budget is 90000 rupee, after booking hotel how much amount will left in my bank.
"""
run_agent(prompt)