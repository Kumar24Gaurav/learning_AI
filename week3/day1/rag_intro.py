import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)

model = "llama-3.3-70b-versatile"

# step 1
knowledge_base = {
    "age":"The age of Kumar Gaurav ranjan is 25 year",
    "net_worth": "The net worth of Kumar Gaurav is 200000"
}

# step 2 retrieval
def retrieve_info(question):
    question=question.lower()
    context = []
    if "age" in question:
        context.append(knowledge_base["age"])
    if "net worth" in question:
        context.append(knowledge_base["net_worth"])
    return "\n".join(context)

def ask_llm(question):
    context=retrieve_info(question)
    system_prompt = f"answer only based on this context. do not hallucinate. context: {context}"
    system_message={
        "role":"system",
        "content":system_prompt
    }
    message ={
        "role":"user",
        "content":question
    }
    messages=[system_message, message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer=response.choices[0].message.content
    return answer

question="what is gaurav's age"

print(ask_llm(question))