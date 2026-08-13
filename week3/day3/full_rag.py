import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")

model = SentenceTransformer("all-MiniLM-L6-v2") #384 vector

client = Groq(api_key=my_api_key)
groq_model = "llama-3.3-70b-versatile"


documents = [
    "Employees receive 24 days of paid leave per year.",

    "Employees work from the office on Tuesday, Wednesday and Thursday.",

    "Monday and Friday are optional work-from-home days.",

    "Employees receive Rs 3000 per month for gym reimbursement.",

    "Employees can claim Rs 2000 per month for home internet.",

    "Employees have a 90 day notice period."
]

document_embedding = model.encode(documents)

def consine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# Safer design: pass context explicitly
def retrieve(q_embedding, docs, d_embeddings):
    scores = []
    for i, doc_emb in enumerate(d_embeddings):
        score = consine_similarity(q_embedding, doc_emb)
        scores.append((score, docs[i]))
    scores.sort(reverse=True)
    return scores[0]

def ask_llm(question, context):
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
    response = client.chat.completions.create(model=groq_model, messages=messages)
    answer=response.choices[0].message.content
    return answer

query = "how much vacation i can take in one year?"
query_embedding = model.encode(query)

score, context = retrieve(query_embedding, documents, document_embedding)
print(ask_llm(query, context))