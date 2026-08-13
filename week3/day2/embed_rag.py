import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

import numpy as np
from sentence_transformers import SentenceTransformer

def consine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    

model = SentenceTransformer("all-MiniLM-L6-v2") #384 vector

# text = "Machine learning is fun."

# embedding = model.encode(text)
# print(embedding.shape)
# print(embedding[:10])

t1 = "win"
t2 = "loose"

e1 = model.encode(t1)
print(e1.shape)
print(e1[:10])
e2 = model.encode(t2)
print(e2.shape)
print(e2[:10])
print("Cosine Similarity: ",consine_similarity(e1,e2))