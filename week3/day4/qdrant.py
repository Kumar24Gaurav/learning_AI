import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

#  CONNECT TO QDRANT

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud")

# CREATE QDRANT COLLECTION

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384


# Delete collection if it already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")


# LOAD OUR KNOWLEDGE

with open("knowledge.txt", "r", encoding="utf-8") as f:
    # doucments = ["line1", "line2", "line3",....]
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(documents)} documents")


# EMBEDDING

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2") 

print("Embedding model ready")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding Size: {len(embeddings[0])}")


# CREATE QDRANT POINTS

points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id=i+1,

        vector=embedding.tolist(),

        payload={
            "text":documents[i]
        }
    )

    points.append(point)


# UPLOAD TO QDRANT

client.upsert(  # upsert = upload + insert => if point already exist upload and if not then insert this point
    collection_name=COLLECTION_NAME,
    points=points
)


print(f"Uploaded {len(points)} documents to Qdrant")


# SEARCH QDRANT

def search(query, top_k=3):
    # Convert the question into an  embedding
    query_vector = model.encode(query).tolist()

    # Search qdrant for similar vectors
    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results


# TEST SEARCH

query = "How many vacation days do i get?"

results = search(query, top_k=3)

print("\nSearch results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()


# CONNECT TO GROQ

groq_client = Groq(api_key=GROQ_API_KEY)


# ASK THE LLM

def ask_llm(question, context):
    prompt = f"""
Answer the question using the information provided below.

context: {context}

Question: {question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content


# COMPLETE RAG PIPELINE


question = "How many vacation days do i get?"

results = search(question, top_k=3)

# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)

answer = ask_llm(question, context)
print(answer)
