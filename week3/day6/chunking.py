from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

text = """
We seek stories, essays, poems, and dispatches that embody a strong sense of place: pieces in which the setting is crucial to character, narrative, mood, and language. We receive many submissions about traveling in foreign countries and discourage writers from submitting conventional travelogues in which narrators report on experiences abroad without reflecting on larger themes.
"""

# 1. Fixed-size chunking

fixed = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0,
)

print("\n=== Fixed-size chunking ===")

for i, chunk in enumerate(fixed.split_text(text)):
    print(f"Chunk {i+1}:\n")
    print(chunk)
    print("-" * 40)

# 2. Paragraph-based chunking

paragraph = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0,
)   

print("\n=== Paragraph-based chunking ===")

for i, chunk in enumerate(paragraph.split_text(text)):
    print(f"Chunk {i+1}:\n")
    print(chunk)
    print("-" * 40)

# Recursive chunking

recursive = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

print("\n=== Recursive chunking ===")

for i, chunk in enumerate(recursive.split_text(text)):
    print(f"Chunk {i+1}:\n")
    print(chunk)
    print("-" * 40)

