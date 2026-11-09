
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# STEP 1: LOAD THE DOCUMENT
# ==========================================

with open("company_policy.txt", "r", encoding="utf-8") as file:
    document = file.read()


# ==========================================
# STEP 2: SPLIT DOCUMENT INTO CHUNKS
# ==========================================

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Test our chunking function
chunks = split_text(document)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)


# ==========================================
# STEP 3: GENERATE EMBEDDINGS
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)

print("\nNumber of chunks:", len(chunks))
print("Embeddings shape:", chunk_embeddings.shape)
print("First 10 values:", chunk_embeddings[0][:10])


# ==========================================
# OLD CODE: SINGLE-CHUNK RETRIEVAL
# Keeping this for reference
# ==========================================

'''
question = "How much does the company pay for professional certifications?"

# Convert the question into an embedding
question_embedding = model.encode(question)

# Calculate cosine similarity against each chunk
scores = model.similarity(
    question_embedding,
    chunk_embeddings
)[0].numpy()

# Find the most relevant chunk
best_chunk_index = np.argmax(scores)

print("\nQuestion:", question)
print("Best matching chunk:", best_chunk_index + 1)
print("Similarity score:", round(float(scores[best_chunk_index]), 3))
print("\nRetrieved text:")
print(chunks[best_chunk_index])
'''


# ==========================================
# OLD CODE: TOP-3 RETRIEVAL
# Keeping this for reference
# ==========================================

'''
top_k = 3

top_indices = np.argsort(scores)[::-1][:top_k]

print("\nQuestion:", question)
print("\nTop 3 relevant chunks:")

for rank, index in enumerate(top_indices, start=1):
    print(f"\n--- Result {rank} ---")
    print("Chunk:", index + 1)
    print("Similarity:", round(float(scores[index]), 3))
    print(chunks[index])
'''


# ==========================================
# STEP 4: CONNECT TO OPENROUTER
# ==========================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# ==========================================
# OLD CODE: GENERATE ANSWER FOR ONE QUESTION
# Keeping this for reference
# ==========================================

'''
# Build context from top 3 chunks
context = "\n\n".join(
    f"[Chunk {index + 1}]\n{chunks[index]}"
    for index in top_indices
)

# Ask the LLM to answer using retrieved context
prompt = f"""
You are an AI assistant for Northstar Technologies.

Answer the question using ONLY the context provided.

Rules:
1. Do not invent information.
2. If the answer is missing, say:
   "I couldn't find that information in the documents."
3. Cite the relevant chunk numbers.
4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}
"""

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nAI ANSWER:")
print(response.choices[0].message.content)
'''


# ==========================================
# STEP 5: NEW ASK_QUESTION FUNCTION
# ==========================================

def ask_question(question):

    # 1. Convert the question into an embedding
    question_embedding = model.encode(question)

    # 2. Calculate similarity against all chunks
    scores = model.similarity(
        question_embedding,
        chunk_embeddings
    )[0].numpy()

    # 3. Retrieve top 3 most relevant chunks
    top_k = 3
    top_indices = np.argsort(scores)[::-1][:top_k]

    print("\nTop 3 retrieved chunks:")

    for rank, index in enumerate(top_indices, start=1):
        print(f"\n--- Result {rank} ---")
        print("Chunk:", index + 1)
        print("Similarity:", round(float(scores[index]), 3))

    # 4. Build context from retrieved chunks
    context = "\n\n".join(
        f"[Chunk {index + 1}]\n{chunks[index]}"
        for index in top_indices
    )

    # 5. Create the prompt
    prompt = f"""
You are an AI assistant for Northstar Technologies.

Answer the question using ONLY the provided context.

Rules:
1. Do not invent information.
2. If the context does not contain the answer, say:
   "I couldn't find that information in the documents."
3. Cite the relevant chunk numbers.
4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}
"""

    # 6. Send the prompt to OpenRouter
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # 7. Return the generated answer
    return response.choices[0].message.content



# ==========================================
# STEP 6: INTERACTIVE QUESTION LOOP
# ==========================================

if __name__ == "__main__":
    while True:
        question = input("\nAsk DocuMind (or type 'exit'): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(question)

        print("\nDocuMind AI:")
        print(answer)

