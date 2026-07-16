import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# --- CONFIGURATION ---
# Initialize the OpenAI client with DeepSeek's base URL
client = OpenAI(
    api_key="sk-016752f903c349febf4ddc804abfab6c",  # Replace with your actual key
    base_url="https://api.deepseek.com"  # Correct base URL (no /v1 needed)
)
MODEL_NAME = "deepseek-chat"  # The model name for DeepSeek

# --- 1. LOAD DOCUMENT ---
with open("company_policy.txt", "r") as f:
    raw_text = f.read()

# Split into chunks (simple split by double newlines)
chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
print(f"📄 Loaded {len(chunks)} document chunks.")

# --- 2. GENERATE EMBEDDINGS ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)

# --- 3. BUILD VECTOR INDEX (FAISS) ---
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print("🔍 Vector index built successfully!")

# --- 4. RETRIEVAL FUNCTION ---
def retrieve(query, top_k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# --- 5. GENERATION FUNCTION (Calls DeepSeek) ---
def ask_question(query):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query)
    context = "\n\n".join(retrieved_chunks)
    
    # Build the prompt with grounding
    prompt = f"""You are a strict HR assistant. Answer the user's question ONLY based on the provided policy text below. 
If the answer cannot be found in the text, say "I don't have that information in the policy."

### Policy Text:
{context}

### User Question: {query}

### Answer:"""
    
    # Call DeepSeek API using the new client
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. RUN THE BOT (Interactive) ---
if __name__ == "__main__":
    print("\n🤖 RAG Policy Bot is ready! Ask questions about the company policy.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        print("Bot: ", end="")
        answer = ask_question(user_input)
        print(answer + "\n")