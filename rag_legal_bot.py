import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# --- USE OLLAMA (FREE LOCAL LLM) ---
client = OpenAI(
    api_key="ollama",  # Dummy key for local use
    base_url="http://localhost:11434/v1"
)
MODEL_NAME = "tinyllama"  # Make sure you've pulled this: ollama pull llama3.2

# --- 1. LOAD DOCUMENT ---
with open("company_policy.txt", "r") as f:
    raw_text = f.read()

# Split into chunks
chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
print(f"📄 Loaded {len(chunks)} document chunks.")

# --- 2. GENERATE EMBEDDINGS ---
print("⏳ Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)

# --- 3. BUILD VECTOR INDEX ---
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print("🔍 Vector index built successfully!")

# --- 4. RETRIEVAL FUNCTION ---
def retrieve(query, top_k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# --- 5. GENERATION FUNCTION ---
def ask_question(query):
    retrieved_chunks = retrieve(query)
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a strict HR assistant. Answer the user's question ONLY based on the provided policy text below. 
If the answer cannot be found in the text, say "I don't have that information in the policy."

### Policy Text:
{context}

### User Question: {query}

### Answer:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. RUN THE BOT ---
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