# 📚 RAG Implementation Understanding

A beginner-friendly implementation of **Retrieval-Augmented Generation (RAG)** that demonstrates how to build a document question-answering system using **Sentence Transformers**, **FAISS**, and an **LLM (DeepSeek or Ollama)**.

This project is designed to help beginners understand the complete RAG pipeline from loading documents to generating grounded answers.

---

## 🚀 Features

- Load a text document
- Split the document into chunks
- Generate sentence embeddings using **all-MiniLM-L6-v2**
- Store embeddings in a **FAISS** vector database
- Retrieve the most relevant document chunks
- Generate answers using:
  - **DeepSeek API**
  - **Ollama** (Local LLM)

---

## 📂 Project Structure

```
RAG_Implementation_Understanding/
│
├── rag_legal_bot.py          # Main RAG application
├── company_policy.txt        # Example knowledge document
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Technologies Used

- Python
- Sentence Transformers
- FAISS
- NumPy
- OpenAI Python SDK
- DeepSeek API (Optional)
- Ollama (Optional)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/RAG_Implementation_Understanding.git

cd RAG_Implementation_Understanding
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install sentence-transformers faiss-cpu openai numpy
```

---

## 🤖 Choose an LLM

### Option 1 — DeepSeek API

```python
openai.api_key = "YOUR_API_KEY"
openai.base_url = "https://api.deepseek.com/v1"

MODEL_NAME = "deepseek-chat"
```

---

### Option 2 — Ollama (Local)

Install Ollama:

https://ollama.com

Download a model:

```bash
ollama pull llama3.2
```

Update the configuration:

```python
openai.api_key = "ollama"
openai.base_url = "http://localhost:11434/v1"

MODEL_NAME = "llama3.2"
```

Start Ollama:

```bash
ollama serve
```

---

## ▶️ Run the Project

```bash
python rag_legal_bot.py
```

Example:

```
🤖 RAG Policy Bot is ready!

You:
What is the leave policy?

Bot:
Employees are entitled to 20 annual leaves.
```

---

## 🧠 How It Works

```
                 Document
                     │
                     ▼
            Split into Chunks
                     │
                     ▼
      SentenceTransformer Embeddings
                     │
                     ▼
           Store in FAISS Index
                     │
────────────────────────────────────────
                     │
              User Question
                     │
                     ▼
      Convert Question to Embedding
                     │
                     ▼
      Retrieve Similar Document Chunks
                     │
                     ▼
     Send Context + Question to LLM
                     │
                     ▼
              Generate Answer
```

---

## 📖 What is RAG?

**RAG (Retrieval-Augmented Generation)** combines information retrieval with a language model.

Instead of asking the AI to answer from memory, the system:

1. Searches the document.
2. Finds the most relevant text.
3. Sends only that text to the LLM.
4. Generates an answer based on the retrieved information.

This makes responses more accurate and grounded in your own documents.

---

## 📚 Learning Objectives

This project helps you understand:

- Document chunking
- Sentence embeddings
- Semantic search
- Vector databases (FAISS)
- Retrieval-Augmented Generation (RAG)
- Prompt grounding
- Using DeepSeek and Ollama with Python

---

## 📄 License

This project is intended for educational purposes.