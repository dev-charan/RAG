from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# ---- CONFIG ----
GROQ_API_KEY = "api"
grok_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ---- STEP 1: Load model and ChromaDB ----
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="my_rag")

# ---- STEP 2: Ask a question ----
query = "how do machines understand images?"
query_embedding = model.encode([query]).tolist()

# ---- STEP 3: Search ChromaDB ----
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

retrieved_chunks = results["documents"][0]
print("Retrieved chunks:")
for i, chunk in enumerate(retrieved_chunks):
    print(f"  Chunk {i+1}: '{chunk}'")
print()

# ---- STEP 4: Call Grok ----
context = "\n".join(retrieved_chunks)

prompt = f"""
You are a helpful assistant.
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer:
"""

response = grok_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
)
print(f"Question: {query}")
print()
print(f"Answer: {response.choices[0].message.content}")