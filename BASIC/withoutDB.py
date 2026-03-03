from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- STEP 1: Our text ----
text = """
Artificial intelligence is transforming the world. 
Machine learning is a subset of AI that learns from data. 
Deep learning uses neural networks with many layers.
Natural language processing helps computers understand human language.
Computer vision allows machines to interpret images.
Reinforcement learning trains agents through rewards and penalties.
AI is being used in healthcare, finance, and education.
Self driving cars use a combination of computer vision and AI.
"""

# ---- STEP 2: Chunk it ----
chunk_size = 100
overlap = 20
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i+chunk_size].strip()
    if chunk:
        chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")
print()

# ---- STEP 3: Embed each chunk ----
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)

print(f"Each chunk embedded into {len(chunk_embeddings[0])} numbers")
print()

# ---- STEP 4: Ask a question ----
query = "how do machines understand images?"
query_embedding = model.encode([query])

# ---- STEP 5: Find most relevant chunk ----
print(f"Query: '{query}'")
print()
print("Searching chunks...\n")

results = []
for i, chunk in enumerate(chunks):
    score = cosine_similarity(query_embedding, [chunk_embeddings[i]])[0][0]
    results.append((score, chunk))

# Sort by highest score
results.sort(reverse=True)

print("Top 3 most relevant chunks:")
for i, (score, chunk) in enumerate(results[:3]):
    print(f"\nRank {i+1} — Score: {score:.4f}")
    print(f"  '{chunk}'")