# src/test_embeddings.py

import numpy as np
import json

# Load metadata
with open("index/metadata.json", "r") as f:
    meta = json.load(f)

docs = meta["documents"]      # list of headings

# Load embeddings
emb = np.load("index/embeddings.npy")

from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "economic growth"
q_emb = model.encode([query])

# Compute cosine similarity
sims = emb @ q_emb.T / (norm(emb, axis=1) * norm(q_emb))

# Top 5 most similar docs
top_ids = sims.ravel().argsort()[::-1][:5]

print("\nTop 5 relevant documents:")
for i in top_ids:
    if i < len(docs):
        print("-", docs[i])
    else:
        print(f"- [Index {i} out of range]")
