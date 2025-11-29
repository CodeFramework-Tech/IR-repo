from preprocess import preprocess_text
from boolean_index import load_index as load_boolean
from tfidf_index import load_tfidf
from bm25_index import load_bm25
from positional_index import load_index as load_positional
import numpy as np
import json

# Load indexes
boolean_index = load_boolean("index/boolean.pkl")
positional_index = load_positional("index/positional.pkl")
tfidf_data = load_tfidf("index/tfidf.pkl")
bm25 = load_bm25("index/bm25.pkl")

# Load embeddings
embeddings = np.load("index/embeddings.npy")

# Load metadata (document text)
with open("index/metadata.json", "r") as f:
    meta = json.load(f)

docs = meta["documents"]

query = "economic growth in Pakistan"
tokens = preprocess_text(query)

print("\n--- Boolean Results ---")
print([docs[i] for i in boolean_index.get(tokens[0], [])])

print("\n--- TF-IDF Top 5 ---")
from numpy.linalg import norm
q_vec = tfidf_data["vectorizer"].transform([" ".join(tokens)])
scores = tfidf_data["matrix"] @ q_vec.T
top = scores.toarray().ravel().argsort()[::-1][:5]
for i in top:
    print("-", docs[i])

print("\n--- BM25 Top 5 ---")
bm25_scores = bm25.get_scores(tokens)
top_bm = bm25_scores.argsort()[::-1][:5]
for i in top_bm:
    print("-", docs[i])

print("\n--- Embeddings Top 5 ---")
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
q_emb = model.encode([" ".join(tokens)])
sims = embeddings @ q_emb.T / (norm(embeddings, axis=1) * norm(q_emb))
top_emb = sims.ravel().argsort()[::-1][:5]
for i in top_emb:
    if i < len(docs):
        print("-", docs[i])
    else:
        print(f"- [Index {i} out of range]")
