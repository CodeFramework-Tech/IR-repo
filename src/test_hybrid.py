
from src.preprocess import preprocess_text
from src.boolean_index import load_index as load_boolean
from src.tfidf_index import load_tfidf
from src.positional_index import load_index as load_positional
import numpy as np
import json
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm


def load_bm25(path):
    with open(path, 'rb') as f:
        bm25_data = pickle.load(f) 
    return BM25Okapi(bm25_data)  


boolean_index = load_boolean("index/boolean.pkl")
positional_index = load_positional("index/positional.pkl")
tfidf_data = load_tfidf("index/tfidf.pkl")
bm25 = load_bm25("index/bm25.pkl")


embeddings = np.load("index/embeddings.npy")


with open("index/metadata.json", "r") as f:
    meta = json.load(f)

docs = meta["documents"]


query = "economic growth in Pakistan"
tokens = preprocess_text(query)

print("\n--- Boolean Results ---")
print([docs[i] for i in boolean_index.get(tokens[0], [])])


print("\n--- TF-IDF Top 5 ---")
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
model = SentenceTransformer('bert-base-nli-mean-tokens') 
q_emb = model.encode([" ".join(tokens)])  
q_emb = q_emb.reshape(1, -1) 


sims = np.dot(embeddings, q_emb.T) / (norm(embeddings, axis=1) * norm(q_emb))

top_emb = sims.ravel().argsort()[::-1][:5]
for i in top_emb:
    if i < len(docs):
        print("-", docs[i])
    else:
        print(f"- [Index {i} out of range]")
