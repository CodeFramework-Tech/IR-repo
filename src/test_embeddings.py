
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm


model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')  
docs = ["Document 1", "Document 2", "Document 3"] 
embeddings = model.encode(docs)
print(f"Document embeddings:\n{embeddings}")  
np.save("index/embeddings.npy", embeddings)  
query = "economic growth"
q_emb = model.encode([query]) 
print(f"Query embedding:\n{q_emb}")  
doc_norms = norm(embeddings, axis=1, keepdims=True)  
query_norm = norm(q_emb, axis=1, keepdims=True) 
sims = np.dot(embeddings, q_emb.T) / (doc_norms * query_norm.T)
print(f"Similarity scores: {sims}")
top_ids = sims.ravel().argsort()[-min(5, len(docs)):][::-1]  
print("\nTop relevant documents:")
for i in top_ids:
    print("-", docs[i]) 
print(f"Top 5 indices: {top_ids}")
