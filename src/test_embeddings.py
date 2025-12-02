# from sentence_transformers import SentenceTransformer
# import numpy as np
# from numpy.linalg import norm
# import json

# # Load the Sentence-Transformer model (this gives 384-dimensional embeddings)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load the dataset (you may already have a list of documents)
# docs = ["Document 1", "Document 2", "Document 3"]  # Example documents

# # Generate the embeddings for the documents
# embeddings = model.encode(docs)  # This will generate 384-dimensional embeddings

# # Save the embeddings to a file
# np.save("index/embeddings.npy", embeddings)  # Save the document embeddings

# # Example query
# query = "economic growth"
# q_emb = model.encode([query])  # Query embedding (384-dimensional)

# # Print the shapes of the embeddings to verify their dimensions
# print("Shape of document embeddings (emb):", embeddings.shape)  # Should be (n_documents, 384)
# print("Shape of query embedding (q_emb):", q_emb.shape)  # Should be (1, 384)

# # Compute cosine similarity
# sims = embeddings @ q_emb.T / (norm(embeddings, axis=1) * norm(q_emb))

# # Top 5 most similar documents
# top_ids = sims.ravel().argsort()[-5:][::-1]

# # Print the top 5 relevant documents
# print("\nTop 5 relevant documents:")
# for i in top_ids:
#     if i < len(embeddings):
#         print("-", docs[i])  # Print the document corresponding to the index
#     else:
#         print(f"- [Index {i} out of range]")

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

# Load the Sentence-Transformer model (this gives 384-dimensional embeddings)
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')  # Smaller model

# Load the dataset (you may already have a list of documents)
docs = ["Document 1", "Document 2", "Document 3"]  # Example documents

# Generate the embeddings for the documents
embeddings = model.encode(docs)  # This will generate 384-dimensional embeddings
print(f"Document embeddings:\n{embeddings}")  # Debug: check document embeddings

# Save the embeddings to a file
np.save("index/embeddings.npy", embeddings)  # Save the document embeddings

# Example query
query = "economic growth"
q_emb = model.encode([query])  # Query embedding (384-dimensional)
print(f"Query embedding:\n{q_emb}")  # Debug: check query embedding

# Normalize the embeddings and query
doc_norms = norm(embeddings, axis=1, keepdims=True)  # Normalize document embeddings
query_norm = norm(q_emb, axis=1, keepdims=True)  # Normalize query embedding

# Compute cosine similarity (normalize each vector before the dot product)
sims = np.dot(embeddings, q_emb.T) / (doc_norms * query_norm.T)

# Debug: Check the similarity scores
print(f"Similarity scores: {sims}")

# Top 5 most similar documents
top_ids = sims.ravel().argsort()[-min(5, len(docs)):][::-1]  # Get the top N indices

# Print the top relevant documents
print("\nTop relevant documents:")
for i in top_ids:
    print("-", docs[i])  # Print the document corresponding to the index
print(f"Top 5 indices: {top_ids}")
