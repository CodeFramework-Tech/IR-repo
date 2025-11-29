# # src/embeddings_expansion.py
# """
# Optional embedding-based query expansion.

# This module tries to use sentence-transformers + faiss if available.
# If not available, it falls back to using WordNet synonyms (weaker).

# AI note: I used AI assistance to integrate a safe fallback pattern for FAISS + SBERT usage,
# and for code structure. Disclose this use in your report.

# To use full embedding mode:
#  - pip install sentence-transformers faiss-cpu
#  - At index building time, call build_embeddings_index(docs)
# """

# import os
# import numpy as np

# USE_FAISS = False
# USE_SBERT = False

# try:
#     import faiss
#     USE_FAISS = True
# except Exception:
#     USE_FAISS = False

# try:
#     from sentence_transformers import SentenceTransformer
#     USE_SBERT = True
# except Exception:
#     USE_SBERT = False

# from nltk.corpus import wordnet
# from preprocess import preprocess_text

# def build_embeddings_index(docs, model_name='all-MiniLM-L6-v2', save_path=None):
#     """
#     Builds embeddings for each document and (optionally) a FAISS index.
#     Returns: embeddings (n x d), optionally faiss_index
#     """
#     texts = [" ".join(preprocess_text(d)) for d in docs]
#     if not USE_SBERT:
#         raise RuntimeError("sentence-transformers not installed. Install it for embedding mode.")
#     model = SentenceTransformer(model_name)
#     embeddings = model.encode(texts, show_progress_bar=True)
#     if USE_FAISS:
#         dim = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dim)
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings.astype('float32'))
#         if save_path:
#             faiss.write_index(index, save_path + '.index')
#             np.save(save_path + '_emb.npy', embeddings)
#         return embeddings, index
#     else:
#         if save_path:
#             np.save(save_path + '_emb.npy', embeddings)
#         return embeddings, None

# def load_embeddings(save_path):
#     import numpy as np
#     emb = np.load(save_path + '_emb.npy')
#     if USE_FAISS:
#         import faiss
#         idx = faiss.read_index(save_path + '.index')
#         return emb, idx
#     return emb, None

# def expand_query_with_embeddings(query_tokens, embeddings, faiss_index=None, model_name='all-MiniLM-L6-v2', top_k=5):
#     """
#     Given processed query tokens, compute embedding and return list of expansion tokens (words).
#     If FAISS+SBERT available, return nearest neighbor documents' most common tokens as expansion.
#     Otherwise fall back to WordNet synonyms.
#     """
#     # fallback: WordNet synonyms of each token
#     if not USE_SBERT:
#         syns = []
#         for t in query_tokens:
#             for syn in wordnet.synsets(t):
#                 for lemma in syn.lemmas():
#                     name = lemma.name().replace('_', ' ')
#                     if name != t:
#                         syns.append(name)
#         return list(dict.fromkeys(syns))[:10]

#     # If SBERT present:
#     model = SentenceTransformer(model_name)
#     qtext = " ".join(query_tokens)
#     q_emb = model.encode([qtext])
#     if faiss_index is not None:
#         import numpy as np
#         faiss.normalize_L2(q_emb)
#         D, I = faiss_index.search(q_emb.astype('float32'), top_k)
#         # I is index of nearest docs
#         # gather tokens from these docs and pick most common tokens
#         # NOTE: to keep simple, return top_k tokens from these docs
#         expansion_tokens = []
#         for idx in I[0]:
#             # This requires the original tokenized docs available in memory in caller.
#             expansion_tokens.append(str(idx))
#         return expansion_tokens
#     else:
#         # if no faiss, compute similarity via dot product
#         emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
#         q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
#         sims = (emb_norm @ q_norm.T).ravel()
#         ids = sims.argsort()[::-1][:top_k]
#         expansion_tokens = [str(i) for i in ids]
#         return expansion_tokens
# src/embeddings_expansion.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load dataset
df = pd.read_csv("data/news.csv")

# Use the correct text column
docs = df["Heading"].astype(str).tolist()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode documents
print("Generating embeddings...")
embeddings = model.encode(docs, show_progress_bar=True)

# Save index
np.save("index/embeddings.npy", embeddings)

# Save metadata (doc IDs → heading)
with open("index/metadata.json", "w") as f:
    json.dump({"documents": docs}, f)

print("Embedding index built successfully!")
