# src/embeddings_expansion.py
"""
Optional embedding-based query expansion.

This module tries to use sentence-transformers + faiss if available.
If not available, it falls back to using WordNet synonyms (weaker).

AI note: I used AI assistance to integrate a safe fallback pattern for FAISS + SBERT usage,
and for code structure. Disclose this use in your report.

To use full embedding mode:
 - pip install sentence-transformers faiss-cpu
 - At index building time, call build_embeddings_index(docs)
"""

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
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pandas as pd

# # Load dataset
# df = pd.read_csv("data/news.csv")

# # Use the correct text column
# docs = df["Heading"].astype(str).tolist()

# # Load model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Encode documents
# print("Generating embeddings...")
# embeddings = model.encode(docs, show_progress_bar=True)

# # Save index
# np.save("index/embeddings.npy", embeddings)

# # Save metadata (doc IDs → heading)
# with open("index/metadata.json", "w") as f:
#     json.dump({"documents": docs}, f)

# print("Embedding index built successfully!")
# import json
# import numpy as np
# import faiss
# import torch
# from transformers import BertTokenizer, BertModel
# import pandas as pd

# # Load dataset
# df = pd.read_csv("data/news.csv")

# # Use the correct text column
# docs = df["Heading"].astype(str).tolist()

# # Initialize BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# def get_bert_embeddings(text):
#     """
#     Given a text, generate BERT embeddings for it.
#     """
#     # Tokenize and get input IDs for BERT
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Take the embedding of the [CLS] token (first token)
#     embeddings = outputs.last_hidden_state[:, 0, :].numpy()
#     return embeddings

# # Generate BERT embeddings for all documents
# print("Generating BERT embeddings...")
# doc_embeddings = np.vstack([get_bert_embeddings(doc) for doc in docs])  # Stack embeddings into a NumPy array

# # Save BERT embeddings to disk
# np.save("index/embeddings.npy", doc_embeddings)

# # Save metadata (doc IDs → heading)
# with open("index/metadata.json", "w") as f:
#     json.dump({"documents": docs}, f)

# print("Embedding index built successfully!")

# # Now, let's create the FAISS index with the BERT embeddings
# dimension = doc_embeddings.shape[1]  # The dimension of your embeddings (e.g., 768 for BERT)
# index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean distance)

# # Add BERT embeddings to the FAISS index
# index.add(doc_embeddings)

# # Save the FAISS index to disk for later use
# faiss.write_index(index, "index/faiss_index.index")

# print("FAISS index built and saved successfully!")
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import faiss  # Import FAISS library

# Load dataset
df = pd.read_csv("data/news.csv")  # Ensure this file exists in your data folder

# Use the correct text column (Here, assuming 'Heading' column contains the documents)
docs = df["Heading"].astype(str).tolist()

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    """
    Given a text, generate BERT embeddings for it.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] token embedding
    return embeddings

# Step 1: Generate BERT embeddings for all documents
print("Generating BERT embeddings...")
doc_embeddings = np.vstack([get_bert_embeddings(doc) for doc in docs])  # Stack embeddings into a NumPy array

# Step 2: Save BERT embeddings to disk
np.save("index/embeddings.npy", doc_embeddings)

# Step 3: Save metadata (doc IDs → heading)
with open("index/metadata.json", "w") as f:
    json.dump({"documents": docs}, f)

print("Embedding index built successfully!")

# Step 4: Now, let's create the FAISS index with the BERT embeddings
dimension = doc_embeddings.shape[1]  # The dimension of your embeddings (e.g., 768 for BERT)
index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean distance)

# Step 5: Add BERT embeddings to the FAISS index
index.add(doc_embeddings)

# Step 6: Save the FAISS index to disk for later use
faiss.write_index(index, "index/faiss_index.index")

print("FAISS index built and saved successfully!")
