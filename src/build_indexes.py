# src/build_indexes.py

import pandas as pd
from boolean_index import build_boolean_index, save_index
from positional_index import build_positional_index, save_index as save_pos
from tfidf_index import build_tfidf_matrix, save_tfidf
from bm25_index import build_bm25_index, save_bm25

# Load dataset
df = pd.read_csv("data/news.csv")
docs = df["Article"].astype(str).tolist()


# Boolean
print("Building Boolean index...")
boolean_idx = build_boolean_index(docs)
save_index(boolean_idx, "index/boolean.pkl")

# Positional
print("Building Positional index...")
pos_idx = build_positional_index(docs)
save_pos(pos_idx, "index/positional.pkl")

# TF-IDF
print("Building TF-IDF index...")
vec, mat = build_tfidf_matrix(docs)
save_tfidf(vec, mat, "index/tfidf.pkl")

# BM25
print("Building BM25 index...")
bm25, tokens = build_bm25_index(docs)
save_bm25(bm25, tokens, "index/bm25.pkl")

print("All indexes built successfully!")
