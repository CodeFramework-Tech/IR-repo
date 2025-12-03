import pandas as pd
import json
from src.boolean_index import build_boolean_index, save_index
from src.positional_index import build_positional_index, save_index as save_pos
from src.tfidf_index import build_tfidf_matrix, save_tfidf
from src.bm25_index import build_bm25, save_bm25

df = pd.read_csv("data/news.csv")
docs = df["Article"].astype(str).tolist()

docs = docs[:100]

metadata = {"documents": df["Heading"].astype(str).tolist()}  
with open("index/metadata.json", "w") as f:
    json.dump(metadata, f)


print("Building Boolean index...")
boolean_idx = build_boolean_index(docs)
save_index(boolean_idx, "index/boolean.pkl")


print("Building Positional index...")
pos_idx = build_positional_index(docs)
save_pos(pos_idx, "index/positional.pkl")


print("Building TF-IDF index...")
vec, mat = build_tfidf_matrix(docs)
save_tfidf(vec, mat, "index/tfidf.pkl")


print("Building BM25 index...")
bm25, tokens = build_bm25(docs)
save_bm25(bm25, tokens, "index/bm25.pkl")

print("All indexes built successfully!")
