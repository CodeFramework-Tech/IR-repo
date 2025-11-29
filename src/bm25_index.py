# src/bm25_index.py
# BM25 index builder
import joblib
from rank_bm25 import BM25Okapi
from preprocess import preprocess_text

def build_bm25_index(docs):
    """
    docs: list of strings
    returns: (bm25_object, tokenized_docs)
    """
    tokenized = [preprocess_text(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def save_bm25(bm25, tokenized_docs, path="index/bm25.pkl"):
    joblib.dump({"bm25": bm25, "docs": tokenized_docs}, path)

def load_bm25(path="index/bm25.pkl"):
    return joblib.load(path)
