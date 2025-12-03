# src/bm25_index.py
# BM25 index builder
import joblib
from rank_bm25 import BM25Okapi
from src.preprocess import preprocess_text  

def build_bm25(docs):
   
    tokenized_docs = [preprocess_text(d) for d in docs]  
    bm25_obj = BM25Okapi(tokenized_docs)  
    return bm25_obj, tokenized_docs

def save_bm25(bm25, tokenized_docs, path="index/bm25.pkl"):
    joblib.dump({"bm25": bm25, "docs": tokenized_docs}, path)

def load_bm25(path="index/bm25.pkl"):
    return joblib.load(path)

def score_query_bm25(query_tokens, bm25_obj):
    return bm25_obj.get_scores(query_tokens)
