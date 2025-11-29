# src/hybrid_search.py
# Top-level search script that ties everything together.
# Usage (example):
#   python -m src.hybrid_search "inflation pakistan"

import os
import sys
import json
import time
import joblib
import numpy as np

from preprocess import preprocess_text
from boolean_index import build_boolean_index, retrieve_and
from tfidf_index import build_tfidf, score_query_tfidf
from bm25_index import build_bm25, score_query_bm25
from re_ranker import Reranker
from hybrid_search import search
print(search("covid pakistan"))

INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news.csv')

# Helper to load documents from CSV - simple implementation
def load_docs_from_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    # try common column names
    for col in ['content', 'text', 'article', 'body']:
        if col in df.columns:
            return df[col].fillna('').astype(str).tolist()
    # fallback: use first text-like column
    # Use concat of all columns for safety
    return df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist()

def build_all_indexes(docs, save=True):
    print("Building boolean index...")
    boolean_index = build_boolean_index(docs)
    if save:
        joblib.dump(boolean_index, os.path.join(INDEX_DIR, 'boolean.pkl'))

    print("Building TF-IDF...")
    vectorizer, tfidf_matrix = build_tfidf(docs)
    if save:
        joblib.dump(vectorizer, os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
        joblib.dump(tfidf_matrix, os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))

    print("Building BM25...")
    bm25_obj, tokenized_docs = build_bm25(docs)
    if save:
        joblib.dump({'bm25': bm25_obj, 'tokenized': tokenized_docs}, os.path.join(INDEX_DIR, 'bm25.pkl'))

    # Save metadata
    if save:
        meta = {'n_docs': len(docs)}
        with open(os.path.join(INDEX_DIR, 'metadata.json'), 'w') as f:
            json.dump(meta, f)

    return {
        'boolean': boolean_index,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'bm25': bm25_obj,
        'docs': docs
    }

def load_all_indexes():
    boolean_index = joblib.load(os.path.join(INDEX_DIR, 'boolean.pkl'))
    vectorizer = joblib.load(os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
    tfidf_matrix = joblib.load(os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))
    bm25_data = joblib.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    bm25_obj = bm25_data['bm25']
    docs = load_docs_from_csv(DATA_PATH)
    return {
        'boolean': boolean_index,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'bm25': bm25_obj,
        'docs': docs
    }

def hybrid_search(query, indexes, top_k=10):
    """
    Steps:
     1) preprocess query
     2) boolean filter to get candidate set
     3) compute TF-IDF and BM25 on candidates
     4) merge scores (weighted)
     5) optional rerank
    """
    q_tokens = preprocess_text(query)
    all_doc_ids = set(range(len(indexes['docs'])))

    # Boolean filter (reduce candidate set)
    candidates = retrieve_and(q_tokens, indexes['boolean'])
    if not candidates:
        # fallback: use all docs
        candidates = all_doc_ids

    candidates = list(candidates)

    # TF-IDF scores (dense vector)
    tfidf_scores_array = score_query_tfidf(q_tokens, indexes['vectorizer'], indexes['tfidf_matrix'])
    # convert to dict for quick lookup on candidate ids
    tfidf_scores = {i: float(tfidf_scores_array[i]) for i in candidates}

    # BM25 scores
    bm25_scores_arr = score_query_bm25(q_tokens, indexes['bm25'])
    bm25_scores = {i: float(bm25_scores_arr[i]) for i in candidates}

    # Merge scores
    merged = []
    for doc_id in candidates:
        tf = tfidf_scores.get(doc_id, 0.0)
        bm = bm25_scores.get(doc_id, 0.0)
        final = 0.4 * tf + 0.6 * bm
        merged.append((doc_id, final, tf, bm))

    # sort
    merged.sort(key=lambda x: x[1], reverse=True)
    top = merged[:top_k]

    # Optional reranking
    reranker = Reranker()
    results = []
    for doc_id, score, tf, bm in top:
        features = np.array([bm, tf, len(indexes['docs'][doc_id].split()), sum(1 for t in q_tokens if t in indexes['docs'][doc_id])])
        rerank_score = reranker.score(features)
        results.append({
            'doc_id': int(doc_id),
            'title_snippet': indexes['docs'][doc_id][:300].replace('\n', ' '),
            'score_raw': float(score),
            'score_rerank': float(rerank_score)
        })

    # final sort by rerank score (or keep merged if reranker not present)
    results.sort(key=lambda x: x['score_rerank'], reverse=True)
    return results

if __name__ == "__main__":
    # CLI behavior
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        docs = load_docs_from_csv(DATA_PATH)
        os.makedirs(INDEX_DIR, exist_ok=True)
        build_all_indexes(docs)
        print("Indexes built and saved to index/")
        sys.exit(0)

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print("Loading indexes (if not built, run: python -m src.hybrid_search build)")
        idxs = load_all_indexes()
        t0 = time.time()
        res = hybrid_search(q, idxs, top_k=10)
        t1 = time.time()
        print(f"Query time: {t1-t0:.3f}s")
        for r in res:
            print(f"Doc {r['doc_id']} | score={r['score_rerank']:.4f}\n{r['title_snippet']}...\n")
    else:
        print("Usage: python -m src.hybrid_search [build] OR python -m src.hybrid_search \"your query here\"")
