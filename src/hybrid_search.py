import sys
import os
import json
import time
import joblib
import numpy as np
from src.preprocess import preprocess_text
from src.ltr_model import LearningToRank
from src.boolean_index import build_boolean_index, retrieve_and
from src.bm25_index import build_bm25, score_query_bm25
from src.re_ranker import Reranker
from src.tfidf_index import build_tfidf_matrix, score_query
import faiss
from transformers import BertTokenizer, BertModel
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news.csv')

if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR, exist_ok=True)

# --- Document Loading Functions ---

def load_docs_from_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    for col in ['content', 'text', 'article', 'body']:
        if col in df.columns:
            return df[col].fillna('').astype(str).tolist()
    return df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist()


def load_metadata(metadata_file):
    
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file {metadata_file} not found.")
        return None

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    if "documents" not in metadata:
        print("Error: 'documents' key not found in metadata.")
        return None

    return metadata


def save_metadata(docs, metadata_file):
    meta = {'n_docs': len(docs), 'documents': docs}
    with open(metadata_file, 'w') as f:
        json.dump(meta, f)
    print(f"Metadata saved to {metadata_file}")


metadata_file = os.path.join(INDEX_DIR, "metadata.json")
metadata = load_metadata(metadata_file)


if metadata is None or "documents" not in metadata:
    print("Error: 'documents' key is missing in metadata. Fallback to loading documents from CSV.")
    docs = load_docs_from_csv(DATA_PATH)
    save_metadata(docs, metadata_file) 
else:
    docs = metadata["documents"]


# --- BERT/Embedding Setup ---
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    raise NotImplementedError("BERT model is disabled to prevent segmentation fault.")

index = None 


def search_with_faiss(query, top_k=10):
    print("Warning: Skipping FAISS/BERT search to avoid segmentation fault.")
    return [], []
        
   

def build_all_indexes(docs, save=True):
    print("Building boolean index...")
    boolean_index = build_boolean_index(docs)
    if save:
        joblib.dump(boolean_index, os.path.join(INDEX_DIR, 'boolean.pkl'))

    print("Building TF-IDF...")
    vectorizer, tfidf_matrix = build_tfidf_matrix(docs)
    if save:
        joblib.dump(vectorizer, os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
        joblib.dump(tfidf_matrix, os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))

    print("Building BM25...")
    bm25_obj, tokenized_docs = build_bm25(docs)
    if save:
        joblib.dump({'bm25': bm25_obj, 'tokenized': tokenized_docs}, os.path.join(INDEX_DIR, 'bm25.pkl'))

    if save:
        save_metadata(docs, os.path.join(INDEX_DIR, 'metadata.json'))

    return {
        'boolean': boolean_index,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'bm25': bm25_obj,
        'docs': docs
    }

def load_all_indexes():
    global index 
    
    boolean_index = joblib.load(os.path.join(INDEX_DIR, 'boolean.pkl'))
    vectorizer = joblib.load(os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
    tfidf_matrix = joblib.load(os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))
    bm25_data = joblib.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    bm25_obj = bm25_data['bm25']
    docs = load_docs_from_csv(DATA_PATH)
    
    faiss_file = os.path.join(INDEX_DIR, 'faiss.index')
    if os.path.exists(faiss_file):
        try:
            index = faiss.read_index(faiss_file)
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load FAISS index. Error: {e}")
            index = None
    else:
        print("Warning: FAISS index file not found. FAISS search will be skipped.")

    return {
        'boolean': boolean_index,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'bm25': bm25_obj,
        'docs': docs
    }

# --- Hybrid Search Function ---

def hybrid_search(query, indexes, top_k=10):
    q_tokens = preprocess_text(query)
    all_doc_ids = set(range(len(indexes['docs'])))

    # Step 1: Candidate Generation
    candidates = retrieve_and(q_tokens, indexes['boolean'])
    if not candidates:
        candidates = all_doc_ids
    candidates = list(candidates)

    # CRITICAL FIX: Filters out invalid index (e.g., 100 on size 100 array)
    max_doc_id = len(indexes['docs']) - 1 
    candidates = [i for i in candidates if 0 <= i <= max_doc_id]

    # Step 2: Calculate scores for candidates
    tfidf_scores_array = score_query(q_tokens, indexes['vectorizer'], indexes['tfidf_matrix'])
    tfidf_scores = {i: float(tfidf_scores_array[i]) for i in candidates}

    bm25_scores_arr = score_query_bm25(q_tokens, indexes['bm25'])
    bm25_scores = {i: float(bm25_scores_arr[i]) for i in candidates}

    # Step 3: FAISS/Semantic Search
    faiss_docs, faiss_distances = search_with_faiss(query, top_k) 
 
    # Step 4: Feature Extraction for Learning-to-Rank (LTR)
    features = []
    for doc_id in candidates:
        features.append([
            tfidf_scores.get(doc_id, 0.0),
            bm25_scores.get(doc_id, 0.0),
            len(indexes['docs'][doc_id].split()) 
        ])
 
    # Step 5: LTR Model Training and Scoring
    ltr_model = LearningToRank()
    if features:
        ltr_model.train(features, [0 if i % 2 == 0 else 1 for i in range(len(candidates))])  
        ltr_scores = ltr_model.predict(features)
    else:
        ltr_scores = []

    # Step 6: Final Ranking and Result Formatting (LTR results)
    ranked_docs = sorted(zip(candidates, ltr_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_id, score in ranked_docs:
        results.append({
            'doc_id': doc_id,
            'score': score,
            'title_snippet': indexes['docs'][doc_id][:300] 
        })


    return results


if __name__ == "__main__":
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
        print(f"Query time: {t1 - t0:.3f}s")
        for r in res:
            print(f"Doc {r['doc_id']} | score={r['score']:.4f}\n{r['title_snippet']}...\n")
    else:
        print("Usage: python -m src.hybrid_search [build] OR python -m src.hybrid_search \"your query here\"")