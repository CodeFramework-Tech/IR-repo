
# import sys
# import os
# import json
# import time
# import joblib
# import numpy as np
# from src.preprocess import preprocess_text
# from ltr_model import LearningToRank
# from src.boolean_index import build_boolean_index, retrieve_and
# from src.bm25_index import build_bm25, score_query_bm25
# from src.re_ranker import Reranker
# from src.tfidf_index import build_tfidf_matrix, score_query

# # Define directory paths
# INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
# DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news.csv')

# # Load documents from CSV
# def load_docs_from_csv(path):
#     import pandas as pd
#     df = pd.read_csv(path)
#     for col in ['content', 'text', 'article', 'body']:
#         if col in df.columns:
#             return df[col].fillna('').astype(str).tolist()
#     return df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist()

# # Build indexes (TF-IDF, BM25, Boolean)
# def build_all_indexes(docs, save=True):
#     print("Building boolean index...")
#     boolean_index = build_boolean_index(docs)
#     if save:
#         joblib.dump(boolean_index, os.path.join(INDEX_DIR, 'boolean.pkl'))

#     print("Building TF-IDF...")
#     vectorizer, tfidf_matrix = build_tfidf_matrix(docs)
#     if save:
#         joblib.dump(vectorizer, os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
#         joblib.dump(tfidf_matrix, os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))

#     print("Building BM25...")
#     bm25_obj, tokenized_docs = build_bm25(docs)
#     if save:
#         joblib.dump({'bm25': bm25_obj, 'tokenized': tokenized_docs}, os.path.join(INDEX_DIR, 'bm25.pkl'))

#     # Save metadata
#     if save:
#         meta = {'n_docs': len(docs)}
#         with open(os.path.join(INDEX_DIR, 'metadata.json'), 'w') as f:
#             json.dump(meta, f)

#     return {
#         'boolean': boolean_index,
#         'vectorizer': vectorizer,
#         'tfidf_matrix': tfidf_matrix,
#         'bm25': bm25_obj,
#         'docs': docs
#     }

# def load_all_indexes():
#     boolean_index = joblib.load(os.path.join(INDEX_DIR, 'boolean.pkl'))
#     vectorizer = joblib.load(os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
#     tfidf_matrix = joblib.load(os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))
#     bm25_data = joblib.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
#     bm25_obj = bm25_data['bm25']
#     docs = load_docs_from_csv(DATA_PATH)
#     return {
#         'boolean': boolean_index,
#         'vectorizer': vectorizer,
#         'tfidf_matrix': tfidf_matrix,
#         'bm25': bm25_obj,
#         'docs': docs
#     }

# def hybrid_search(query, indexes, top_k=10):
#     q_tokens = preprocess_text(query)
#     all_doc_ids = set(range(len(indexes['docs'])))

#     # Boolean filter (reduce candidate set)
#     candidates = retrieve_and(q_tokens, indexes['boolean'])
#     if not candidates:
#         candidates = all_doc_ids
#     candidates = list(candidates)

#     # TF-IDF and BM25 scores
#     tfidf_scores_array = score_query(q_tokens, indexes['vectorizer'], indexes['tfidf_matrix'])
#     tfidf_scores = {i: float(tfidf_scores_array[i]) for i in candidates}

#     bm25_scores_arr = score_query_bm25(q_tokens, indexes['bm25'])
#     bm25_scores = {i: float(bm25_scores_arr[i]) for i in candidates}

#     # Feature extraction for LTR
#     features = []
#     for doc_id in candidates:
#         features.append([
#             tfidf_scores.get(doc_id, 0.0),
#             bm25_scores.get(doc_id, 0.0),
#             len(indexes['docs'][doc_id].split())  # Example feature: document length
#         ])
    
#     # Train LTR model (in real use, train first, here we assume pre-trained)
#     ltr_model = LearningToRank()
#     ltr_model.train(features, [0 if i % 2 == 0 else 1 for i in range(len(candidates))])  # Dummy labels

#     # Predict rankings using LTR model
#     ltr_scores = ltr_model.predict(features)

#     # Rank documents based on LTR scores
#     ranked_docs = sorted(zip(candidates, ltr_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
#     results = []
#     for doc_id, score in ranked_docs:
#         results.append({
#             'doc_id': doc_id,
#             'score': score,
#             'title_snippet': indexes['docs'][doc_id][:300]  # Snippet from the document
#         })

#     return results

# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == 'build':
#         docs = load_docs_from_csv(DATA_PATH)
#         os.makedirs(INDEX_DIR, exist_ok=True)
#         build_all_indexes(docs)
#         print("Indexes built and saved to index/")
#         sys.exit(0)

#     if len(sys.argv) > 1:
#         q = " ".join(sys.argv[1:])
#         print("Loading indexes (if not built, run: python -m src.hybrid_search build)")
#         idxs = load_all_indexes()
#         t0 = time.time()
#         res = hybrid_search(q, idxs, top_k=10)
#         t1 = time.time()
#         print(f"Query time: {t1 - t0:.3f}s")
#         for r in res:
#             print(f"Doc {r['doc_id']} | score={r['score']:.4f}\n{r['title_snippet']}...\n")
#     else:
#         print("Usage: python -m src.hybrid_search [build] OR python -m src.hybrid_search \"your query here\"")
# ////////////////
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# import json
# import time
# import joblib
# import numpy as np
# from src.preprocess import preprocess_text
# from ltr_model import LearningToRank
# from src.boolean_index import build_boolean_index, retrieve_and
# from src.bm25_index import build_bm25, score_query_bm25
# from src.re_ranker import Reranker
# from src.tfidf_index import build_tfidf_matrix, score_query
# import faiss  # Import FAISS library
# from transformers import BertTokenizer, BertModel
# import torch
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# # Define directory paths
# INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
# DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news.csv')

# # Load the FAISS index
# index = faiss.read_index("index/faiss_index.index")

# # Load the metadata (the list of documents)
# with open("index/metadata.json", "r") as f:
#     metadata = json.load(f)
# docs = metadata["documents"]
# print(metadata)  # Debugging line to check metadata

# # Initialize BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# def get_bert_embeddings(text):
#     """
#     Given a text, generate BERT embeddings for it.
#     """
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] token embedding
#     return embeddings

# def search_with_faiss(query, top_k=10):
#     """
#     Given a query, search the FAISS index for the most similar documents.
#     """
#     # Step 1: Generate BERT embeddings for the query
#     query_embedding = get_bert_embeddings(query)

#     # Step 2: Perform FAISS search (Find top_k nearest neighbors)
#     distances, indices = index.search(query_embedding, top_k)  # Returns distances and indices of the closest documents

#     # Step 3: Retrieve the documents based on the indices
#     retrieved_docs = [docs[i] for i in indices[0]]  # indices[0] contains the top_k closest document indices
    
#     return retrieved_docs, distances[0]  # Return the documents and their corresponding distances (similarity)

# # Load documents from CSV
# def load_docs_from_csv(path):
#     import pandas as pd
#     df = pd.read_csv(path)
#     for col in ['content', 'text', 'article', 'body']:
#         if col in df.columns:
#             return df[col].fillna('').astype(str).tolist()
#     return df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist()

# # Build indexes (TF-IDF, BM25, Boolean)
# def build_all_indexes(docs, save=True):
#     print("Building boolean index...")
#     boolean_index = build_boolean_index(docs)
#     if save:
#         joblib.dump(boolean_index, os.path.join(INDEX_DIR, 'boolean.pkl'))

#     print("Building TF-IDF...")
#     vectorizer, tfidf_matrix = build_tfidf_matrix(docs)
#     if save:
#         joblib.dump(vectorizer, os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
#         joblib.dump(tfidf_matrix, os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))

#     print("Building BM25...")
#     bm25_obj, tokenized_docs = build_bm25(docs)
#     if save:
#         joblib.dump({'bm25': bm25_obj, 'tokenized': tokenized_docs}, os.path.join(INDEX_DIR, 'bm25.pkl'))

#     # Save metadata
#     if save:
#         meta = {'n_docs': len(docs)}
#         with open(os.path.join(INDEX_DIR, 'metadata.json'), 'w') as f:
#             json.dump(meta, f)

#     return {
#         'boolean': boolean_index,
#         'vectorizer': vectorizer,
#         'tfidf_matrix': tfidf_matrix,
#         'bm25': bm25_obj,
#         'docs': docs
#     }

# def load_all_indexes():
#     boolean_index = joblib.load(os.path.join(INDEX_DIR, 'boolean.pkl'))
#     vectorizer = joblib.load(os.path.join(INDEX_DIR, 'tfidf_vectorizer.pkl'))
#     tfidf_matrix = joblib.load(os.path.join(INDEX_DIR, 'tfidf_matrix.pkl'))
#     bm25_data = joblib.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
#     bm25_obj = bm25_data['bm25']
#     docs = load_docs_from_csv(DATA_PATH)
#     return {
#         'boolean': boolean_index,
#         'vectorizer': vectorizer,
#         'tfidf_matrix': tfidf_matrix,
#         'bm25': bm25_obj,
#         'docs': docs
#     }

# def hybrid_search(query, indexes, top_k=10):
#     q_tokens = preprocess_text(query)
#     all_doc_ids = set(range(len(indexes['docs'])))

#     # Boolean filter (reduce candidate set)
#     candidates = retrieve_and(q_tokens, indexes['boolean'])
#     if not candidates:
#         candidates = all_doc_ids
#     candidates = list(candidates)

#     # TF-IDF and BM25 scores
#     tfidf_scores_array = score_query(q_tokens, indexes['vectorizer'], indexes['tfidf_matrix'])
#     tfidf_scores = {i: float(tfidf_scores_array[i]) for i in candidates}

#     bm25_scores_arr = score_query_bm25(q_tokens, indexes['bm25'])
#     bm25_scores = {i: float(bm25_scores_arr[i]) for i in candidates}

#     # FAISS search results
#     faiss_docs, faiss_distances = search_with_faiss(query, top_k)  # FAISS search
    
#     # Feature extraction for LTR
#     features = []
#     for doc_id in candidates:
#         features.append([
#             tfidf_scores.get(doc_id, 0.0),
#             bm25_scores.get(doc_id, 0.0),
#             len(indexes['docs'][doc_id].split())  # Example feature: document length
#         ])
    
#     # Train LTR model (in real use, train first, here we assume pre-trained)
#     ltr_model = LearningToRank()
#     ltr_model.train(features, [0 if i % 2 == 0 else 1 for i in range(len(candidates))])  # Dummy labels

#     # Predict rankings using LTR model
#     ltr_scores = ltr_model.predict(features)

#     # Rank documents based on LTR scores
#     ranked_docs = sorted(zip(candidates, ltr_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
#     results = []
#     for doc_id, score in ranked_docs:
#         results.append({
#             'doc_id': doc_id,
#             'score': score,
#             'title_snippet': indexes['docs'][doc_id][:300]  # Snippet from the document
#         })

#     # Integrate FAISS results (use FAISS documents if not found in traditional ranking)
#     for doc, dist in zip(faiss_docs, faiss_distances):
#         results.append({
#             'doc_id': doc,
#             'score': dist,
#             'title_snippet': doc[:300]
#         })

#     return results

# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == 'build':
#         docs = load_docs_from_csv(DATA_PATH)
#         os.makedirs(INDEX_DIR, exist_ok=True)
#         build_all_indexes(docs)
#         print("Indexes built and saved to index/")
#         sys.exit(0)

#     if len(sys.argv) > 1:
#         q = " ".join(sys.argv[1:])
#         print("Loading indexes (if not built, run: python -m src.hybrid_search build)")
#         idxs = load_all_indexes()
#         t0 = time.time()
#         res = hybrid_search(q, idxs, top_k=10)
#         t1 = time.time()
#         print(f"Query time: {t1 - t0:.3f}s")
#         for r in res:
#             print(f"Doc {r['doc_id']} | score={r['score']:.4f}\n{r['title_snippet']}...\n")
#     else:
#         print("Usage: python -m src.hybrid_search [build] OR python -m src.hybrid_search \"your query here\"")
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
import faiss  # Import FAISS library
from transformers import BertTokenizer, BertModel
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Define directory paths
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news.csv')

# Function to load documents from CSV
def load_docs_from_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    for col in ['content', 'text', 'article', 'body']:
        if col in df.columns:
            return df[col].fillna('').astype(str).tolist()
    return df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist()

# Function to load metadata safely
def load_metadata(metadata_file):
    # Check if metadata file exists
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file {metadata_file} not found.")
        return None

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    # Debugging: Print metadata to verify
    print("Metadata loaded:", metadata)

    # Check if 'documents' key exists in metadata
    if "documents" not in metadata:
        print("Error: 'documents' key not found in metadata.")
        return None

    return metadata

# Function to save metadata
def save_metadata(docs, metadata_file):
    meta = {'n_docs': len(docs), 'documents': docs}
    with open(metadata_file, 'w') as f:
        json.dump(meta, f)
    print(f"Metadata saved to {metadata_file}")

# Load metadata (the list of documents)
metadata_file = "index/metadata.json"
metadata = load_metadata(metadata_file)

# Fallback if metadata is not loaded
if metadata is None or "documents" not in metadata:
    print("Error: 'documents' key is missing in metadata. Fallback to loading documents from CSV.")
    docs = load_docs_from_csv(DATA_PATH)
    save_metadata(docs, metadata_file)  # Save the new metadata
else:
    docs = metadata["documents"]

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

def search_with_faiss(query, top_k=10):
    """
    Given a query, search the FAISS index for the most similar documents.
    """
    # Step 1: Generate BERT embeddings for the query
    query_embedding = get_bert_embeddings(query)

    # Step 2: Perform FAISS search (Find top_k nearest neighbors)
    distances, indices = index.search(query_embedding, top_k)  # Returns distances and indices of the closest documents

    # Step 3: Retrieve the documents based on the indices
    retrieved_docs = [docs[i] for i in indices[0]]  # indices[0] contains the top_k closest document indices
    
    return retrieved_docs, distances[0]  # Return the documents and their corresponding distances (similarity)

# Build indexes (TF-IDF, BM25, Boolean)
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

    # Save metadata
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
    q_tokens = preprocess_text(query)
    all_doc_ids = set(range(len(indexes['docs'])))

    # Boolean filter (reduce candidate set)
    candidates = retrieve_and(q_tokens, indexes['boolean'])
    if not candidates:
        candidates = all_doc_ids
    candidates = list(candidates)

    # TF-IDF and BM25 scores
    tfidf_scores_array = score_query(q_tokens, indexes['vectorizer'], indexes['tfidf_matrix'])
    tfidf_scores = {i: float(tfidf_scores_array[i]) for i in candidates}

    bm25_scores_arr = score_query_bm25(q_tokens, indexes['bm25'])
    bm25_scores = {i: float(bm25_scores_arr[i]) for i in candidates}

    # FAISS search results
    faiss_docs, faiss_distances = search_with_faiss(query, top_k)  # FAISS search
    
    # Feature extraction for LTR
    features = []
    for doc_id in candidates:
        features.append([
            tfidf_scores.get(doc_id, 0.0),
            bm25_scores.get(doc_id, 0.0),
            len(indexes['docs'][doc_id].split())  # Example feature: document length
        ])
    
    # Train LTR model (in real use, train first, here we assume pre-trained)
    ltr_model = LearningToRank()
    ltr_model.train(features, [0 if i % 2 == 0 else 1 for i in range(len(candidates))])  # Dummy labels

    # Predict rankings using LTR model
    ltr_scores = ltr_model.predict(features)

    # Rank documents based on LTR scores
    ranked_docs = sorted(zip(candidates, ltr_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_id, score in ranked_docs:
        results.append({
            'doc_id': doc_id,
            'score': score,
            'title_snippet': indexes['docs'][doc_id][:300]  # Snippet from the document
        })

    # Integrate FAISS results (use FAISS documents if not found in traditional ranking)
    for doc, dist in zip(faiss_docs, faiss_distances):
        results.append({
            'doc_id': doc,
            'score': dist,
            'title_snippet': doc[:300]
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
