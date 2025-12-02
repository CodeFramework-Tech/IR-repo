# src/bm25_index.py
# BM25 index builder
import joblib
from rank_bm25 import BM25Okapi
from src.preprocess import preprocess_text  # Ensure this function is implemented correctly

def build_bm25(docs):
    """
    Build the BM25 index from a list of documents.

    Args:
    - docs: List of document strings.

    Returns:
    - bm25_obj: A BM25 object that can be used for scoring queries.
    - tokenized_docs: The tokenized versions of the documents.
    """
    # Tokenize documents using preprocess_text (which handles cleaning, stopwords, etc.)
    tokenized_docs = [preprocess_text(d) for d in docs]  # Ensure preprocess_text works here
    bm25_obj = BM25Okapi(tokenized_docs)  # Build the BM25 model
    return bm25_obj, tokenized_docs

def save_bm25(bm25, tokenized_docs, path="index/bm25.pkl"):
    """
    Save the BM25 object and tokenized documents to disk.

    Args:
    - bm25: The BM25 object.
    - tokenized_docs: The tokenized documents.
    - path: Path where the BM25 object will be saved.
    """
    joblib.dump({"bm25": bm25, "docs": tokenized_docs}, path)

def load_bm25(path="index/bm25.pkl"):
    """
    Load the BM25 object and tokenized documents from disk.

    Args:
    - path: Path to the saved BM25 object.
    
    Returns:
    - dict: Contains "bm25" and "docs".
    """
    return joblib.load(path)

def score_query_bm25(query_tokens, bm25_obj):
    """
    Score a query using the BM25 object.

    Args:
    - query_tokens: The query tokens (list of strings).
    - bm25_obj: The BM25 object built from the corpus.

    Returns:
    - scores: A list of BM25 scores for each document.
    """
    return bm25_obj.get_scores(query_tokens)
