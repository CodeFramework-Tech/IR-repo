# src/boolean_index.py
# Simple boolean inverted index: term -> set(docIDs)
# Student-style and straightforward.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import joblib
from src.preprocess import preprocess_text

def build_boolean_index(docs):
    """
    docs: list of strings
    returns: dict term -> set(doc_ids)
    """
    index = {}
    for doc_id, text in enumerate(docs):
        toks = preprocess_text(text)
        seen = set()
        for t in toks:
            if t not in seen:
                index.setdefault(t, set()).add(doc_id)
                seen.add(t)
    return index

def save_index(index, path):
    joblib.dump(index, path)

def load_index(path):
    return joblib.load(path)

def boolean_and(postings_list):
    if not postings_list:
        return set()
    res = postings_list[0].copy()
    for p in postings_list[1:]:
        res &= p
    return res

def boolean_or(postings_list):
    res = set()
    for p in postings_list:
        res |= p
    return res

def boolean_not(all_docs_set, posting):
    return all_docs_set - posting

def retrieve_and(query_tokens, index):
    postings = [index.get(t, set()) for t in query_tokens]
    return boolean_and(postings)
def boolean_retrieve(query_tokens, index):
    """
    Simple AND boolean retrieval.
    Returns a set of doc_ids that contain ALL query tokens.
    """
    posting_lists = []
    for token in query_tokens:
        if token in index:
            posting_lists.append(set(index[token]))
        else:
            posting_lists.append(set())

    if not posting_lists:
        return set()

    return set.intersection(*posting_lists)
