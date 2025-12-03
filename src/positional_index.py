

import joblib
from src.preprocess import preprocess_text

def build_positional_index(docs):
    index = {}
    for doc_id, text in enumerate(docs):
        toks = preprocess_text(text)
        for pos, t in enumerate(toks):
            index.setdefault(t, {}).setdefault(doc_id, []).append(pos)
    return index

def save_index(index, path):
    joblib.dump(index, path)

def load_index(path):
    return joblib.load(path)

def phrase_query(query_tokens, pos_index):
    
    if not query_tokens:
        return set()
    if len(query_tokens) == 1:
        return set(pos_index.get(query_tokens[0], {}).keys())

    
    first_post = pos_index.get(query_tokens[0], {})
    candidate_docs = set(first_post.keys())
    for i in range(1, len(query_tokens)):
        token = query_tokens[i]
        post = pos_index.get(token, {})
        candidate_docs &= set(post.keys())
        if not candidate_docs:
            return set()

    results = set()
    for doc in candidate_docs:
        positions_list = [pos_index[q][doc] for q in query_tokens]
      
        base_positions = positions_list[0]
        for p in base_positions:
            match = True
            for offset_idx in range(1, len(positions_list := positions_list)):
                if (p + offset_idx) not in positions_list[offset_idx]:
                    match = False
                    break
            if match:
                results.add(doc)
                break
    return results
