# src/evaluation.py
# Small utilities to help compute Precision@k and run manual evaluation.
# You will run hybrid_search and then manually label results.

from collections import defaultdict
from src.hybrid_search import load_all_indexes, hybrid_search

def precision_at_k(relevance_list, k):
    if k == 0:
        return 0.0
    return sum(relevance_list[:k]) / float(k)

def run_manual_evaluation(queries, top_k=10):
    """
    queries: list of (query_text, manual_relevance_lists) OR just query_text
    If manual_relevance_lists not provided, this function prints results for you to judge manually.
    """
    idxs = load_all_indexes()
    results = {}
    for q in queries:
        print("=== QUERY ===")
        print(q)
        hits = hybrid_search(q, idxs, top_k=top_k)
        for rank, h in enumerate(hits, start=1):
            print(rank, " - Doc", h['doc_id'], "score:", h['score_rerank'])
            print(h['title_snippet'][:250])
            print('-'*40)
        print("\nPlease mark relevance for each of top results and record P@5/P@10 in your table.\n")
    return results

# Example usage:
# queries = ["inflation pakistan", "ai regulation", "football world cup"]
# run_manual_evaluation(queries)
