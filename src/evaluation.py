# import sys
# import os

# # Make sure the parent directory is added to the path for module imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# # Direct import
# from hybrid_search import load_all_indexes, hybrid_search

# import numpy as np

# def precision_at_k(relevance_list, k):
#     """
#     Calculate Precision@k: Fraction of relevant documents in top k results.

#     Args:
#     - relevance_list (list): Relevance of documents in the top k (1 for relevant, 0 for not).
#     - k (int): The rank position at which to calculate precision.

#     Returns:
#     - float: Precision at k.
#     """
#     if k == 0:
#         return 0.0
#     return sum(relevance_list[:k]) / float(k)

# def mean_average_precision(relevance_list):
#     """
#     Calculate Mean Average Precision (MAP) for a given relevance list.

#     Args:
#     - relevance_list (list): A list of relevance scores (1 for relevant, 0 for non-relevant).

#     Returns:
#     - float: The Mean Average Precision score.
#     """
#     average_precisions = []
#     for k in range(1, len(relevance_list) + 1):
#         precision_at_k_val = precision_at_k(relevance_list, k)
#         average_precisions.append(precision_at_k_val)
#     return np.mean(average_precisions)

# def ndcg_at_k(relevance_list, k):
#     """
#     Compute NDCG (Normalized Discounted Cumulative Gain) at rank position k.

#     Args:
#     - relevance_list (list): A list of relevance scores (1 for relevant, 0 for non-relevant).
#     - k (int): The rank position at which to calculate NDCG.

#     Returns:
#     - float: The NDCG score.
#     """
#     dcg = 0.0
#     for i in range(k):
#         if i < len(relevance_list):
#             dcg += relevance_list[i] / np.log2(i + 2)

#     idcg = 0.0
#     for i in range(min(k, len(relevance_list))):
#         idcg += 1 / np.log2(i + 2)

#     return dcg / idcg if idcg != 0 else 0.0

# def calculate_precision_recall_at_k(query, top_k=10):
#     """
#     Calculate Precision@k, Recall@k, MAP, and NDCG for a given query using the top-k ranked results.

#     Args:
#     - query (str): The query text.
#     - top_k (int): The number of top results to evaluate.

#     Returns:
#     - tuple: Precision@k, Recall@k, MAP, and NDCG.
#     """
#     idxs = load_all_indexes()
#     hits = hybrid_search(query, idxs, top_k=top_k)

#     # Define ground truth data for relevance
#     ground_truth = {
#         "covid pakistan": [0, 2],
#         "ai regulation": [1, 3],
#         "football world cup": [0, 1],
#         "inflation pakistan": [0, 1],  # Add the relevant document IDs here
  
 
 
#     }

#     relevant_docs = ground_truth.get(query, [])

#     # Convert hits to list of doc_ids and compute relevance
#     relevance_list = [1 if hit['doc_id'] in relevant_docs else 0 for hit in hits]

#     # Calculate Precision, Recall, MAP, and NDCG
#     precision = precision_at_k(relevance_list, top_k)
#     recall = np.sum(relevance_list) / float(len(relevant_docs))  # Use ground truth length for recall
#     map_score = mean_average_precision(relevance_list)
#     ndcg_score = ndcg_at_k(relevance_list, top_k)

#     return precision, recall, map_score, ndcg_score

# def run_manual_evaluation(queries, top_k=10):
#     """
#     queries: list of (query_text, manual_relevance_lists) OR just query_text
#     If manual_relevance_lists not provided, this function prints results for you to judge manually.
#     """
#     idxs = load_all_indexes()
#     results = {}
#     for q in queries:
#         print("=== QUERY ===")
#         print(q)
#         hits = hybrid_search(q, idxs, top_k=top_k)
#         for rank, h in enumerate(hits, start=1):
#             print(rank, " - Doc", h['doc_id'], "score:", h['score_rerank'])
#             print(h['title_snippet'][:250])
#             print('-'*40)
#         print("\nPlease mark relevance for each of top results and record P@5/P@10 in your table.\n")
#     return results

import sys
import os

# Make sure the parent directory is added to the path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybrid_search import load_all_indexes, hybrid_search
from ltr_model import LearningToRank
import numpy as np

def precision_at_k(relevance_list, k):
    """
    Calculate Precision@k: Fraction of relevant documents in top k results.
    
    Args:
    - relevance_list (list): Relevance of documents in the top k (1 for relevant, 0 for not).
    - k (int): The rank position at which to calculate precision.
    
    Returns:
    - float: Precision at k.
    """
    if k == 0:
        return 0.0
    return sum(relevance_list[:k]) / float(k)

def mean_average_precision(relevance_list):
    """
    Calculate Mean Average Precision (MAP) for a given relevance list.
    
    Args:
    - relevance_list (list): A list of relevance scores (1 for relevant, 0 for non-relevant).
    
    Returns:
    - float: The Mean Average Precision score.
    """
    average_precisions = []
    for k in range(1, len(relevance_list) + 1):
        precision_at_k_val = precision_at_k(relevance_list, k)
        average_precisions.append(precision_at_k_val)
    return np.mean(average_precisions)

def ndcg_at_k(relevance_list, k):
    """
    Compute NDCG (Normalized Discounted Cumulative Gain) at rank position k.
    
    Args:
    - relevance_list (list): A list of relevance scores (1 for relevant, 0 for non-relevant).
    - k (int): The rank position at which to calculate NDCG.
    
    Returns:
    - float: The NDCG score.
    """
    dcg = 0.0
    for i in range(k):
        if i < len(relevance_list):
            dcg += relevance_list[i] / np.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(k, len(relevance_list))):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg != 0 else 0.0

# def calculate_precision_recall_at_k(query, top_k=10):
#     """
#     Calculate Precision@k, Recall@k, MAP, and NDCG for a given query using the top-k ranked results.
    
#     Args:
#     - query (str): The query text.
#     - top_k (int): The number of top results to evaluate.
    
#     Returns:
#     - tuple: Precision@k, Recall@k, MAP, and NDCG.
#     """
#     idxs = load_all_indexes()
#     hits = hybrid_search(query, idxs, top_k=top_k)

#     # Define ground truth data for relevance
#     ground_truth = {
#         "covid pakistan": [0, 2],
#         "ai regulation": [1, 3],
#         "football world cup": [0, 1],
#         "inflation pakistan": [0, 1],  # Add the relevant document IDs here
#     }

#     relevant_docs = ground_truth.get(query, [])

#     # If no relevant documents are found, handle the case
#     if len(relevant_docs) == 0:
#         recall = 0.0
#     else:
#         # Convert hits to list of doc_ids and compute relevance
#         relevance_list = [1 if hit['doc_id'] in relevant_docs else 0 for hit in hits]

#         # Calculate Precision, Recall, MAP, and NDCG
#         precision = precision_at_k(relevance_list, top_k)
#         recall = np.sum(relevance_list) / float(len(relevant_docs))  # Use ground truth length for recall
#         map_score = mean_average_precision(relevance_list)
#         ndcg_score = ndcg_at_k(relevance_list, top_k)
    
#     return precision, recall, map_score, ndcg_score
def extract_features(docs):
    """
    Extract features for Learning to Rank. 
    These can include query-document similarity, document metadata, etc.
    For simplicity, let's assume some basic features here.
    """
    features = []
    for doc in docs:
        features.append([
            doc['score_rerank'],  # Example: Reranking score from previous methods
            len(doc['title_snippet'].split()),  # Length of document snippet (just an example feature)
            # Add more features like content-based similarity, etc.
        ])
    return np.array(features)
def calculate_precision_recall_at_k(query, top_k=10):
    """
    Calculate Precision@k, Recall@k, MAP, and NDCG for a given query using the top-k ranked results.
    
    Args:
    - query (str): The query text.
    - top_k (int): The number of top results to evaluate.
    
    Returns:
    - tuple: Precision@k, Recall@k, MAP, and NDCG.
    """
    # Load all indexes
    idxs = load_all_indexes()

    # Get search results from hybrid_search
    hits = hybrid_search(query, idxs, top_k=top_k)

    # Define ground truth data for relevance
    ground_truth = {
        "covid pakistan": [0, 2],
        "ai regulation": [1, 3],
        "football world cup": [0, 1],
        "inflation pakistan": [0, 1],  # Add the relevant document IDs here
    }

    relevant_docs = ground_truth.get(query, [])

    # Print hits and relevant documents for debugging
    print(f"Hits for query: {query}")
    print(hits)  # Print search results
    print(f"Relevant documents: {relevant_docs}")  # Print the ground truth relevant docs

    # If no relevant documents are found, handle the case
    if len(relevant_docs) == 0:
        recall = 0.0
    else:
        # Convert hits to list of doc_ids and compute relevance
        relevance_list = [1 if hit['doc_id'] in relevant_docs else 0 for hit in hits]

        # Print the relevance list for debugging
        print(f"Relevance list for query '{query}': {relevance_list}")

        # Calculate Precision, Recall, MAP, and NDCG
        precision = precision_at_k(relevance_list, top_k)
        recall = np.sum(relevance_list) / float(len(relevant_docs))  # Use ground truth length for recall
        map_score = mean_average_precision(relevance_list)
        ndcg_score = ndcg_at_k(relevance_list, top_k)

    return precision, recall, map_score, ndcg_score


def evaluate_ltr_model(query, ranked_docs):
    """Evaluate Learning to Rank results using MAP, NDCG, etc."""
    ltr = LearningToRank()
    X_test = extract_features(ranked_docs)  # Define feature extraction
    y_pred = ltr.predict(X_test)
    return calculate_metrics(y_pred, ranked_docs)

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
def calculate_metrics(predictions, docs):
    """
    Calculate metrics like MAP, Precision, Recall, etc.
    You can define these metrics as needed.
    """
    # Just an example for Precision@k calculation
    precision_at_k = np.mean([1 if pred >= 0.5 else 0 for pred in predictions])  # Simplified calculation
    return precision_at_k