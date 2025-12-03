
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybrid_search import load_all_indexes, hybrid_search
from ltr_model import LearningToRank
import numpy as np

def precision_at_k(relevance_list, k):
  
    if k == 0:
        return 0.0
    return sum(relevance_list[:k]) / float(k)

def mean_average_precision(relevance_list):
  
    average_precisions = []
    for k in range(1, len(relevance_list) + 1):
        precision_at_k_val = precision_at_k(relevance_list, k)
        average_precisions.append(precision_at_k_val)
    return np.mean(average_precisions)

def ndcg_at_k(relevance_list, k):

    dcg = 0.0
    for i in range(k):
        if i < len(relevance_list):
            dcg += relevance_list[i] / np.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(k, len(relevance_list))):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg != 0 else 0.0


def extract_features(docs):
    
    features = []
    for doc in docs:
        features.append([
            doc['score_rerank'], 
            len(doc['title_snippet'].split()), 
           
        ])
    return np.array(features)
def calculate_precision_recall_at_k(query, top_k=10):
    idxs = load_all_indexes()

  
    hits = hybrid_search(query, idxs, top_k=top_k)


    ground_truth = {
        "covid pakistan": [0, 2],
        "ai regulation": [1, 3],
        "football world cup": [0, 1],
        "inflation pakistan": [0, 1], 
    }

    relevant_docs = ground_truth.get(query, [])

    print(f"Hits for query: {query}")
    print(hits) 
    print(f"Relevant documents: {relevant_docs}") 

  
    if len(relevant_docs) == 0:
        recall = 0.0
    else:
       
        relevance_list = [1 if hit['doc_id'] in relevant_docs else 0 for hit in hits]

       
        print(f"Relevance list for query '{query}': {relevance_list}")

        
        precision = precision_at_k(relevance_list, top_k)
        recall = np.sum(relevance_list) / float(len(relevant_docs))  
        map_score = mean_average_precision(relevance_list)
        ndcg_score = ndcg_at_k(relevance_list, top_k)

    return precision, recall, map_score, ndcg_score


def evaluate_ltr_model(query, ranked_docs):

    ltr = LearningToRank()
    X_test = extract_features(ranked_docs)  
    y_pred = ltr.predict(X_test)
    return calculate_metrics(y_pred, ranked_docs)

def run_manual_evaluation(queries, top_k=10):
    
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
  

    precision_at_k = np.mean([1 if pred >= 0.5 else 0 for pred in predictions])  # Simplified calculation
    return precision_at_k