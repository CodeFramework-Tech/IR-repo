import sys
import os


def calculate_precision_recall(retrieved_docs, relevant_docs):

    TP = len(set(retrieved_docs) & set(relevant_docs))
    FP = len(set(retrieved_docs) - set(relevant_docs))
    FN = len(set(relevant_docs) - set(retrieved_docs))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


retrieved_docs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

# Replace these numbers with your actual relevant document IDs from your file.
relevant_docs = [5,20,31]


precision, recall = calculate_precision_recall(retrieved_docs, relevant_docs)
TP = len(set(retrieved_docs) & set(relevant_docs))

print(f"\n--- Evaluation Results ---")
print(f"Query: 'The role of BERT in hybrid information retrieval systems'")
print(f"Retrieved IDs: {retrieved_docs}")
print(f"Relevant IDs (Ground Truth): {relevant_docs}")
print(f"True Positives (TP): {TP}")
print("------------------------------")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
