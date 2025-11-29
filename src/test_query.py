# src/test_queries.py

from preprocess import preprocess_text
from boolean_index import boolean_retrieve
import joblib

# load TF-IDF index
data = joblib.load("index/tfidf.pkl")
vectorizer = data["vectorizer"]
tfidf_matrix = data["matrix"]

# load boolean index
boolean_index = joblib.load("index/boolean.pkl")

# Example query
query = "covid vaccine news"

tokens = preprocess_text(query)
print("\nPreprocessed tokens:", tokens)

# Boolean filter
all_docs = set(range(tfidf_matrix.shape[0]))
bool_result = boolean_retrieve(tokens, boolean_index, all_docs)

print("\nBoolean results:", bool_result)
