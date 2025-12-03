from preprocess import preprocess_text
from boolean_index import boolean_retrieve
import joblib


data = joblib.load("index/tfidf.pkl")
vectorizer = data["vectorizer"]
tfidf_matrix = data["matrix"]
boolean_index = joblib.load("index/boolean.pkl")
query = "covid vaccine news"
tokens = preprocess_text(query)
print("\nPreprocessed tokens:", tokens)
all_docs = set(range(tfidf_matrix.shape[0]))
bool_result = boolean_retrieve(tokens, boolean_index, all_docs)

print("\nBoolean results:", bool_result)
