# src/tfidf_index.py
# Build and save TF-IDF matrix
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text

def build_tfidf_matrix(docs):
    """
    docs: list of raw document strings
    returns: vectorizer, tfidf_matrix
    """
    clean_docs = [" ".join(preprocess_text(d)) for d in docs]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(clean_docs)

    return vectorizer, matrix

def save_tfidf(vectorizer, matrix, path="index/tfidf.pkl"):
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, path)

def load_tfidf(path="index/tfidf.pkl"):
    return joblib.load(path)
