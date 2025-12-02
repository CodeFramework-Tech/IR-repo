# # src/tfidf_index.py
# # Build and save TF-IDF matrix
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from preprocess import preprocess_text

# def build_tfidf_matrix(docs):
#     """
#     docs: list of raw document strings
#     returns: vectorizer, tfidf_matrix
#     """
#     clean_docs = [" ".join(preprocess_text(d)) for d in docs]

#     vectorizer = TfidfVectorizer()
#     matrix = vectorizer.fit_transform(clean_docs)

#     return vectorizer, matrix

# def save_tfidf(vectorizer, matrix, path="index/tfidf.pkl"):
#     joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, path)

# def load_tfidf(path="index/tfidf.pkl"):
#     return joblib.load(path)
# src/tfidf_index.py
# Build and save TF-IDF matrix
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
# from preprocess import preprocess_text
from src.preprocess import preprocess_text

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

def score_query(query_tokens, vectorizer, tfidf_matrix):
    """
    Calculate the similarity score for a query using the TF-IDF matrix.

    Args:
    - query_tokens (list): Tokens of the query.
    - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    - tfidf_matrix (scipy.sparse.csr_matrix): The pre-computed TF-IDF matrix.

    Returns:
    - array: The similarity scores for each document.
    """
    query_tfidf = vectorizer.transform([" ".join(query_tokens)])
    cosine_similarities = (query_tfidf * tfidf_matrix.T).toarray()
    return cosine_similarities[0]
