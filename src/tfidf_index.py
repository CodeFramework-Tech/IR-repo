import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import preprocess_text

def build_tfidf_matrix(docs):
    clean_docs = [" ".join(preprocess_text(d)) for d in docs]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(clean_docs)
    return vectorizer, matrix
def save_tfidf(vectorizer, matrix, path="index/tfidf.pkl"):
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, path)
def load_tfidf(path="index/tfidf.pkl"):
    return joblib.load(path)
def score_query(query_tokens, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([" ".join(query_tokens)])
    cosine_similarities = (query_tfidf * tfidf_matrix.T).toarray()
    return cosine_similarities[0]
