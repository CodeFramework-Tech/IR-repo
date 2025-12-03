
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'index', 'reranker_model.pkl')

def featurize(candidate_doc_id, query_tokens, docs, tfidf_scores, bm25_scores):
    
    doc = docs[candidate_doc_id]
    doc_tokens = doc.split() 
    doc_len = len(doc_tokens)
    overlap = sum(1 for t in query_tokens if t in doc_tokens)
    bm = float(bm25_scores[candidate_doc_id]) if candidate_doc_id in bm25_scores else 0.0
    tf = float(tfidf_scores[candidate_doc_id]) if candidate_doc_id in tfidf_scores else 0.0
    return np.array([bm, tf, doc_len, overlap], dtype=float)

class Reranker:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            
            self.model = None

    def train(self, X, y):
     
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_PATH)
        self.model = clf

    def score(self, features):
       
        if self.model:
            return self.model.predict_proba(features.reshape(1, -1))[0,1]
     
        bm, tf, doc_len, overlap = features
        return 0.6 * bm + 0.3 * tf + 0.001 * overlap - 0.00001 * doc_len
