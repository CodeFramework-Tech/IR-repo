from sklearn.svm import SVC
import numpy as np

class LearningToRank:
    def __init__(self):
        self.model = SVC(kernel='linear')

    def train(self, X_train, y_train):
        """Train the model on feature vectors and labels."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict ranking scores for test queries."""
        return self.model.predict(X_test)
