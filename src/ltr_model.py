from sklearn.svm import SVC
import numpy as np

class LearningToRank:
    def __init__(self):
        self.model = SVC(kernel='linear')
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
    
        return self.model.predict(X_test)
