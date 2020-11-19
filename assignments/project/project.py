import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier


class my_model():
    def __init__(self):
        self.preprocessor = CountVectorizer(stop_words='english')
        self.clf = PassiveAggressiveClassifier(C=0.1, fit_intercept=True, n_iter_no_change=20, validation_fraction=0.8)

    def fit(self, X, y):
        # do not exceed 29 mins
        XX = self.preprocessor.fit_transform(X["description"])
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).transform(XX)
        self.clf.fit(X_final, y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        XX = self.preprocessor.transform(X["description"])
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).transform(XX)
        predictionsOfModel = self.clf.predict(X_final)
        return predictionsOfModel

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    # Train model
    clf = my_model()
    clf.fit(X, y)
    runtime = (time.time() - start) / 60.0
    print(runtime)
    predictions = clf.predict(X)
    print(predictions)
