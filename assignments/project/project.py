import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier


def cleanup_feature(feat):
    return feat.map(lambda x: str(x).strip("#" + " "))


def get_train_test_df(X):
    return cleanup_feature(X["title"]) + " " \
           + cleanup_feature(X["description"]) + " " \
           + cleanup_feature(X["telecommuting"]) + " " \
           + cleanup_feature(X["has_questions"])


class my_model:
    def __init__(self):
        # defines the self function used in fit and predict
        self.preprocessor = CountVectorizer(stop_words='english')
        self.clf = PassiveAggressiveClassifier(C=0.1, fit_intercept=True, n_iter_no_change=10, validation_fraction=0.8)

    def fit(self, X, y):
        # do not exceed 29 mins
        X_df = get_train_test_df(X)
        XX = self.preprocessor.fit_transform(X_df)
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).fit_transform(XX)
        self.clf.fit(X_final, y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X_df = get_train_test_df(X)
        XX = self.preprocessor.transform(X_df)
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).fit_transform(XX)
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
