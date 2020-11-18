import pandas as pd
import time

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.tree import DecisionTreeClassifier

from assignment8.my_evaluation import my_evaluation


class my_model():
    def __init__(self):
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        self.clf = PassiveAggressiveClassifier()

    def fit(self, X, y):
        # do not exceed 29 mins
        XX = self.preprocessor.fit_transform(X["description"])
        self.clf.get_params(deep=True)
        self.clf.fit(XX, y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        transform = PCA(n_components=100)
        XX = self.preprocessor.transform(X["description"])
        predictionsOfModel = self.clf.predict(XX)
        return predictionsOfModel
    # change the name to predictions before submission




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
