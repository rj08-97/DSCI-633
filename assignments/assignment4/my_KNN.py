import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y

    def dist(self,x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])

        kwargs = {"X": self.X, "p": self.p, "x": x}
        algorithm = dict()
        algorithm["euclidean"] = euclidean
        algorithm["manhattan"] = manhattan
        algorithm["cosine"] = cosine
        algorithm["minkowski"] = minkowski

        if not self.metric in algorithm:
            raise Exception("Unknown criterion.")

        return algorithm[self.metric](**kwargs)

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        distances = self.dist(x)
        labels = self.y
        mapped_labels = [(distance, label) for distance, label in zip(distances, labels)]
        mapped_labels.sort(key=lambda x: x[0])

        output = Counter(map(lambda element: element[1], mapped_labels[:self.n_neighbors]))
        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs

def minkowski(X, p, x):
    ' Implementation of the minkowski distance'
    distances = []
    points = [row.tolist() for ind, row in X.iterrows()]
    for point in points:
        distances.append(sum([abs(point[index] - x[index]) ** p for index in range(len(point))]))
    return np.array(distances)


def euclidean(X, x,p):
    ' Implementation of the euclidean distance'
    distances = []
    points = [row.tolist() for ind, row in X.iterrows()]
    for point in points:
        distances.append(np.sqrt(sum([((point[index] - x[index]) ** 2) for index in range(len(point))])))
    return np.array(distances)

def manhattan(X,x,p):
    ' Implementation of the manhattan distance'
    distances = []
    points = [row.tolist() for ind, row in X.iterrows()]
    for point in points:
        distances.append(sum([abs(point[index] - x[index]) for index in range(len(point))]))
    return np.array(distances)


def cosine(X,x,p):
    ' Implementation of the cosine distance'
    distances = []
    points = [row.tolist() for ind, row in X.iterrows()]
    for point in points:
        sumof_points = sum([point[x] ** 2 for x in range(len(point))])
        sumof_x = sum([x[v] ** 2 for v in range(len(x))])
        num = sum([point[val] * x[val] for val in range(len(point))])
        denom = np.sqrt(sumof_points) * np.sqrt(sumof_x)
        distances.append(1-float(num/denom))
    return np.array(distances)