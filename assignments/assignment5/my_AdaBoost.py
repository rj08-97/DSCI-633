import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace
import math

class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)
            # Compute alpha for estimator i (don't forget to use k for multi-class)
            alpha = np.log((1.0 - error)/error) + np.log(k - 1)
            self.alpha.append(alpha)

            #update wi
            w = w*np.exp(alpha*diffs)
            w = w/sum(w)

            #for index in range(len(diffs)):
             #   normalizing_factor = sum(w)
              #  if not diffs[index]:
               #     w[index] = math.pow(w[index]/normalizing_factor, self.alpha[-1])
        # Normalize alpha
        self.alpha = self.alpha / np.sum(self.alpha)

        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        for label in self.classes_:
            # Calculate probs for each label
            probsvalue = []
            for key in range(self.n_estimators):
                probsvalue.append(self.alpha[key] * (self.estimators[key].predict(X) == label))
            probs[label] = np.sum(probsvalue,axis=0)
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs
