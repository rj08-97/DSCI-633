import pandas as pd
import numpy as np
from collections import Counter


class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str

        # instantiate self.classes_
        self.classes_ = list(set(list(y)))
        # instantiate self.P_y
        self.P_y = Counter(y)

        features = [feature for feature in X]
        feature_values = [X[feature].unique() for feature in features]
        feature_probabilities = dict()
        labels = [label for label in self.classes_]

        for label in labels:
            number_of_occurences_of_label = Counter(y)[label]
            feature_attribute_probability_value = dict()
            for feature in range(0, len(feature_values)):
                # number of rows
                all_possible_feature_values = feature_values[feature]
                feature_attribute_probability_value[feature] = dict()
                number_of_each_possible_value = Counter(X[feature])

                for feat_val in all_possible_feature_values:
                    feature_attribute_probability_value[feature][feat_val] = dict()
                    feat_prob = (sum(
                                        [feat_val == X[feature][entry] and y[entry] == label for entry in range(len(X[feature]))]
                                    ) + self.alpha
                                ) / ( number_of_occurences_of_label +
                                    (len(number_of_each_possible_value.keys()) * self.alpha)
                                )

                    feature_attribute_probability_value[feature][feat_val] = feat_prob

            feature_probabilities[label] = feature_attribute_probability_value

        self.P = feature_probabilities

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions
