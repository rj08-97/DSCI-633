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
        self.classes_ = list(set(list(y)))
        self.P_y = Counter(y)

        self.P = {}
        #stores the key values of P_y
        y_val = list(self.P_y.keys())
        #intializes the list for label,key and feature/value storing
        val_counts_list = []
        prob_val_list = []
        val_arr_list = []
        key_val_list = []

        for i in range(len(y_val)):
            print("i", i)
            for j in range(len(X.columns)):
                val_count = []
                val_arr = []
                h = [k for k in range(0, len(X[j]))]
                key_val = [X[j][k] for k in h]
                for k in h:
                    try:
                        val_count.append(X[j][y.loc[y == y_val[i]].index.values].value_counts()[X[j][k]] + self.alpha)
                        val_arr.append(X[j][y.loc[y == y_val[i]].index.values].value_counts()[X[j][k]])
                    except:
                        val_arr.append(0)
                        val_count.append(0 + self.alpha)
                num_avail_cat = len(X[j])
                val_arr_list.append(val_arr)
                val_counts_list.append(val_count)
                prob_val_list.append(val_count / (sum(val_arr) + num_avail_cat * self.alpha))
                key_val_list.append(key_val)

        for n in range(0, len(y_val)):
            dict_a = {}
            dict_b = {}
            for column in range(0, len(X.columns)):
                for k in range(0,len(key_val_list[j])):
                    dict_b[key_val_list[j][k]] = prob_val_list[j][k]
                dict_a[column] = dict_b
            self.P[y_val[n]] = dict_a


        total_labels = len(list(y))
        counted_labels = dict()
        for label in list(y):
            if label in counted_labels:
               counted_labels[label] = counted_labels[label] + 1
            else:
                counted_labels[label] = 1

        prob_y = dict()
        for key in counted_labels:
            prob_y[key] = counted_labels[key]/total_labels
        self.P_y = prob_y

        return

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