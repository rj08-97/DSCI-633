import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from pdb import set_trace

def pca(X, n_components = 5):
    #  Use svd to perform PCA on X
    #  Inputs:
    #     X: input matrix
    #     n_components: number of principal components to keep
    #  Output:
    #     principal_components: the top n_components principal_components
    #     X_pca = X.dot(principal_components)

    U, s, Vh = svd(X)

    principal_components = Vh.T[:,:n_components]
    return principal_components

def vector_norm(x, norm="Min-Max"):
    # Calculate the normalized vector
    # Input x: 1-d np.array
    if norm == "Min-Max":
        min_norm_value = min(x)
        max_norm_value = max(x)
        x_norm = (x - min_norm_value) / (max_norm_value - min_norm_value)
    elif norm == "L1":
        for key in range(0,len(x)):
            sum_x = np.sum(abs(x[key]))
        x_norm = x / sum_x
    elif norm == "L2":
        for key in range(0,len(x)):
            sum_square = np.sum(x[key] ** 2)
        x_norm = x/np.sqrt(sum_square)
    elif norm == "Standard_Score":
        mean = np.mean(x)
        standard_deviation = np.std(x)
        x_norm = (x-mean)/standard_deviation
    else:
        raise Exception("Unknown normlization.")
    return x_norm

def normalize(X, norm="Min-Max", axis = 1):
    #  Inputs:
    #     X: input matrix
    #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
    #     axis = 0: normalize rows
    #     axis = 1: normalize columns
    #  Output:
    #     X_norm: normalized matrix (numpy.array)

    X_norm = deepcopy(np.asarray(X))
    m, n = X_norm.shape
    if axis == 1:
        for col in range(n):
            X_norm[:,col] = vector_norm(X_norm[:,col], norm=norm)
    elif axis == 0:
        X_norm = np.array([vector_norm(X_norm[i], norm=norm) for i in range(m)])
    else:
        raise Exception("Unknown axis.")
    return X_norm

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    sample = [] * len(y)
    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    # Write your own code below
    for value in range(0,len(y_array)):
        sample.append(int(np.ceil(ratio * np.random.choice(len(y_array),replace=replace))))
    return sample