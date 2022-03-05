import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import pandas as pd

def NMF(Y, latent_features = 2, epochs=100000, lr=10e-4, beta=10e-5):
    Y = np.array(Y)
    lh = lw = beta
    m, n = Y.shape
    nr = np.count_nonzero(Y)
    k = latent_features

    W = np.random.rand(m,k)
    H = np.random.rand(n,k)

    rated = Y.copy()
    rated[rated > 0] = 1
    rated[rated <= 0] = 0

    best_W = W
    best_H = H
    best_e = 0
    not_reduced = 0
    history_e = []

    for i in range(epochs):
        diff = np.dot(W, H.T) - Y
        diff = diff * rated / nr

        grad_W = diff.dot(H)
        W = W - lr * (grad_W + lw * W)

        grad_H = diff.T.dot(W)
        H = H - lr * (grad_H + lh * H)

        e = 0.5*(nr*LA.norm(diff)**2 + lw*(LA.norm(W) ** 2) + lh * (LA.norm(H) ** 2))
        if (i > 0):
            if (best_e - e >= 10e-4):
                best_e = e
                best_H = H
                best_W = W
                not_reduced = 0
            else:
                not_reduced += 1
        else:
            best_e = e
        history_e.append(e)
        print(i, e)
        if(not_reduced >= 100):
            break
    return best_e, best_W, best_H, history_e

def utility_matrix(data, m=943, n=1682):
    Y = np.zeros((943, 1682))
    for i in range(len(data)):
        user_id = data[i, 0]
        movie_id = data[i, 1]
        rating = data[i, 2]
        Y[user_id - 1, movie_id - 1] = rating
    return Y

def read_file():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

    rate_train = ratings_base.values
    rate_test = ratings_test.values

    Y_train = utility_matrix(rate_train)
    Y_test = utility_matrix(rate_test)
    return Y_train, Y_test

def MSE(Y_test, Y_predict):
    rated = Y_test.copy()
    rated[rated > 0] = 1
    rated[rated <= 0] = 0
    nr = np.count_nonzero(rated)

    diff = Y_predict - Y_test
    diff = diff * rated / nr

    Y_predict = Y_predict * rated
    mse = nr * LA.norm(diff) ** 2
    return mse

if __name__ == "__main__":
    Y_train, Y_test = read_file()
    k = 500
    best_e, W, H, history_e = NMF(Y_train, k, epochs=10000, lr = 10e-2, beta=10e-3)
    Y_pred = np.round(W.dot(H.T))
    Y_pred[Y_pred > 5] = 5
    Y_pred[Y_pred <= 0] = 1
    print(Y_pred[0:5,0:5])
    mse = MSE(Y_train, W.dot(H.T))
    print("MSE on train data: " + str(mse))
    mse = MSE(Y_test, W.dot(H.T))
    print("MSE on test data: " + str(mse))
    print('---------------')
    x = range(len(history_e))
    y = history_e
    plt.plot(x, y)
    plt.show()