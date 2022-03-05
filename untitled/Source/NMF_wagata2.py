import numpy as np
from sklearn.decomposition import NMF
from numpy import linalg as LA, math
import pandas as pd
import jgraph
import warnings
warnings.filterwarnings('ignore')

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):

    rated = R.copy()
    thresh = rated > 0
    rated[thresh] = 1
    rated[~thresh] = 0

    count = np.count_nonzero(rated)

    Q = Q.T

    for step in range(steps):
        # P -= alpha/count*((P.dot(Q.T)).dot(Q) - (R*rated).dot(Q)) + alpha*P
        # Q -= beta/count* ((Q.dot(P.T)).dot(P) - (R*rated).T.dot(P)) + beta*Q
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        hypothesis = (R - P.dot(Q))* rated

        RMSE = LA.norm(hypothesis) / math.sqrt(count)

        e = LA.norm(hypothesis) ** 2 \
            + (beta / 2) * (LA.norm(P) ** 2 + LA.norm(Q) ** 2)

        print('epoch =', step, ' e =', np.round(e,2), ' RMSE =', np.round(RMSE,2))
    return P, Q

r_cols = ['user_id', 'player_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('user_team.csv', sep=',', names=r_cols, encoding='latin-1')
rate_train = ratings_base.values
rate_train = np.array(rate_train)
R = np.zeros((max(rate_train[:,0]) + 1, 20200), dtype=float)
print(R.shape)
for row in rate_train:
    user_id = row[0]
    player_id = row[1]
    point = row[2]
    R[user_id, player_id] = 100

N, M = R.shape

K = 100

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K, steps = 2000, alpha=10e-5,beta=10e-5)
nR = np.dot(nP, nQ)

unrated = R.copy()
thresh = unrated > 0
unrated[thresh] = 0
unrated[~thresh] = 1

np.savetxt('nR.csv', nR*unrated, fmt='%0.3f', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)