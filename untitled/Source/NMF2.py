import numpy as np
from sklearn.decomposition import NMF
from numpy import linalg as LA

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):

    rated = R.copy()
    thresh = rated > 0
    rated[thresh] = 1
    rated[~thresh] = 0

    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        hypothesis = R - np.dot(P, Q)
        hypothesis = hypothesis * rated

        e = LA.norm(hypothesis)**2 \
            + (beta/2)*(LA.norm(P)**2 + LA.norm(Q)**2)

        print(e)

        if e < 0.001:
            break
    return P, Q.T

R = [[0, 4.0, 2.0, 0],
    [4.0, 0, 4.0, 0],
    [0, 5.0, 0, 2.0],
    [0, 3.0, 4.0, 1.0]]

R = np.array(R)

N, M = R.shape
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K, alpha=10e-3,beta=10e-3)
nR = np.dot(nP, nQ.T)
print(nP)
print(nQ)
print(np.round(nR))