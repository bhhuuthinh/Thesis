import numpy as np
from numpy import linalg as LA

x = [9, 15, 25, 14, 10, 18, 0, 16, 5, 19, 16, 20]
y = [39, 56, 93, 61, 50, 75, 32, 85, 42, 70, 66, 80]

Cov = np.cov(x, y)

w, v = LA.eig(Cov)

X = [x, y]
X = np.array(X)
X = X.T

print(X)
v1 = np.array(v[:, 0])
v1 = v1.T

Y = np.matmul(X, v1)

print(Y)