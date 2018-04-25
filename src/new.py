import numpy as np
from scipy.linalg import hadamard
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


def landmarks(X, m):
    n = X.shape[0]
    indx = np.random.choice(n, m, replace=False)
    Z = X[indx]
    return Z


def nystroem(X, n_samples=20, r=2, d=2):
    Z = landmarks(X, n_samples)
    C = (X.dot(Z.T)) ** d
    W = (Z.dot(Z.T)) ** d
    V, A, _ = np.linalg.svd(W)
    V = V[:, :r]
    S_inv = np.sqrt(1 / A[:r])
    L = C.dot(V) * S_inv
    K = L.dot(L.T)
    V, A, _ = np.linalg.svd(L.T @ L)
    A_inv = 1 / A
    Y = L.dot(V) * np.sqrt(A_inv)
    return K, Y

def nystroem_qr(X, n_samples=20, r=2, d=2):
    Z = landmarks(X, n_samples)
    C = (X.dot(Z.T)) ** d
    W = (Z.dot(Z.T)) ** d
    Q, R = np.linalg.qr(C)
    W_inv = W
    V, A, _ = np.linalg.svd(R.dot(W_inv) * (R.T))
    U = Q.dot(V[:, :r])
    L = U * (np.sqrt(A[:r]))
    K = L.dot(L.T)
    return K, U

X, y = make_circles(n_samples=400, factor=.3, noise=.05)
reds = y == 0
blues = y == 1

basis = 40
rank = 2

K1, X1 = nystroem(X, n_samples=basis, r=rank, d=2)
K2, X2 = nystroem_qr(X, n_samples=basis, r=rank, d=2)

print("K: " + str(np.linalg.norm((X @ X.T) ** 2, 'fro')))
print("K1: " + str(np.linalg.norm(K1, 'fro')))
print("K2: " + str(np.linalg.norm(K2, 'fro')))

origin = (X @ X.T) ** 2
print(origin.shape)
print((K1).shape)
print((K2).shape)

plt.figure(1)
plt.title("Original space")
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.figure(2)
plt.title("Original Nystroem")
plt.scatter(X1[reds, 0], X1[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X1[blues, 0], X1[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.figure(3)
plt.title("Nystroem QR")
plt.scatter(X2[reds, 0], X2[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X2[blues, 0], X2[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.show()
