import numpy as np
from scipy.linalg import hadamard
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


def landmarks(X, m):
    n = X.shape[0]
    indx = np.random.choice(n, m, replace=False)
    Z = X[indx]
    return Z


def nystroem(X, samples=20, r=2, d=2):
    Z = landmarks(X, samples)
    C = (X.dot(Z.T)) ** d
    W = (Z.dot(Z.T)) ** d
    V, A, _ = np.linalg.svd(W)
    V = V[:, :r]
    S_inv = np.sqrt(1 / A[:r])
    L = C.dot(V) * S_inv
    K = L.dot(L.T)
    V, A, _ = np.linalg.svd(L.T.dot(L))
    A_inv = 1 / A
    X_out = L.dot(V) * np.sqrt(A_inv)
    return K, X_out


def nystroem_qr(X, samples=20, r=2, d=2):
    Z = landmarks(X, samples)
    C = (X.dot(Z.T)) ** d
    W = (Z.dot(Z.T)) ** d
    Q, R = np.linalg.qr(C)
    W_inv = W
    V, A, _ = np.linalg.svd(R.dot(W_inv) * (R.T))
    U = Q.dot(V[:, :r])
    L = U * (np.sqrt(A[:r]))
    K = L.dot(L.T)
    return K, U


def power_of_2(n):
    count = 0
    if (n and not (n & (n - 1))):
        return n
    while (n != 0):
        n >>= 1
        count += 1
    res = 1 << count
    return res


def efficient(X, r=2, l=6, d=2):
    K_origin = (X.dot(X.T)) ** d
    n = X.shape[0]
    size = power_of_2(n)
    samples = r + l
    p = size - n
    R = np.concatenate([np.zeros((p, samples)), np.identity(n)[:, np.random.permutation(n)[:samples]]])
    K = np.concatenate([np.concatenate([K_origin, np.zeros((p, n))]), np.zeros((size, p))], axis=1)
    D = np.diag(np.random.randint(2, size=size) * 2 - 1)
    H = hadamard(size)
    W = R.T.dot(H.dot(D.dot(K)))
    U, _, _ = np.linalg.svd(W.T)
    Q = U[:, :samples]
    A = D.dot(H.dot(R))
    B = (np.linalg.inv(A.T.dot(Q))).dot(W.dot(Q))
    U, E, _ = np.linalg.svd(B.T)
    U = U[:, :r]
    E = E[:r]
    X_out = ((np.sqrt(E) * U).T.dot(Q.T)).T[:X.shape[0]]
    K = X_out.dot(X_out.T)
    return K, X_out

def frobenius(K_hat, K):
    return np.linalg.norm(K-K_hat, 'fro')/np.linalg.norm(K, 'fro')

X, y = make_circles(n_samples=4000, factor=.3, noise=.05)
reds = y == 0
blues = y == 1

basis = 40
rank = 2

K_origin = (X.dot(X.T)) ** 2

K1, X1 = nystroem(X, samples=basis, r=rank, d=2)
K2, X2 = nystroem_qr(X, samples=basis, r=rank, d=2)
K3, X3 = efficient(X, r=2, l=6, d=2)

print("K: " + str(np.linalg.norm(K_origin, 'fro')))
print("K1: " + str(np.linalg.norm(K1, 'fro')))
print("K2: " + str(np.linalg.norm(K2, 'fro')))
print("K3: " + str(np.linalg.norm(K3, 'fro')))

print("K - K1: " + str(frobenius(K_origin, K1)))
print("K - K2: " + str(frobenius(K_origin, K2)))
print("K - K3: " + str(frobenius(K_origin, K3)))


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

plt.figure(4)
plt.title("Efficient Nystroem")
plt.scatter(X3[reds, 0], X3[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X3[blues, 0], X3[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.show()
