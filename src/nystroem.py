import numpy as np
from scipy.linalg import hadamard


def landmarks(X, m):
    n = X.shape[0]
    indx = np.random.choice(n, m, replace=False)
    Z = X[indx]
    return Z


def nystroem(X, samples=10, r=2, d=2):
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


def nystroem_qr(X, samples=10, r=2, d=2):
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


def efficient(X, r=2, l=5, d=2):
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

def exact(X, samples=10, r=2, d=2):
    K_origin = (X.dot(X.T)) ** d
    U, S, _ = np.linalg.svd(K_origin)
    U = U[:,:r]
    S = S[:r]
    X_out = U * (np.sqrt(S))
    K = X_out.dot(X_out.T)
    return K, X_out


def frobenius(K_origin, K):
    return np.linalg.norm(K_origin - K, 'fro') / np.linalg.norm(K_origin, 'fro')
