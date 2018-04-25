from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from src.nystroem import *

X, y = make_circles(n_samples=500, factor=.3, noise=.05)
reds = y == 0
blues = y == 1

m_subsamples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
m_subsamples_short = [10, 20, 30, 40, 50]
rank = 2
degree = 2
K_origin = (X.dot(X.T)) ** degree

K1K = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plt.figure(1)
plt.title("Kernel Approx. Error")

def add(x, y):
    return list(map(lambda a, b: a + b, x, y))

m_subsamples_1 = list(map(lambda x: x + 4, m_subsamples))
for i in range(4):
    k1k = []
    for m in m_subsamples_1:
        K1, X1 = nystroem(X, samples=m, r=rank, d=degree)
        frob1 = frobenius(K_origin, K1)
        k1k.append(frob1)
    K1K = add(K1K, k1k)

K1K = list(map(lambda x: x/4, K1K))


plt.plot(m_subsamples, K1K, marker='o', linestyle='--', color='r', label='Square')


K1K = []
K2K = []
K3K = []
for m in m_subsamples:
    print("----------------------------------")
    print("number of subsamples: "+str(m))
    print("----------------------------------")
    K1, X1 = nystroem(X, samples=m, r=rank, d=degree)
    K2, X2 = exact(X, samples=m, r=rank, d=degree)
    K3, X3 = efficient(X, r=2, l=5, d=degree)
    frob1 = frobenius(K_origin, K1)
    frob2 = frobenius(K_origin, K2)
    frob3 = frobenius(K_origin, K3)
    K1K.append(frob1)
    K2K.append(frob2)
    K3K.append(frob3)

    print("K - K1: " + str(frob1))
    print("K - K2: " + str(frob2))
    print("K - K3: " + str(frob3))
    print("----------------------------------")


plt.plot(m_subsamples, K2K, marker='x', linestyle='-', color='g', label='Square')
plt.plot(m_subsamples, K3K, marker='o', linestyle='--', color='b', label='Square')

plt.xlabel("m")
plt.ylabel("Kernel Approx. Error")
plt.show()

