from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from src.nystroem import *

X, y = make_circles(n_samples=4000, factor=.3, noise=.05)
reds = y == 0
blues = y == 1

basis = 40
rank = 2

K_origin = (X.dot(X.T)) ** 2

K1, X1 = nystroem(X, samples=20, r=rank, d=2)
K4, X4 = nystroem(X, samples=100, r=rank, d=2)
K2, X2 = exact(X, samples=basis, r=rank, d=2)
K3, X3 = efficient(X, r=2, l=5, d=2)

print("K: " + str(np.linalg.norm(K_origin, 'fro')))
print("K1: " + str(np.linalg.norm(K1, 'fro')))
print("K2: " + str(np.linalg.norm(K2, 'fro')))
print("K3: " + str(np.linalg.norm(K3, 'fro')))

print("K - K1: " + str(frobenius(K_origin, K1)))
print("K - K2: " + str(frobenius(K_origin, K2)))
print("K - K3: " + str(frobenius(K_origin, K3)))

X_origin = X
X_origin[:, 1] = np.abs(X_origin[:, 1])

plt.figure(1)
plt.title("Original space")
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.figure(2)
plt.title("Original Nystroem (m = 20)")
plt.scatter(X1[reds, 0], X1[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X1[blues, 0], X1[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.figure(3)
plt.title("Exact Nystroem")
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

plt.figure(5)
plt.title("Original Nystroem (m = 100)")
plt.scatter(X4[reds, 0], X4[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X4[blues, 0], X4[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.show()
