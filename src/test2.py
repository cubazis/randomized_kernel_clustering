import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from src.nystrom import nystroem as origin
from src.nystrom import findRep
from src.nystrom import landmarks
from src.nystrom import lowrank
from src.nystrom import decom
from src.nystrom import efficient

basis = 40
rank = 2

X, y = make_circles(n_samples=400, factor=.3, noise=.05)
reds = y == 0
blues = y == 1

X_copy = np.copy(X)

K1, X1 = origin(X, n_samples=basis, r=rank, kernel="polynomial", degree=2)

X2 = lowrank(X, landmarks(X, basis), rank, 'Poly', [2, 0])
X3 = decom(X, landmarks(X, basis), rank, 'Poly', [2, 0])

K4, X4 = efficient(X)


plt.figure(1)
plt.title("Original space")
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.figure(2)
plt.title("Origin Nystroem (Pavel)")
plt.scatter(X1[reds, 0], X1[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X1[blues, 0], X1[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.figure(3)
plt.title("Low Rank Nystroem")
plt.scatter(X2[reds, 0], X2[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X2[blues, 0], X2[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.figure(4)
plt.title("QR Decomposition Nystroem")
plt.scatter(X3[reds, 0], X3[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X3[blues, 0], X3[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


plt.figure(5)
plt.title("Efficient Approach")
plt.scatter(X4[reds, 0], X4[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X4[blues, 0], X4[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.show()

