from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from src.nystrom import *


basis = 20
rank = 2

X, y = make_circles(n_samples=100, factor=.3, noise=.05)

reds = y == 0
blues = y == 1

K, X4 = efficient(X)

plt.figure()
plt.title("Efficient Approach")
plt.scatter(X4[reds, 0], X4[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X4[blues, 0], X4[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.show()