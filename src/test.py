import numpy as np
from src.nystrom import pdist
from src.nystrom import findRep
import src.nystrom as nys

from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

A = np.array([[1, 2, 1], [3, 4, 2], [3, 6, 1]])

'''
X = np.array([[1, 2], [3, 4]])
print(nys.decom(X, findRep(X, 1), 1, 'Poly', [2, 0]))
'''

np.random.seed(42)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)


#kek = np.column_stack((X, y))


plt.figure()
#plt.subplot(2, 2, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1


X1 = X[reds]
X2 = X[blues]

"""
# Plot
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.show()
"""


nyst = Nystroem(kernel="polynomial", degree=20, n_components=20)
K_hat = nyst.fit_transform(X)

kmeans = KMeans(n_clusters=2)
labels_pred = kmeans.fit_predict(K_hat)

p_reds = labels_pred == 0
p_blues = labels_pred == 1

plt.scatter(K_hat[p_reds, 0], K_hat[p_reds, 1], c="red",
            s=10, edgecolor='k')
plt.scatter(K_hat[p_blues, 0], K_hat[p_blues, 1], c="blue",
            s=10, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()