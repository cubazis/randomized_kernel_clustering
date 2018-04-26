from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from src.nystroem import *
from src.tools import accuracy


plt.figure(1)
plt.title("Original")
Acc1 = [0,0,0,0,0]
Acc2 = [0,0,0,0,0]
Acc3 = [0,0,0,0,0]
Acc4 = [0,0,0,0,0]
m_subsamples = [10, 20, 30, 40, 50]


def add(x, y):
    return list(map(lambda a, b: a + b, x, y))

X, y = make_circles(n_samples=300, factor=.3, noise=.05)

xrange = 100

for i in range(xrange):
    acc1 = []
    acc2 = []
    acc3 = []
    for m in m_subsamples:

        K1, X1 = nystroem(X, samples=m, r=2, d=2)
        K2, X2 = exact(X, samples=m, r=2, d=2)
        K3, X3 = efficient(X, r=2, l=5, d=2)
        j = i
        kmeans1 = KMeans(n_clusters=2, random_state=j%10).fit(K1)
        kmeans2 = KMeans(n_clusters=2, random_state=j).fit(K2)
        kmeans3 = KMeans(n_clusters=2, random_state=j).fit(K3)

        labels1 = kmeans1.labels_
        labels2 = kmeans2.labels_
        labels3 = kmeans3.labels_
        acc1.append(accuracy(y, labels1))
        acc2.append(accuracy(y, labels2))
        acc3.append(accuracy(y, labels3))

    Acc1 = add(Acc1, acc1)
    Acc2 = add(Acc2, acc2)
    Acc3 = add(Acc3, acc3)
    print(i, str(Acc1[:1])+"-"+str(Acc1[-1:]) , str(Acc2[:1])+"-"+str(Acc2[-1:]) , str(Acc3[:1])+"-"+str(Acc3[-1:]) )

Acc1 = list(map(lambda x: x/xrange, Acc1))
Acc2 = list(map(lambda x: x/xrange, Acc2))
Acc3 = list(map(lambda x: x/xrange, Acc3))

plt.plot(m_subsamples, Acc1, marker='o', linestyle='--', color='r', label='Square')
plt.plot(m_subsamples, Acc2, marker='o', linestyle='--', color='b', label='Square')
plt.plot(m_subsamples, Acc3, marker='o', linestyle='--', color='g', label='Square')


plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()








