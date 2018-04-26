import itertools
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from src.nystroem import *

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import normalize

df = pd.read_csv('../data/segmentation.csv')

X =  df.values[:,1:].astype(np.float)
labels = df.values[:,:1].ravel()
le = preprocessing.LabelEncoder()
y = le.fit_transform(labels)
X = normalize(X)

print(X.shape)
print(np.unique(y))

m_subsamples = [10, 20, 30, 40, 50]

def add(x, y):
    return list(map(lambda a, b: a + b, x, y))

def accuracy(y_true, y_pred):
    best_score = 0.0
    for permut in itertools.permutations(range(7)):
        permut = np.array(permut)
        y_pred_permut = permut[y_pred]
        cur_result = accuracy_score(y_true, y_pred_permut)
        best_score = max(best_score, cur_result)
    return best_score

Acc1 = [0,0,0,0,0]
Acc2 = []
Acc3 = []

xrange = 10

for i in range(xrange):
    acc1 = []
    for m in m_subsamples:
        K1, X1 = nystroem(X, samples=m, r=2, d=2)
        kmeans1 = KMeans(n_clusters=2, random_state=i).fit_predict(X1)
        labels1 = kmeans1
        acc1.append(accuracy(y, labels1))
    Acc1 = add(Acc1, acc1)
    print(i, str(Acc1[:1])+"-"+str(Acc1[-1:]))
Acc1 = list(map(lambda x: x/xrange, Acc1))



for m in m_subsamples:
    K2, X2 = exact(X, samples=m, r=2, d=2)
    K3, X3 = efficient(X, r=2, l=5, d=2)

    kmeans2 = KMeans(n_clusters=2, random_state=0).fit_predict(X2)
    kmeans3 = KMeans(n_clusters=2, random_state=0).fit_predict(X3)

    labels2 = kmeans2
    labels3 = kmeans3
    Acc2.append(accuracy(y, labels2))
    Acc3.append(accuracy(y, labels3))

plt.figure(1)
plt.title("Accuracy")
plt.plot(m_subsamples, Acc1, marker='o', linestyle='--', color='r', label='Square')
plt.plot(m_subsamples, Acc2, marker='o', linestyle='--', color='b', label='Square')
plt.plot(m_subsamples, Acc3, marker='o', linestyle='--', color='g', label='Square')


plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()