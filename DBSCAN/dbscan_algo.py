import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X=pd.read_csv('dbscan_data.csv')
Y=pd.read_csv('dbscan_data.csv')

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
#print(db.labels_)
#print(db.core_sample_indices_)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
#print(core_samples_mask)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = ['y', 'b', 'g','r']

plt.subplot(1,2,1)

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'

    class_member_mask = (labels == k)
    #print(class_member_mask)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)

plt.title('number of clusters: %d' %n_clusters_)

noisy=[]

for i in range(len(labels)):
    if(labels[i]==-1):
        t=tuple(Y.iloc[i,0:2])
        noisy.append(t)
        Y=Y.drop(i,axis=0)

print("The Noisy tuples that were removed : ")
for i in noisy:
    print(i)

db2 = DBSCAN(eps=0.3, min_samples=10).fit(Y)
core_samples_mask2 = np.zeros_like(db2.labels_, dtype=bool)
core_samples_mask2[db2.core_sample_indices_] = True
labels2 = db2.labels_

n_clusters_2 = len(set(labels2)) - (1 if -1 in labels2 else 0)

unique_labels2 = set(labels2)
colors2 = ['y', 'b', 'g','r']

plt.subplot(1,2,2)

for k, col in zip(unique_labels2, colors2):
    if k == -1:
        col = 'k'

    class_member_mask2 = (labels2 == k)
    #print(class_member_mask)

    xy = Y[class_member_mask2 & core_samples_mask2]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)

    xy = Y[class_member_mask2 & ~core_samples_mask2]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)

plt.title('number of clusters_(Noise_Removed): %d' %n_clusters_)


plt.show()

