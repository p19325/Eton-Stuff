import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering

dataset = pd.read_csv('./datasets/Mall_Customers.csv')
X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

clusterers = [
    KMeans(n_clusters=5, init='k-means++', random_state=0, n_init='auto'),
    AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward'),
]

titles = ['K-Means', 'Agglomerative']

fig, axes = plt.subplots(1, len(clusterers), figsize=(14, 6))

palette = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown']

for i, clusterer in enumerate(clusterers):
    labels = clusterer.fit_predict(X)

    for lbl in np.unique(labels):
        axes[i].scatter(X[labels == lbl, 0], X[labels == lbl, 1], s=60, c=palette[lbl % len(palette)], label=f'Cluster {lbl + 1}', edgecolors='k', linewidths=0.5)

    # Centroids (only for K-Means)
    if hasattr(clusterer, 'cluster_centers_'):
        axes[i].scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1], s=200, c='yellow', marker='X', edgecolors='black', linewidths=1.0, label='Centroids')        

    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Annual Income (k$)')
    axes[i].set_ylabel('Spending Score (1-100)')
    axes[i].legend(frameon=False)

plt.tight_layout()
plt.show()
