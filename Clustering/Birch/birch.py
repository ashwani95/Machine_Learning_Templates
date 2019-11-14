# Birch Clustering Algorithm

# Importing Libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# Random dataset from sklearn. Feel free to use your own dataset
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch

# Creating blobs
X, clusters = make_blobs(n_samples=450, centers=6, cluster_std=0.70, random_state=0)

# Applying birch algorithm and fitting model
brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
brc.fit(X)

labels = brc.predict(X)

#Plotting the graph
plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')

plt.show(block=True)

