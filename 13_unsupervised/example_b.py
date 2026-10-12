# UNSUPERVISED LEARNING | EXAMPLE B
# Density clustering with DBSCAN
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
X, _ = make_moons(n_samples=400, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)
print("Non-noise cluster count:", len(set(labels) - {-1}))
print("Noise points:", np.sum(labels == -1))
# -1 is DBSCAN noise; cluster IDs are arbitrary.
