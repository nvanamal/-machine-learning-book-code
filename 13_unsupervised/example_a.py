# UNSUPERVISED LEARNING | EXAMPLE A
# K-means discovers groups
# Dr. Nagendrababu Vanamala

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
X, _ = make_blobs(n_samples=400, centers=3,
                 random_state=42)
# Ignore generated labels: training has no task target.
Xs = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = model.fit_predict(Xs)
print("Silhouette:", silhouette_score(Xs, clusters))
print("First assignments:", clusters[:10])
# Inspect clusters and compare other values of k.
