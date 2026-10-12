# K-MEANS CLUSTERING — Example A

**Cluster and measure silhouette**  
Book page: **44** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/10_kmeans/example_a.py) · [All topics](../../README.md)

## What this example teaches

Divides observations into k groups using distances to centroids. It alternates between assigning points to the nearest centroid and updating each centroid to the average of its members. Cluster IDs have no intrinsic meaning.

## Python code

```python
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X, _ = make_blobs(n_samples=500, centers=3,
                 cluster_std=0.7, random_state=42)
model = make_pipeline(StandardScaler(),
    KMeans(n_clusters=3, n_init=10, random_state=42))
labels = model.fit_predict(X)
Xs = model.named_steps["standardscaler"].transform(X)
print("Silhouette:", silhouette_score(Xs, labels))
print("Cluster counts:", __import__("numpy").bincount(labels))
# Also inspect stability and domain usefulness.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_blobs` | Import make_blobs. |
| 2 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 3 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 4 | `from sklearn.cluster import KMeans` | Import KMeans. |
| 5 | `from sklearn.metrics import silhouette_score` | Import silhouette_score. |
| 6 | `(X, _) = (` | Load or generate the example dataset. |
| 7 | `    make_blobs(n_samples=500, centers=3, cluster_std=0.7,` | Supply arguments to the preceding operation. |
| 8 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 9 | `)` | Close the preceding multi-line operation. |
| 10 | `model = (` | Chain preprocessing with the estimator. |
| 11 | `    make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init=` | Standardize using statistics fitted on training data. |
| 12 | `    10, random_state=42))` | Fix the seed and other options for repeatability. |
| 13 | `)` | Close the preceding multi-line operation. |
| 14 | `labels = model.fit_predict(X)` | Fit the clustering model and assign groups. |
| 15 | `Xs = model.named_steps['standardscaler'].transform(X)` | Apply the previously fitted transformation. |
| 16 | `print('Silhouette:', silhouette_score(Xs, labels))` | Print the result shown in the output box. |
| 17 | `print('Cluster counts:', __import__('numpy').bincount(labels))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Silhouette: 0.8910207449839237
Cluster counts: [167 166 167]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Silhouette evaluates compactness and separation in the scaled feature space. Cluster counts show the group sizes.

## Try it yourself

Repeat the run with several seeds and compare assignments after allowing label permutations.

## Important check

Cluster IDs are arbitrary. Cluster 0 is not inherently better or more important than cluster 1.
