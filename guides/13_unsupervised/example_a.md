# UNSUPERVISED LEARNING — Example A

**K-means discovers groups**  
Book page: **56** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/13_unsupervised/example_a.py) · [All topics](../../README.md)

## What this example teaches

Uses inputs without supplied task labels to learn patterns or representations. Includes clustering, dimensionality reduction and density estimation. Objectives differ across algorithms; there is no universal unsupervised loss.

## Python code

```python
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
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_blobs` | Import make_blobs. |
| 2 | `from sklearn.cluster import KMeans` | Import KMeans. |
| 3 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 4 | `from sklearn.metrics import silhouette_score` | Import silhouette_score. |
| 5 | `X, _ = make_blobs(n_samples=400, centers=3, random_state=42)` | Load or generate the example dataset. |
| 6 | `Xs = StandardScaler().fit_transform(X)` | Learn the transformation and apply it. |
| 7 | `model = KMeans(n_clusters=3, n_init=10, random_state=42)` | Create or fit the selected estimator. |
| 8 | `clusters = model.fit_predict(Xs)` | Fit the clustering model and assign groups. |
| 9 | `print('Silhouette:', silhouette_score(Xs, clusters))` | Print the result shown in the output box. |
| 10 | `print('First assignments:', clusters[:10])` | Print the result shown in the output box. |

## Recorded output from the book

```text
Silhouette: 0.845857065650404
First assignments: [1 0 1 2 0 0 0 1 0 2]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

K-means uses feature structure without a training target. Silhouette evaluates the discovered grouping internally.

## Try it yourself

Change feature scaling and inspect whether the cluster interpretation remains meaningful.

## Important check

A discovered cluster is not automatically a real-world class or diagnosis.
