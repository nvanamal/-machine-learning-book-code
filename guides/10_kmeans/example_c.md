# K-MEANS CLUSTERING — Example C

**Compare candidate cluster counts**  
Book page: **46** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/10_kmeans/example_c.py) · [All topics](../../README.md)

## What this example teaches

Divides observations into k groups using distances to centroids. It alternates between assigning points to the nearest centroid and updating each centroid to the average of its members. Cluster IDs have no intrinsic meaning.

## Python code

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X, _ = make_blobs(n_samples=400, centers=3, random_state=42)
X = StandardScaler().fit_transform(X)
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, n_init=10,
                    random_state=42).fit_predict(X)
    print(k, silhouette_score(X, labels))
# Internal exploratory scores; also check stability and meaning.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_blobs` | Import make_blobs. |
| 2 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 3 | `from sklearn.cluster import KMeans` | Import KMeans. |
| 4 | `from sklearn.metrics import silhouette_score` | Import silhouette_score. |
| 5 | `X, _ = make_blobs(n_samples=400, centers=3, random_state=42)` | Load or generate the example dataset. |
| 6 | `X = StandardScaler().fit_transform(X)` | Learn the transformation and apply it. |
| 7 | `for k in [2, 3, 4, 5]:` | Repeat candidate values to compare clustering. |
| 8 | `    labels = (` | Fit the clustering model and assign groups. |
| 9 | `        KMeans(n_clusters=k, n_init=10, random_state=42).` | Fix the seed and other options for repeatability. |
| 10 | `        fit_predict(X)` | Supply arguments to the preceding operation. |
| 11 | `    )` | Close the preceding multi-line operation. |
| 12 | `    print(k, silhouette_score(X, labels))` | Print the result shown in the output box. |

## Recorded output from the book

```text
2 0.6869887512028753
3 0.845857065650404
4 0.6838842323497352
5 0.5269559223367607
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Candidate k values are compared using an internal separation score, not supplied class labels.

## Try it yourself

Check cluster sizes, inertia and stability alongside silhouette rather than relying on one score.

## Important check

A visually meaningful segmentation can differ from the k maximizing an internal metric.
