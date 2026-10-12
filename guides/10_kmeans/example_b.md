# K-MEANS CLUSTERING — Example B

**Color grouping as image quantization**  
Book page: **45** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/10_kmeans/example_b.py) · [All topics](../../README.md)

## What this example teaches

Divides observations into k groups using distances to centroids. It alternates between assigning points to the nearest centroid and updating each centroid to the average of its members. Cluster IDs have no intrinsic meaning.

## Python code

```python
import numpy as np
from sklearn.cluster import KMeans
rng = np.random.default_rng(42)
# A synthetic RGB image; each pixel is a 3-feature vector.
image = rng.random((20, 20, 3))
pixels = image.reshape(-1, 3)
model = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = model.fit_predict(pixels)
compressed = model.cluster_centers_[labels].reshape(image.shape)
print("Original/compressed shapes:", image.shape, compressed.shape)
print("Number of palette colors:", len(model.cluster_centers_))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.cluster import KMeans` | Import KMeans. |
| 3 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 4 | `image = rng.random((20, 20, 3))` | Create a synthetic RGB image array. |
| 5 | `pixels = image.reshape(-1, 3)` | Reshape each pixel into a three-value feature row. |
| 6 | `model = KMeans(n_clusters=4, n_init=10, random_state=42)` | Create or fit the selected estimator. |
| 7 | `labels = model.fit_predict(pixels)` | Fit the clustering model and assign groups. |
| 8 | `compressed = model.cluster_centers_[labels].reshape(image.shape)` | Replace each pixel by its cluster-centroid color. |
| 9 | `(` | Print the result shown in the output box. |
| 10 | `    print('Original/compressed shapes:', image.shape, compressed` | Supply arguments to the preceding operation. |
| 11 | `    .shape)` | Supply arguments to the preceding operation. |
| 12 | `)` | Close the preceding multi-line operation. |
| 13 | `print('Number of palette colors:', len(model.cluster_centers_))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Original/compressed shapes: (20, 20, 3) (20, 20, 3)
Number of palette colors: 4
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Each RGB pixel is replaced by its assigned centroid color. The shape is retained while the palette is reduced.

## Try it yourself

Increase the palette from four to eight centroids and compare reconstruction error.

## Important check

This example creates a synthetic image. Its compression behaviour is not a natural-image quality benchmark.
