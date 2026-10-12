# PRINCIPAL COMPONENT ANALYSIS (PCA) — Example B

**Held-out reconstruction from two components**  
Book page: **49** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/11_pca/example_b.py) · [All topics](../../README.md)

## What this example teaches

Finds orthogonal directions that capture high variance. Projecting onto the leading directions reduces feature count. PCA uses no class labels and is not a classifier. High variance need not mean high target relevance.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
X, _ = load_iris(return_X_y=True)
Xtr, Xte = train_test_split(X, random_state=42)
scaler = StandardScaler().fit(Xtr)
A, B = scaler.transform(Xtr), scaler.transform(Xte)
pca = PCA(n_components=2).fit(A)
reconstructed = pca.inverse_transform(pca.transform(B))
print("Standardized reconstruction MSE:",
      mean_squared_error(B, reconstructed))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 4 | `from sklearn.decomposition import PCA` | Import PCA. |
| 5 | `from sklearn.metrics import mean_squared_error` | Import squared-error evaluation. |
| 6 | `X, _ = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 7 | `Xtr, Xte = train_test_split(X, random_state=42)` | Reserve unseen data for testing. |
| 8 | `scaler = StandardScaler().fit(Xtr)` | Fit model parameters using training inputs. |
| 9 | `A, B = (scaler.transform(Xtr), scaler.transform(Xte))` | Store transformed training or test features. |
| 10 | `pca = PCA(n_components=2).fit(A)` | Fit model parameters using training inputs. |
| 11 | `reconstructed = pca.inverse_transform(pca.transform(B))` | Reconstruct from the reduced representation. |
| 12 | `(` | Print the result shown in the output box. |
| 13 | `    print('Standardized reconstruction MSE:', mean_squared_error` | Compute mean squared prediction or reconstruction error. |
| 14 | `    (B, reconstructed))` | Supply arguments to the preceding operation. |
| 15 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Standardized reconstruction MSE: 0.03238100830540001
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Reconstruction error measures what is lost when projecting onto two components, in standardized feature units.

## Try it yourself

Repeat with one, two and three components; compare held-out reconstruction MSE.

## Important check

Low reconstruction error does not prove that a downstream classifier will perform well.
