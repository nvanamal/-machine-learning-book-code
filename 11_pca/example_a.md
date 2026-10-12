# PRINCIPAL COMPONENT ANALYSIS (PCA) — Example A

**Project onto two PCA components**  
Book page: **48** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/11_pca/example_a.py) · [All topics](../../README.md)

## What this example teaches

Finds orthogonal directions that capture high variance. Projecting onto the leading directions reduces feature count. PCA uses no class labels and is not a classifier. High variance need not mean high target relevance.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X, _ = load_iris(return_X_y=True)
Xtr, Xte = train_test_split(X, test_size=0.2,
                           random_state=42)
model = make_pipeline(StandardScaler(),
                      PCA(n_components=2))
Ztr = model.fit_transform(Xtr)
Zte = model.transform(Xte)
print("Shapes:", Ztr.shape, Zte.shape)
print("Variance ratios:",
      model.named_steps["pca"].explained_variance_ratio_)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 5 | `from sklearn.decomposition import PCA` | Import PCA. |
| 6 | `X, _ = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 7 | `Xtr, Xte = train_test_split(X, test_size=0.2, random_state=42)` | Reserve unseen data for testing. |
| 8 | `model = make_pipeline(StandardScaler(), PCA(n_components=2))` | Chain preprocessing with the estimator. |
| 9 | `Ztr = model.fit_transform(Xtr)` | Learn the transformation and apply it. |
| 10 | `Zte = model.transform(Xte)` | Apply the previously fitted transformation. |
| 11 | `print('Shapes:', Ztr.shape, Zte.shape)` | Print the result shown in the output box. |
| 12 | `(` | Print the result shown in the output box. |
| 13 | `    print('Variance ratios:', model.named_steps['pca'].` | Supply arguments to the preceding operation. |
| 14 | `    explained_variance_ratio_)` | Supply arguments to the preceding operation. |
| 15 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Shapes: (120, 2) (30, 2)
Variance ratios: [0.72551423 0.23000922]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The reduced arrays contain two component coordinates per sample. Variance ratios summarize retained training variation.

## Try it yourself

Compare two components with a 95% explained-variance setting and inspect the resulting dimensions.

## Important check

PCA captures variance, not target relevance. A low-variance direction may still matter for prediction.
