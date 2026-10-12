# PRINCIPAL COMPONENT ANALYSIS (PCA) — Example C

**PCA inside a predictive validation pipeline**  
Book page: **50** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/11_pca/example_c.py) · [All topics](../../README.md)

## What this example teaches

Finds orthogonal directions that capture high variance. Projecting onto the leading directions reduces feature count. PCA uses no class labels and is not a classifier. High variance need not mean high target relevance.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), PCA(n_components=2),
                      LogisticRegression(max_iter=1000))
print("CV accuracy:", cross_val_score(model, Xtr, ytr, cv=3).mean())
model.fit(Xtr, ytr)
print("Test accuracy:", model.score(Xte, yte))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 2 | `from sklearn.model_selection import (` | Import train_test_split, cross_val_score. |
| 3 | `    train_test_split,` | Supply arguments to the preceding operation. |
| 4 | `    cross_val_score,` | Supply arguments to the preceding operation. |
| 5 | `)` | Close the preceding multi-line operation. |
| 6 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 7 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 8 | `from sklearn.decomposition import PCA` | Import PCA. |
| 9 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
| 10 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 11 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 12 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 13 | `)` | Close the preceding multi-line operation. |
| 14 | `model = (` | Chain preprocessing with the estimator. |
| 15 | `    make_pipeline(StandardScaler(), PCA(n_components=2),` | Standardize using statistics fitted on training data. |
| 16 | `    LogisticRegression(max_iter=1000))` | Choose the estimator and its tuning options. |
| 17 | `)` | Close the preceding multi-line operation. |
| 18 | `(` | Print the result shown in the output box. |
| 19 | `    print('CV accuracy:', cross_val_score(model, Xtr, ytr, cv=3)` | Supply arguments to the preceding operation. |
| 20 | `    .mean())` | Supply arguments to the preceding operation. |
| 21 | `)` | Close the preceding multi-line operation. |
| 22 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 23 | `print('Test accuracy:', model.score(Xte, yte))` | Print the result shown in the output box. |

## Recorded output from the book

```text
CV accuracy: 0.8572783309625415
Test accuracy: 0.8947368421052632
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

PCA is fitted inside each CV training fold through the pipeline; test samples are transformed using fitted steps.

## Try it yourself

Compare this pipeline with the same classifier without PCA using identical splits.

## Important check

Do not fit PCA on test rows before validation, even though PCA does not use class labels.
