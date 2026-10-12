# SUPERVISED LEARNING — Example B

**Regression uses numeric labels**  
Book page: **53** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/12_supervised/example_b.py) · [All topics](../../README.md)

## What this example teaches

Trains a model using labeled examples (x,y). Classification predicts categories; regression predicts numbers. An algorithm such as a tree or neural network implements this paradigm. Labels specify the training target.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=8,
    n_informative=4, noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(StandardScaler(), Ridge(alpha=1))
model.fit(Xtr, ytr)
print("MAE:", mean_absolute_error(yte, model.predict(Xte)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_regression` | Import a numeric-target data generator. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 5 | `from sklearn.linear_model import Ridge` | Import Ridge. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=450, n_features=8, n_informative=4` | Set feature count and useful-signal dimensions. |
| 9 | `    , noise=15, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 12 | `model = make_pipeline(StandardScaler(), Ridge(alpha=1))` | Chain preprocessing with the estimator. |
| 13 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 14 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 11.915051018905867
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Regression requires numeric labels and an error metric in the target’s scale.

## Try it yourself

Add a mean-prediction baseline and compare MAE using the same test rows.

## Important check

A regressor should not be evaluated with classification accuracy.
