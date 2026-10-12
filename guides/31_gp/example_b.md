# GAUSSIAN PROCESS REGRESSION — Example B

**Kernel length scale changes similarity**  
Book page: **129** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/31_gp/example_b.py) · [All topics](../../README.md)

## What this example teaches

Places a probability distribution over functions. A mean function and covariance kernel define prior relationships between inputs. Conditioning on noisy observations gives a predictive mean and uncertainty under the model.

## Python code

```python
import numpy as np
from sklearn.gaussian_process.kernels import RBF
X = np.array([[0.], [1.], [2.]])
for length in [0.3, 2.0]:
    K = RBF(length)(X)
    print("Length scale:", length)
    print(np.round(K, 3))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.gaussian_process.kernels import RBF` | Import RBF. |
| 3 | `X = np.array([[0.0], [1.0], [2.0]])` | Build or transform the input feature matrix. |
| 4 | `for length in [0.3, 2.0]:` | Repeat episodes or steps of interaction. |
| 5 | `    K = RBF(length)(X)` | Assign K for the next step. |
| 6 | `    print('Length scale:', length)` | Print the result shown in the output box. |
| 7 | `    print(np.round(K, 3))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Length scale: 0.3
[[1.    0.004 0.   ]
 [0.004 1.    0.004]
 [0.    0.004 1.   ]]
Length scale: 2.0
[[1.    0.882 0.607]
 [0.882 1.    0.882]
 [0.607 0.882 1.   ]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The RBF length scale determines how quickly similarity decays as input distance grows.

## Try it yourself

Compare near-diagonal versus broad covariance structure for several length scales.

## Important check

A kernel matrix encodes assumed similarity; it does not automatically match the physical process.
