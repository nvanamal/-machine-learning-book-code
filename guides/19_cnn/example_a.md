# CONVOLUTIONAL NEURAL NETWORK (CNN) — Example A

**A 2-D filter on an image patch**  
Book page: **80** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/19_cnn/example_a.py) · [All topics](../../README.md)

## What this example teaches

A neural architecture with local filters whose weights are shared across spatial positions. Filters produce feature maps; pooling or stride changes resolution. A learned head maps features to a task output.

## Python code

```python
import numpy as np
X = np.arange(16).reshape(4, 4)
K = np.array([[1., 0.], [0., -1.]])
Y = np.empty((3, 3))
for i in range(3):
    for j in range(3):
        Y[i, j] = np.sum(X[i:i+2, j:j+2] * K)
print("Feature map:", Y)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.arange(16).reshape(4, 4)` | Build or transform the input feature matrix. |
| 3 | `K = np.array([[1.0, 0.0], [0.0, -1.0]])` | Assign K for the next step. |
| 4 | `Y = np.empty((3, 3))` | Assign Y for the next step. |
| 5 | `for i in range(3):` | Repeat episodes or steps of interaction. |
| 6 | `    for j in range(3):` | Repeat episodes or steps of interaction. |
| 7 | `        Y[i, j] = np.sum(X[i:i + 2, j:j + 2] * K)` | Assign Y[i, j] for the next step. |
| 8 | `print('Feature map:', Y)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Feature map: [[-5. -5. -5.]
 [-5. -5. -5.]
 [-5. -5. -5.]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A shared 2-by-2 filter is applied at each valid image position. A 4-by-4 input therefore produces a 3-by-3 map.

## Try it yourself

Try a vertical-edge kernel and inspect which positions respond strongly.

## Important check

The code performs cross-correlation, the operation commonly called convolution in neural layers.
