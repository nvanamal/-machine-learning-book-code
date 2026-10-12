# SOFTMAX REGRESSION — Example B

**Cross-entropy and score gradient**  
Book page: **73** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/17_softmax/example_b.py) · [All topics](../../README.md)

## What this example teaches

A linear multiclass classifier. One score is computed for each class; softmax converts scores into a probability vector summing to one. It is a neural network with an output layer and no hidden layer.

## Python code

```python
import numpy as np
z = np.array([1., 2., 0.])
p = np.exp(z - z.max()); p /= p.sum()
y = 1
loss = -np.log(p[y])
grad = p.copy(); grad[y] -= 1
print("Loss:", round(loss, 4))
print("Gradient:", np.round(grad, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `z = np.array([1.0, 2.0, 0.0])` | Create a shared latent signal for the features. |
| 3 | `p = np.exp(z - z.max())` | Assign p for the next step. |
| 4 | `p /= p.sum()` | Update p. |
| 5 | `y = 1` | Define or generate the corresponding targets. |
| 6 | `loss = -np.log(p[y])` | Assign loss for the next step. |
| 7 | `grad = p.copy()` | Assign grad for the next step. |
| 8 | `grad[y] -= 1` | Update grad[y]. |
| 9 | `print('Loss:', round(loss, 4))` | Print the result shown in the output box. |
| 10 | `print('Gradient:', np.round(grad, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Loss: 0.4076
Gradient: [ 0.2447 -0.3348  0.09  ]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The score gradient equals predicted probability minus the one-hot target. Its entries sum to approximately zero.

## Try it yourself

Change the correct-class score and inspect how loss and gradient change.

## Important check

This derivative is with respect to scores, not automatically every network weight.
