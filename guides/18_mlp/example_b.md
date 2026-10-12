# MULTILAYER PERCEPTRON (MLP) — Example B

**One explicit backpropagation update**  
Book page: **77** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/18_mlp/example_b.py) · [All topics](../../README.md)

## What this example teaches

A feed-forward neural network with fully connected hidden layers and nonlinear activations. Training adjusts weights by gradients propagated backward through the network.

## Python code

```python
import numpy as np
X = np.array([[1., 2.], [2., 1.]])
y = np.array([[1.], [0.]])
W1 = np.full((2, 3), 0.1); W2 = np.full((3, 1), 0.1)
h = np.maximum(0, X @ W1)
pred = h @ W2
error = 2 * (pred - y) / len(X)
gW2 = h.T @ error
gW1 = X.T @ ((error @ W2.T) * (h > 0))
W1 -= 0.1 * gW1; W2 -= 0.1 * gW2
print("MSE before:", np.mean((pred - y) ** 2))
print("MSE after:", np.mean((np.maximum(0, X @ W1) @ W2 - y) ** 2))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.array([[1.0, 2.0], [2.0, 1.0]])` | Build or transform the input feature matrix. |
| 3 | `y = np.array([[1.0], [0.0]])` | Define or generate the corresponding targets. |
| 4 | `W1 = np.full((2, 3), 0.1)` | Set input-to-hidden weight parameters. |
| 5 | `W2 = np.full((3, 1), 0.1)` | Set hidden-to-output weight parameters. |
| 6 | `h = np.maximum(0, X @ W1)` | Compute or initialize the hidden representation. |
| 7 | `pred = h @ W2` | Store predictions for evaluation. |
| 8 | `error = 2 * (pred - y) / len(X)` | Compute a prediction residual or loss derivative. |
| 9 | `gW2 = h.T @ error` | Compute the output-layer loss gradient. |
| 10 | `gW1 = X.T @ (error @ W2.T * (h > 0))` | Compute the input-layer loss gradient. |
| 11 | `W1 -= 0.1 * gW1` | Update W1. |
| 12 | `W2 -= 0.1 * gW2` | Update W2. |
| 13 | `print('MSE before:', np.mean((pred - y) ** 2))` | Print the result shown in the output box. |
| 14 | `(` | Print the result shown in the output box. |
| 15 | `    print('MSE after:', np.mean((np.maximum(0, X @ W1) @ W2 - y)` | Supply arguments to the preceding operation. |
| 16 | `    ** 2))` | Supply arguments to the preceding operation. |
| 17 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
MSE before: 0.4180999999999999
MSE after: 0.3880604490605684
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Two weight gradients are computed before either matrix is updated. The displayed MSE shows one optimization step.

## Try it yourself

Change the learning rate and observe whether the single-step loss improves.

## Important check

Compute gradients using the old weights; updating one layer early can alter the intended gradient.
