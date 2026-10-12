# MULTILAYER PERCEPTRON (MLP) — Example A

**Forward pass through a two-layer network**  
Book page: **76** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/18_mlp/example_a.py) · [All topics](../../README.md)

## What this example teaches

A feed-forward neural network with fully connected hidden layers and nonlinear activations. Training adjusts weights by gradients propagated backward through the network.

## Python code

```python
import numpy as np
X = np.array([[1., -1.], [0.5, 2.]])
W1 = np.array([[1., 0.5], [-0.5, 1.]])
h = np.maximum(0, X @ W1)
W2 = np.array([[1.], [-1.]])
print("Hidden features:", h)
print("Predictions:", h @ W2)
# Fixed weights illustrate computation; no training here.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.array([[1.0, -1.0], [0.5, 2.0]])` | Build or transform the input feature matrix. |
| 3 | `W1 = np.array([[1.0, 0.5], [-0.5, 1.0]])` | Set input-to-hidden weight parameters. |
| 4 | `h = np.maximum(0, X @ W1)` | Compute or initialize the hidden representation. |
| 5 | `W2 = np.array([[1.0], [-1.0]])` | Set hidden-to-output weight parameters. |
| 6 | `print('Hidden features:', h)` | Print the result shown in the output box. |
| 7 | `print('Predictions:', h @ W2)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Hidden features: [[1.5  0.  ]
 [0.   2.25]]
Predictions: [[ 1.5 ]
 [-2.25]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The hidden activations are ReLU transformations, followed by an output projection using fixed weights.

## Try it yourself

Replace ReLU with tanh and compare the hidden values and predictions.

## Important check

A forward calculation is not a trained neural network or evidence of accuracy.
