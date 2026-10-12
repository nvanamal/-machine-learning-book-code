# RECURRENT NEURAL NETWORK (RNN) — Example B

**Batch sequence shapes**  
Book page: **89** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/21_rnn/example_b.py) · [All topics](../../README.md)

## What this example teaches

Processes a sequence by repeatedly updating a hidden state with shared parameters. The state summarizes prior inputs and supports time-dependent outputs. Training commonly uses backpropagation through time.

## Python code

```python
import numpy as np
rng = np.random.default_rng(42)
X = rng.normal(size=(2, 4, 3))
Wx = rng.normal(scale=0.2, size=(3, 5))
Wh = np.eye(5) * 0.5
h = np.zeros((2, 5))
for t in range(4):
    h = np.tanh(X[:, t] @ Wx + h @ Wh)
print("Batch/time/input shape:", X.shape)
print("Final hidden shape:", h.shape)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 3 | `X = rng.normal(size=(2, 4, 3))` | Build or transform the input feature matrix. |
| 4 | `Wx = rng.normal(scale=0.2, size=(3, 5))` | Set input-to-state weight parameters. |
| 5 | `Wh = np.eye(5) * 0.5` | Set recurrent state weight parameters. |
| 6 | `h = np.zeros((2, 5))` | Compute or initialize the hidden representation. |
| 7 | `for t in range(4):` | Repeat episodes or steps of interaction. |
| 8 | `    h = np.tanh(X[:, t] @ Wx + h @ Wh)` | Compute or initialize the hidden representation. |
| 9 | `print('Batch/time/input shape:', X.shape)` | Print the result shown in the output box. |
| 10 | `print('Final hidden shape:', h.shape)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Batch/time/input shape: (2, 4, 3)
Final hidden shape: (2, 5)
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The tensor contains batch, time and input-feature axes. The final hidden tensor has one state per batch member.

## Try it yourself

Increase the hidden width and confirm every matrix multiplication still has compatible dimensions.

## Important check

Reset states between independent sequences; do not leak one subject’s history into another.
