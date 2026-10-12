# RECURRENT NEURAL NETWORK (RNN) — Example C

**Gradient propagation through time**  
Book page: **90** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/21_rnn/example_c.py) · [All topics](../../README.md)

## What this example teaches

Processes a sequence by repeatedly updating a hidden state with shared parameters. The state summarizes prior inputs and supports time-dependent outputs. Training commonly uses backpropagation through time.

## Python code

```python
import numpy as np
for scale in [0.5, 1.2]:
    gradient = 1.0
    for step in range(20):
        gradient *= scale
    print("Jacobian factor", scale, "after 20 steps:", round(gradient, 6))
# Simplified repeated Jacobian product, not a full RNN training loop.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `for scale in [0.5, 1.2]:` | Repeat episodes or steps of interaction. |
| 3 | `    gradient = 1.0` | Compute the loss gradient at this step. |
| 4 | `    for step in range(20):` | Repeat episodes or steps of interaction. |
| 5 | `        gradient *= scale` | Update gradient. |
| 6 | `    (` | Print the result shown in the output box. |
| 7 | `        print('Jacobian factor', scale, 'after 20 steps:', round` | Supply arguments to the preceding operation. |
| 8 | `        (gradient, 6))` | Supply arguments to the preceding operation. |
| 9 | `    )` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Jacobian factor 0.5 after 20 steps: 1e-06
Jacobian factor 1.2 after 20 steps: 38.3376
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Repeated Jacobian factors below one vanish; factors above one grow. This is a simplified gradient-propagation example.

## Try it yourself

Compare 5, 20 and 50 steps for both factors.

## Important check

Real RNN gradients also depend on activations and matrices, not just one scalar factor.
