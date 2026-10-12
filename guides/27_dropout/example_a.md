# DROPOUT — Example A

**Inverted dropout during training**  
Book page: **112** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/27_dropout/example_a.py) · [All topics](../../README.md)

## What this example teaches

Randomly removes selected activations during training to regularize a network. Inverted dropout scales survivors by 1/(1-p), preserving the expected activation. At standard inference it is disabled.

## Python code

```python
import numpy as np
rng = np.random.default_rng(42)
h = np.ones(8); p = 0.25
mask = rng.random(h.shape) >= p
train_h = h * mask / (1 - p)
print("Mask:", mask.astype(int))
print("Training activations:", train_h)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 3 | `h = np.ones(8)` | Compute or initialize the hidden representation. |
| 4 | `p = 0.25` | Assign p for the next step. |
| 5 | `mask = rng.random(h.shape) >= p` | Select retained activations or labeled examples. |
| 6 | `train_h = h * mask / (1 - p)` | Mask and rescale training activations. |
| 7 | `print('Mask:', mask.astype(int))` | Print the result shown in the output box. |
| 8 | `print('Training activations:', train_h)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Mask: [1 1 1 1 0 1 1 1]
Training activations: [1.33333333 1.33333333 1.33333333 1.33333333 0.         1.33333333
 1.33333333 1.33333333]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Retained activations are scaled by 1/(1-p). Zero entries show the sampled training mask.

## Try it yourself

Rerun with different seeds and compare masks while keeping the dropout probability fixed.

## Important check

Inverted dropout is random during training; its mask is not a learned feature selector.
