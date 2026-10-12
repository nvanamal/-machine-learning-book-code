# GENERATIVE ADVERSARIAL NETWORK (GAN) — Example A

**Discriminator probabilities**  
Book page: **136** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/33_gan/example_a.py) · [All topics](../../README.md)

## What this example teaches

Trains a generator to produce samples and a discriminator to distinguish generated from real data. Their objectives interact. The original minimax objective is often replaced by alternatives for training stability.

## Python code

```python
import numpy as np
x = np.array([-1., 0., 1.])
a, b = 1.0, 0.0
p_real = 1 / (1 + np.exp(-(a * x + b)))
print("Discriminator probabilities:", np.round(p_real, 3))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `x = np.array([-1.0, 0.0, 1.0])` | Assign x for the next step. |
| 3 | `a, b = (1.0, 0.0)` | Assign a, b for the next step. |
| 4 | `p_real = 1 / (1 + np.exp(-(a * x + b)))` | Assign p_real for the next step. |
| 5 | `print('Discriminator probabilities:', np.round(p_real, 3))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Discriminator probabilities: [0.269 0.5   0.731]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The discriminator maps one-dimensional inputs to probabilities through a logistic score with fixed parameters.

## Try it yourself

Reverse the sign of a and inspect how the ranking changes.

## Important check

These probabilities are a toy discriminator calculation, not verified generated-sample quality.
