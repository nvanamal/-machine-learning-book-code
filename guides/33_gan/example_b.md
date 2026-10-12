# GENERATIVE ADVERSARIAL NETWORK (GAN) — Example B

**Train a one-dimensional location GAN**  
Book page: **137** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/33_gan/example_b.py) · [All topics](../../README.md)

## What this example teaches

Trains a generator to produce samples and a discriminator to distinguish generated from real data. Their objectives interact. The original minimax objective is often replaced by alternatives for training stability.

## Python code

```python
import numpy as np
rng = np.random.default_rng(42)
a, b, theta = 0.0, 0.0, 0.0
for step in range(2000):
    real = rng.normal(2, 1, 128); fake = rng.normal(0, 1, 128) + theta
    dr = 1 / (1 + np.exp(-np.clip(a * real + b, -30, 30)))
    df = 1 / (1 + np.exp(-np.clip(a * fake + b, -30, 30)))
    a += 0.03 * (np.mean((1 - dr) * real) - np.mean(df * fake))
    b += 0.03 * (np.mean(1 - dr) - np.mean(df))
    z = rng.normal(0, 1, 128)
    dg = 1 / (1 + np.exp(-np.clip(a * (z + theta) + b, -30, 30)))
    theta += 0.03 * a * np.mean(1 - dg)
print("Learned generator location:", round(theta, 3))
print("Target location:", 2.0)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 3 | `a, b, theta = (0.0, 0.0, 0.0)` | Assign a, b, theta for the next step. |
| 4 | `for step in range(2000):` | Repeat episodes or steps of interaction. |
| 5 | `    real = rng.normal(2, 1, 128)` | Sample observations from the target distribution. |
| 6 | `    fake = rng.normal(0, 1, 128) + theta` | Sample outputs from the current generator. |
| 7 | `    dr = 1 / (1 + np.exp(-np.clip(a * real + b, -30, 30)))` | Compute discriminator scores for real samples. |
| 8 | `    df = 1 / (1 + np.exp(-np.clip(a * fake + b, -30, 30)))` | Compute discriminator scores for generated samples. |
| 9 | `    a += 0.03 * (np.mean((1 - dr) * real) - np.mean(df * fake))` | Update a. |
| 10 | `    b += 0.03 * (np.mean(1 - dr) - np.mean(df))` | Update b. |
| 11 | `    z = rng.normal(0, 1, 128)` | Create a shared latent signal for the features. |
| 12 | `    dg = (` | Score new generator samples for its update. |
| 13 | `        1 / (1 + np.exp(-np.clip(a * (z + theta) + b, -30, 30)))` | Supply arguments to the preceding operation. |
| 14 | `    )` | Close the preceding multi-line operation. |
| 15 | `    theta += 0.03 * a * np.mean(1 - dg)` | Update theta. |
| 16 | `print('Learned generator location:', round(theta, 3))` | Print the result shown in the output box. |
| 17 | `print('Target location:', 2.0)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Learned generator location: 1.999
Target location: 2.0
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The location generator and logistic discriminator update alternately. The learned location is compared with the target mean.

## Try it yourself

Repeat training with several seeds and plot location over iterations.

## Important check

This one-dimensional GAN tests a narrow distribution family and says nothing about deep image-generation quality.
