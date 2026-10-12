# GENERATIVE ADVERSARIAL NETWORK (GAN) — Example C

**Generator loss differs from original minimax loss**  
Book page: **138** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/33_gan/example_c.py) · [All topics](../../README.md)

## What this example teaches

Trains a generator to produce samples and a discriminator to distinguish generated from real data. Their objectives interact. The original minimax objective is often replaced by alternatives for training stability.

## Python code

```python
import numpy as np
p_fake = np.array([0.1, 0.4, 0.8])
minimax = np.mean(np.log(1 - p_fake))
non_saturating = -np.mean(np.log(p_fake))
print("Minimax generator loss:", round(minimax, 4))
print("Non-saturating loss:", round(non_saturating, 4))
# Different objectives/gradient behaviour; values are not interchangeable.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `p_fake = np.array([0.1, 0.4, 0.8])` | Assign p_fake for the next step. |
| 3 | `minimax = np.mean(np.log(1 - p_fake))` | Assign minimax for the next step. |
| 4 | `non_saturating = -np.mean(np.log(p_fake))` | Assign non_saturating for the next step. |
| 5 | `print('Minimax generator loss:', round(minimax, 4))` | Print the result shown in the output box. |
| 6 | `print('Non-saturating loss:', round(non_saturating, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Minimax generator loss: -0.7419
Non-saturating loss: 1.1473
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The two printed losses correspond to different generator objectives and therefore have different numeric scales.

## Try it yourself

Plot both losses and their derivatives as the discriminator’s fake probability approaches zero.

## Important check

Do not compare the values as if they were the same objective or select training solely by these losses.
