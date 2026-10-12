# CONVOLUTIONAL NEURAL NETWORK (CNN) — Example B

**Max pooling downsamples a feature map**  
Book page: **81** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/19_cnn/example_b.py) · [All topics](../../README.md)

## What this example teaches

A neural architecture with local filters whose weights are shared across spatial positions. Filters produce feature maps; pooling or stride changes resolution. A learned head maps features to a task output.

## Python code

```python
import numpy as np
X = np.arange(16).reshape(4, 4)
pooled = X.reshape(2, 2, 2, 2).max(axis=(1, 3))
print("Input shape:", X.shape)
print("Pooled values:", pooled)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.arange(16).reshape(4, 4)` | Build or transform the input feature matrix. |
| 3 | `pooled = X.reshape(2, 2, 2, 2).max(axis=(1, 3))` | Assign pooled for the next step. |
| 4 | `print('Input shape:', X.shape)` | Print the result shown in the output box. |
| 5 | `print('Pooled values:', pooled)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Input shape: (4, 4)
Pooled values: [[ 5  7]
 [13 15]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Each non-overlapping 2-by-2 region contributes its maximum; the 4-by-4 map becomes 2-by-2.

## Try it yourself

Compute average pooling over the same regions and compare the outputs.

## Important check

Verify reshape axes: pooling across the wrong axes mixes unrelated regions.
