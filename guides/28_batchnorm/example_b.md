# BATCH NORMALIZATION — Example B

**Learned affine scale and shift**  
Book page: **117** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/28_batchnorm/example_b.py) · [All topics](../../README.md)

## What this example teaches

Normalizes activations using minibatch statistics during training, followed by learned scale and shift. In evaluation it usually uses accumulated running statistics. It is distinct from input scaling and LayerNorm.

## Python code

```python
import numpy as np
normalized = np.array([[-1., 1.], [1., -1.]])
gamma = np.array([2., 0.5]); beta = np.array([1., -1.])
y = gamma * normalized + beta
print("Affine-normalized activations:", y)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `normalized = np.array([[-1.0, 1.0], [1.0, -1.0]])` | Center or normalize the feature activations. |
| 3 | `gamma = np.array([2.0, 0.5])` | Assign gamma for the next step. |
| 4 | `beta = np.array([1.0, -1.0])` | Assign beta for the next step. |
| 5 | `y = gamma * normalized + beta` | Define or generate the corresponding targets. |
| 6 | `print('Affine-normalized activations:', y)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Affine-normalized activations: [[-1.  -0.5]
 [ 3.  -1.5]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Learned gamma and beta can rescale and shift normalized activations, allowing the layer to recover useful ranges.

## Try it yourself

Set gamma to zero and verify that the output becomes beta.

## Important check

Normalization does not force every downstream representation to have zero mean and unit variance.
