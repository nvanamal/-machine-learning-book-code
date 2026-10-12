# TRANSFORMER ARCHITECTURE | EXAMPLE C
# A miniature attention-plus-feedforward block
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.array([[1., 0.], [0., 1.]])
scores = X @ X.T / np.sqrt(2)
A = np.exp(scores - scores.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
y = X + A @ X
norm = (y - y.mean(axis=1, keepdims=True)) / np.sqrt(y.var(axis=1, keepdims=True) + 1e-5)
F = np.maximum(0, norm @ np.eye(2)) @ np.eye(2)
print("Toy block output:", np.round(norm + F, 3))
# Fixed projections, one head; illustrative sublayers only.
