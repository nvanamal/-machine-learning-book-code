# SOFTMAX REGRESSION | EXAMPLE A
# Stable softmax probabilities
# Dr. Nagendrababu Vanamala

import numpy as np
scores = np.array([[2., 1., -1.]])
shifted = scores - scores.max(axis=1, keepdims=True)
p = np.exp(shifted)
p /= p.sum(axis=1, keepdims=True)
print("Probabilities:", np.round(p, 4))
print("Row sums:", p.sum(axis=1))
