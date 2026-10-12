# ATTENTION MECHANISMS | EXAMPLE A
# Scaled dot-product attention
# Dr. Nagendrababu Vanamala

import numpy as np
Q = np.array([[1., 0.]])
K = np.eye(2); V = np.array([[10.], [20.]])
scores = Q @ K.T / np.sqrt(2)
A = np.exp(scores - scores.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
print("Weights:", np.round(A, 4))
print("Context:", np.round(A @ V, 4))
