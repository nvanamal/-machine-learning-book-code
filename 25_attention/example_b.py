# ATTENTION MECHANISMS | EXAMPLE B
# Causal self-attention mask
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.eye(3)
scores = X @ X.T / np.sqrt(3)
scores[np.triu_indices(3, 1)] = -np.inf
A = np.exp(scores - scores.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
print("Causal weights:", np.round(A, 3))
print("Future-position weights are zero:", np.allclose(np.triu(A, 1), 0))
