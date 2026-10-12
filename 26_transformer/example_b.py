# TRANSFORMER ARCHITECTURE | EXAMPLE B
# Layer normalization and a residual branch
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.array([[1., 2., 3.], [2., 4., 6.]])
F = np.maximum(0, X * 0.2)
y = X + F
normalized = (y - y.mean(axis=1, keepdims=True))
normalized /= np.sqrt(y.var(axis=1, keepdims=True) + 1e-5)
print("Normalized rows:", np.round(normalized, 3))
# Normalize features within each token, not across the batch.
