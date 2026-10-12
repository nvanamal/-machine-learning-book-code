# TRANSFER LEARNING / FINE-TUNING | EXAMPLE C
# Fine-tune a transferred hidden weight matrix
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.array([[1., 0.], [0., 1.]])
y = np.array([[1.], [0.]])
W = np.array([[0.2, 0.1], [0.1, 0.2]])  # transferred toy weights
V = np.full((2, 1), 0.1)  # a new head
for step in range(80):
    h = np.maximum(0, X @ W); pred = h @ V
    error = 2 * (pred - y) / len(X)
    gV = h.T @ error
    gW = X.T @ ((error @ V.T) * (h > 0))
    V -= 0.1 * gV; W -= 0.02 * gW
print("Fine-tuned predictions:", np.round(np.maximum(0, X @ W) @ V, 3))
# Both the new head and transferred layer are updated.
