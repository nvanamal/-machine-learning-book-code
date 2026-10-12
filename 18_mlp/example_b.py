# MULTILAYER PERCEPTRON (MLP) | EXAMPLE B
# One explicit backpropagation update
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.array([[1., 2.], [2., 1.]])
y = np.array([[1.], [0.]])
W1 = np.full((2, 3), 0.1); W2 = np.full((3, 1), 0.1)
h = np.maximum(0, X @ W1)
pred = h @ W2
error = 2 * (pred - y) / len(X)
gW2 = h.T @ error
gW1 = X.T @ ((error @ W2.T) * (h > 0))
W1 -= 0.1 * gW1; W2 -= 0.1 * gW2
print("MSE before:", np.mean((pred - y) ** 2))
print("MSE after:", np.mean((np.maximum(0, X @ W1) @ W2 - y) ** 2))
