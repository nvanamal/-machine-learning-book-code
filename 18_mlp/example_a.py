# MULTILAYER PERCEPTRON (MLP) | EXAMPLE A
# Forward pass through a two-layer network
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.array([[1., -1.], [0.5, 2.]])
W1 = np.array([[1., 0.5], [-0.5, 1.]])
h = np.maximum(0, X @ W1)
W2 = np.array([[1.], [-1.]])
print("Hidden features:", h)
print("Predictions:", h @ W2)
# Fixed weights illustrate computation; no training here.
