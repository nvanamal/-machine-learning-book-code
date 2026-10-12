# RESIDUAL NETWORKS (RESNET) | EXAMPLE A
# Identity skip connection
# Dr. Nagendrababu Vanamala

import numpy as np
x = np.array([[1., -2.]])
W = np.array([[0.2, 0.1], [0.1, 0.2]])
F = np.maximum(0, x @ W)
y = x + F
print("Residual:", F)
print("Skip plus residual:", y)
