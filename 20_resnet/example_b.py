# RESIDUAL NETWORKS (RESNET) | EXAMPLE B
# Projection aligns feature dimensions
# Dr. Nagendrababu Vanamala

import numpy as np
x = np.array([[1., 2.]])
P = np.array([[1., 0., 1.], [0., 1., 1.]])
W = np.full((2, 3), 0.1)
y = x @ P + np.maximum(0, x @ W)
print("Input/output shapes:", x.shape, y.shape)
print("Projected residual output:", y)
