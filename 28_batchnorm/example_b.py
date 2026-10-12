# BATCH NORMALIZATION | EXAMPLE B
# Learned affine scale and shift
# Dr. Nagendrababu Vanamala

import numpy as np
normalized = np.array([[-1., 1.], [1., -1.]])
gamma = np.array([2., 0.5]); beta = np.array([1., -1.])
y = gamma * normalized + beta
print("Affine-normalized activations:", y)
