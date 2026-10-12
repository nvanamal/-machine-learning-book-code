# CONVOLUTIONAL NEURAL NETWORK (CNN) | EXAMPLE B
# Max pooling downsamples a feature map
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.arange(16).reshape(4, 4)
pooled = X.reshape(2, 2, 2, 2).max(axis=(1, 3))
print("Input shape:", X.shape)
print("Pooled values:", pooled)
