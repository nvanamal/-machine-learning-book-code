# CONVOLUTIONAL NEURAL NETWORK (CNN) | EXAMPLE A
# A 2-D filter on an image patch
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.arange(16).reshape(4, 4)
K = np.array([[1., 0.], [0., -1.]])
Y = np.empty((3, 3))
for i in range(3):
    for j in range(3):
        Y[i, j] = np.sum(X[i:i+2, j:j+2] * K)
print("Feature map:", Y)
