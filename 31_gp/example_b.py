# GAUSSIAN PROCESS REGRESSION | EXAMPLE B
# Kernel length scale changes similarity
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.gaussian_process.kernels import RBF
X = np.array([[0.], [1.], [2.]])
for length in [0.3, 2.0]:
    K = RBF(length)(X)
    print("Length scale:", length)
    print(np.round(K, 3))
