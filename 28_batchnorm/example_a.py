# BATCH NORMALIZATION | EXAMPLE A
# Normalize each feature in a batch
# Dr. Nagendrababu Vanamala

import numpy as np
X = np.array([[1., 10.], [2., 20.], [3., 30.]])
mean = X.mean(axis=0); var = X.var(axis=0)
Y = (X - mean) / np.sqrt(var + 1e-5)
print("Normalized means:", np.round(Y.mean(axis=0), 5))
print("Normalized variances:", np.round(Y.var(axis=0), 5))
