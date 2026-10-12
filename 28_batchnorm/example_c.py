# BATCH NORMALIZATION | EXAMPLE C
# Running statistics for inference
# Dr. Nagendrababu Vanamala

import numpy as np
running_mean, running_var = 0.0, 1.0
momentum = 0.1  # fraction of new statistics
batch = np.array([1., 2., 3.])
running_mean = (1 - momentum) * running_mean + momentum * batch.mean()
running_var = (1 - momentum) * running_var + momentum * batch.var()
query = np.array([2.])
y = (query - running_mean) / np.sqrt(running_var + 1e-5)
print("Running mean/variance:", running_mean, running_var)
print("Inference value:", y)
# Toy convention; frameworks can use different variance estimators.
