# RESIDUAL NETWORKS (RESNET) | EXAMPLE C
# Identity path appears in the gradient
# Dr. Nagendrababu Vanamala

import numpy as np
W = np.array([[0.2, 0.1], [0.1, 0.2]])
# Linear residual toy: y = x + W @ x.
J = np.eye(2) + W
upstream = np.ones(2)
print("Jacobian:", J)
print("Input gradient:", J.T @ upstream)
