# CONVOLUTIONAL NEURAL NETWORK (CNN) | EXAMPLE C
# A 1-D convolutional spectral feature
# Dr. Nagendrababu Vanamala

import numpy as np
signal = np.array([0., 0., 1., 3., 1., 0., 0.])
kernel = np.array([-1., 0., 1.])
feature = np.array([np.sum(signal[i:i+3] * kernel) for i in range(5)])
print("Local gradient feature:", feature)
print("Largest response:", np.max(np.abs(feature)))
# Shared fixed kernel; learning filters requires backpropagation.
